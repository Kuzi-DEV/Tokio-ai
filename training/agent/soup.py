"""Average several LoRA adapters of the same base into one adapter, exactly.

    python training/agent/soup.py OUT ADAPTER1 ADAPTER2 ...

mean_i(B_i A_i) = [B_1/k ... B_k/k] [A_1; ...; A_k], so the average of k
rank-r adapters is one rank-k*r adapter. lora_alpha is scaled by k so the
scale alpha/r is unchanged. Averaging A and B separately would not do this.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main(out, *srcs):
    cfgs = [json.loads((Path(s) / "adapter_config.json").read_text()) for s in srcs]
    r, alpha = cfgs[0]["r"], cfgs[0]["lora_alpha"]
    assert all(c["r"] == r and c["lora_alpha"] == alpha and not c.get("use_rslora") for c in cfgs)
    ws = [load_file(str(Path(s) / "adapter_model.safetensors")) for s in srcs]
    k = len(ws)
    merged = {}
    for name in ws[0]:
        ts = [w[name].float() for w in ws]
        if "lora_A" in name:
            merged[name] = torch.cat(ts, 0)
        elif "lora_B" in name:
            merged[name] = torch.cat([t / k for t in ts], 1)
        else:
            raise ValueError(f"unexpected tensor {name}")
    merged = {n: t.to(ws[0][n].dtype).contiguous() for n, t in merged.items()}
    Path(out).mkdir(parents=True, exist_ok=True)
    save_file(merged, str(Path(out) / "adapter_model.safetensors"))
    cfg = dict(cfgs[0], r=r * k, lora_alpha=alpha * k)
    (Path(out) / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for f in Path(srcs[0]).iterdir():
        if f.name not in ("adapter_model.safetensors", "adapter_config.json") and f.is_file():
            (Path(out) / f.name).write_bytes(f.read_bytes())
    print(f"souped {k} adapters -> {out} (rank {r * k})")


if __name__ == "__main__":
    main(*sys.argv[1:])
