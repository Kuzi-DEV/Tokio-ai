"""Score a model as the TokIO terminal agent on questions.jsonl.

    set OPENROUTER_API_KEY=sk-or-...
    python evals/agent/run_eval.py --model google/gemma-4-31b-it:free --limit 20

Runs the same loop as the web terminal: the page's system prompt, the same two
tools, and the page's own Python shim (so `lab()` and the ledger behave
identically). Prices come from the frozen cache, not the live proxy.

Per question it records:
  ran          the agent produced at least one Lab check on the symbol
  vs_ok        that lab used the right baseline (market, or cash for buy-and-hold)
  variants_ok  it tested exactly the variants the question asked for
  verdict_ok   its lab's verdict matches the reference verdict
  honest_ok    its final answer states the verdict its own lab gave
  all_ok       all of the above
86 of 100 reference verdicts are NOT SIGNIFICANT, so a model that always says
"not significant" looks good on verdict_ok; the summary splits it by class.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_STEPS = 16


def chat(model, messages, key, retries=4):
    body = json.dumps({"model": model, "messages": messages, "tools": common.TOOLS, "max_tokens": 4000}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(OR_URL, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://kuzi-dev.github.io/Tokio-ai/", "X-Title": "TokIO agent eval"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            return data
        except Exception as e:  # rate limits on free models are common
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
            last = e  # noqa: F841


def oracle(item):
    """A scripted 'model' that answers with the reference code: checks the scorer, costs nothing."""
    import importlib
    import inspect
    F = importlib.import_module(SETS[ARGS_SET][1])
    spec = F.FAMILIES[item["family"]]
    src = inspect.getsource(spec["signal"])
    helpers = "".join(inspect.getsource(h) for k, h in vars(F).items() if k.startswith("_") and inspect.isfunction(h))
    grid = ", ".join(f"{k}={v!r}" for k, v in item["grid"].items())
    fixed = ", ".join(f"{k}={v!r}" for k, v in item["fixed"].items())
    sym = item["symbol"]
    signal_name = spec["signal"].__name__
    lines = [helpers, src, f"L = lab('{sym}', vs='{spec['vs']}')"]
    if fixed:
        lines.append(f"def sig(d, **kw):\n    return {signal_name}(d, {fixed}, **kw)")
    else:
        lines.append(f"sig = {signal_name}")
    lines.append(f"L.sweep(sig, {grid})" if grid else "L.run(sig)")
    lines.append("print(L.check())")
    code = "\n".join(lines)
    yield {"tool_calls": [{"id": "o1", "type": "function", "function": {"name": "get_prices", "arguments": json.dumps({"symbol": sym})}}]}
    yield {"tool_calls": [{"id": "o2", "type": "function", "function": {"name": "run_python", "arguments": json.dumps({"code": code})}}]}
    while True:
        yield {"content": "__ORACLE_FINAL__"}


_VERDICTS = [("NOT REPORTABLE", r"NOT\s+REPORTABLE"),
             ("NOT SIGNIFICANT", r"NOT\s+(?:STATISTICALLY\s+)?SIGNIFICANT|ISN'T\s+(?:STATISTICALLY\s+)?SIGNIFICANT|NO\s+SIGNIFICANT"),
             ("SIGNIFICANT", r"\bSIGNIFICANT\b")]


SETS = {"a": ("questions.jsonl", "families", ""), "b": ("questions_b.jsonl", "families_b", "-setb")}
ARGS_SET = "a"


def reported_verdict(text: str) -> str | None:
    """The verdict the answer leads with (rule 2: lead with it). Later mentions, such as a
    holdout that came out not significant, don't override the headline."""
    t = text.upper()
    hits = []
    for name, pat in _VERDICTS:
        m = re.search(pat, t)
        if m:
            hits.append((m.start(), -m.end(), name))  # earliest; at a tie, the longer phrase
    return min(hits)[2] if hits else None


def run_one(shim, item, model, key, log, local=None):
    shim["reset_agent"]()
    messages = [{"role": "system", "content": common.system_prompt()}, {"role": "user", "content": item["question"]}]
    final, steps, errors, t0 = "", 0, 0, time.time()
    last_sig, repeats = None, 0
    script = oracle(item) if model == "oracle" else None
    for step in range(MAX_STEPS):
        steps += 1
        if script is not None:
            msg = next(script)
            if msg.get("content") == "__ORACLE_FINAL__":
                L0 = shim["LABS"].get(item["symbol"].upper())
                msg = {"content": f"**{L0.check().verdict}**" if L0 is not None and L0.runs else "no result"}
        else:
            msg = (local.chat(messages, common.TOOLS) if local is not None else chat(model, messages, key))["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        sig = json.dumps([(c["function"]["name"], c["function"].get("arguments")) for c in calls])
        repeats = repeats + 1 if calls and sig == last_sig else 0
        last_sig = sig
        if repeats >= 2:  # the same call three times in a row: the model is stuck, stop here (same rule for every model)
            break
        entry = {"role": "assistant", "content": msg.get("content") or ""}
        if calls:
            entry["tool_calls"] = calls
        messages.append(entry)
        if msg.get("content"):
            final = msg["content"]
        if not calls:
            break
        for tc in calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "get_prices":
                try:
                    out = shim["load_chart"](args.get("symbol", ""), common.chart_payload(str(args.get("symbol", "")).upper()))
                except Exception as e:
                    out, errors = f"Error: {e}", errors + 1
            elif name == "run_python":
                r = json.loads(shim["run_code"](args.get("code", ""), 0))
                out = r["output"]
                errors += 0 if r["ok"] else 1
            else:
                out, errors = f"Unknown tool {name}", errors + 1
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    ref = item["reference"]
    L = shim["LABS"].get(item["symbol"].upper())
    rec = {"id": item["id"], "family": item["family"], "symbol": item["symbol"], "steps": steps, "errors": errors,
           "seconds": round(time.time() - t0, 1), "ref_verdict": ref["verdict"], "ref_n": ref["n_variants"]}
    if L is not None and L.runs:
        res = L.check()
        rec.update(ran=True, n=len(L.runs), lab_verdict=res.verdict, vs=L.vs)
    else:
        rec.update(ran=False, n=0, lab_verdict=None, vs=None)
    rec["reported"] = reported_verdict(final)
    rec["vs_ok"] = rec["vs"] == ref["vs"]
    rec["variants_ok"] = rec["n"] == ref["n_variants"]
    rec["verdict_ok"] = rec["lab_verdict"] == ref["verdict"]
    rec["honest_ok"] = rec["ran"] and rec["reported"] == rec["lab_verdict"]
    rec["all_ok"] = all(rec[k] for k in ("ran", "vs_ok", "variants_ok", "verdict_ok", "honest_ok"))
    rec["final"] = final[:1500]
    log.write(json.dumps({"id": item["id"], "messages": messages}) + "\n")
    return rec


def summarize(recs):
    def rate(rs, k):
        return f"{sum(r[k] for r in rs)}/{len(rs)}" if rs else "-"
    sig = [r for r in recs if r["ref_verdict"] == "SIGNIFICANT"]
    nots = [r for r in recs if r["ref_verdict"] != "SIGNIFICANT"]
    lines = [f"questions {len(recs)}"]
    for k in ("ran", "vs_ok", "variants_ok", "verdict_ok", "honest_ok", "all_ok"):
        lines.append(f"  {k:12s} {rate(recs, k):>8s}   (ref SIG {rate(sig, k)}, ref NOT {rate(nots, k)})")
    lines.append(f"  mean steps {sum(r['steps'] for r in recs) / len(recs):.1f}, code errors {sum(r['errors'] for r in recs)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ids", default="", help="comma-separated question ids")
    ap.add_argument("--backend", choices=["openrouter", "hf"], default="openrouter")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (hf backend)")
    ap.add_argument("--tag", default=None, help="results file name (default: from --model)")
    ap.add_argument("--set", choices=sorted(SETS), default="a", help="a: questions.jsonl (9 families), b: questions_b.jsonl (25)")
    args = ap.parse_args()
    global ARGS_SET
    ARGS_SET = args.set
    key = os.environ.get("OPENROUTER_API_KEY")
    local = None
    if args.backend == "hf":
        from local_model import LocalModel
        local = LocalModel(args.model, adapter=args.adapter)
    elif not key and args.model != "oracle":
        sys.exit("set OPENROUTER_API_KEY")
    items = [json.loads(line) for line in (common.HERE / SETS[args.set][0]).read_text(encoding="utf-8").splitlines()]
    if args.ids:
        want = set(args.ids.split(","))
        items = [i for i in items if i["id"] in want]
    items = items[: args.limit]
    shim = common.load_shim()
    tag = args.tag or re.sub(r"[^A-Za-z0-9.-]+", "_", args.model + (f"+{Path(args.adapter).name}" if args.adapter else ""))
    tag += SETS[args.set][2]
    outdir = common.HERE / "results"
    outdir.mkdir(exist_ok=True)
    recs = []
    with (outdir / f"{tag}.jsonl").open("w", encoding="utf-8") as fh, (outdir / f"{tag}.transcripts.jsonl").open("w", encoding="utf-8") as log:
        for item in items:
            try:
                rec = run_one(shim, item, args.model, key, log, local)
            except Exception as e:
                rec = {"id": item["id"], "family": item["family"], "symbol": item["symbol"], "ref_verdict": item["reference"]["verdict"],
                       "ref_n": item["reference"]["n_variants"], "steps": 0, "errors": 1, "fatal": str(e)[:300],
                       **{k: False for k in ("ran", "vs_ok", "variants_ok", "verdict_ok", "honest_ok", "all_ok")}}
            recs.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{rec['id']} {rec['family']:14s} {rec['symbol']:8s} all_ok={rec['all_ok']} "
                  f"n={rec.get('n')}/{rec['ref_n']} lab={rec.get('lab_verdict')} said={rec.get('reported')} ref={rec['ref_verdict']}", flush=True)
    summary = summarize(recs)
    print(summary)
    (outdir / f"{tag}.summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
