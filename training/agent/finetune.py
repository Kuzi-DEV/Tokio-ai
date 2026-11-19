"""QLoRA fine-tune of a small chat model on training/agent/data (fits a 4 GB GPU).

    python training/agent/finetune.py --model Qwen/Qwen3-1.7B --epochs 2 --out training/agent/out/lora

One example per assistant turn: the prompt is the conversation so far,
rendered exactly as at inference (chat template, tools, generation prompt,
thinking off), and the loss covers only that turn. The output head runs only
on the target positions, so a long tool-filled prompt doesn't need a
(sequence x vocabulary) logit tensor in memory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent


def render(tok, messages, tools, gen_prompt):
    return tok.apply_chat_template(messages, tools=tools, add_generation_prompt=gen_prompt, tokenize=False, enable_thinking=False)


def examples(tok, path, max_len):
    out, skipped = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        msgs, tools = rec["messages"], rec["tools"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prompt = render(tok, msgs[:i], tools, True)
            full = render(tok, msgs[: i + 1], tools, False)
            if not full.startswith(prompt):
                skipped += 1
                continue
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            t_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(t_ids) > max_len:
                skipped += 1
                continue
            out.append((p_ids, t_ids))
    return out, skipped


def loss_on(model, head, p_ids, t_ids, device):
    ids = torch.tensor([p_ids + t_ids], device=device)
    hidden = model(input_ids=ids).last_hidden_state  # (1, L, d)
    # position j predicts token j+1: targets are t_ids, predicted from positions len(p)-1 .. L-2
    h = hidden[0, len(p_ids) - 1: len(p_ids) + len(t_ids) - 1]
    logits = head(h).float()
    return Fn.cross_entropy(logits, torch.tensor(t_ids, device=device), reduction="sum"), len(t_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--out", default=str(HERE / "out" / "lora"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import bitsandbytes as bnb

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    train, sk = examples(tok, HERE / "data" / "train.jsonl", args.max_len)
    dev, skd = examples(tok, HERE / "data" / "dev.jsonl", args.max_len)
    lens = sorted(len(p) + len(t) for p, t in train)
    print(f"train {len(train)} turns (skipped {sk}), dev {len(dev)} (skipped {skd}); "
          f"tokens median {lens[len(lens) // 2]}, max {lens[-1]}", flush=True)

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=q, dtype=torch.bfloat16, device_map="cuda")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    body = model.base_model.model.model  # decoder stack, returns hidden states
    head = model.base_model.model.lm_head
    device = next(model.parameters()).device

    opt = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(train) * args.epochs / args.accum)
    warm = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    def dev_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for p, t in dev:
                l, k = loss_on(body, head, p, t, device)
                tot, n = tot + l.item(), n + k
        model.train()
        return tot / n

    print(f"dev loss before: {dev_loss():.4f}", flush=True)
    model.train()
    order, step, t0 = [], 0, time.time()
    n_items = int(len(train) * args.epochs)
    while len(order) < n_items:
        e = list(range(len(train)))
        random.shuffle(e)
        order += e
    order = order[:n_items]
    run_loss, run_tok = 0.0, 0
    for i, idx in enumerate(order, 1):
        p, t = train[idx]
        l, k = loss_on(body, head, p, t, device)
        (l / k / args.accum).backward()
        run_loss, run_tok = run_loss + l.item(), run_tok + k
        if i % args.accum == 0 or i == len(order):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                el = time.time() - t0
                print(f"step {step}/{total_steps} loss {run_loss / run_tok:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{el / 60:.1f} min, eta {el / i * (len(order) - i) / 60:.0f} min, "
                      f"gpu {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
                run_loss, run_tok = 0.0, 0
        if i % len(train) == 0:
            print(f"epoch {i // len(train)} dev loss {dev_loss():.4f}", flush=True)
            model.save_pretrained(args.out)
    print(f"dev loss after: {dev_loss():.4f}", flush=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
