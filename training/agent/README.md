# Agent training data

Fine-tuning data for a small TokIO agent model, scored later on `evals/agent`.

```bash
python training/agent/make_data.py --per-pair 3 --workers 4
```

Every trajectory is **executed**, not written by an LLM. The generator runs each one through the web terminal's own Python shim (the code `evals/agent/run_eval.py` scores), so:

- every tool output in the data is the real output;
- every final answer is built from the Lab's actual result (verdict first, corrected p, Sharpe, haircut, probability of overfitting, sample);
- a trajectory is dropped unless the Lab counted exactly the variants the question asked for, with the right baseline.

## What's in it

`data/train.jsonl` (523) and `data/dev.jsonl` (100, held-out tickers). One record per line: `{"messages", "tools", "meta"}`, in OpenAI chat format with tool calls.

| | |
|---|---|
| Strategy types | 9 (EMA cross, MACD, volatility filter, down streaks, near 52-week high, stochastic, seasonality, dip buying, long/short trend) |
| Tickers | 20 train + 4 dev |
| SIGNIFICANT in-sample | 72 (12%); each one then runs `final_test` once on the holdout, and most fail there (the answer says so) |
| Baseline | 75% vs market, 25% vs cash (when the question asks about absolute profit) |
| Vague questions | 95: the agent states the standard version it picked (rule 5) |
| Error recovery | 99: first attempt uses `d["Close"]`, gets the real KeyError, fixes it (rule 7) |

## Leakage guard

The strategy types and tickers do not overlap the test set at all (`make_data.py` asserts it). Index proxies of test tickers (`^GSPC` and similar) are left out. The model is scored on strategies and tickers it has never seen.

## Fine-tune v1 (2026-11-19): Qwen3-1.7B, QLoRA r=16, 2 epochs, RTX 3050 Ti 4 GB

```bash
python training/agent/finetune.py --model Qwen/Qwen3-1.7B --epochs 2 --out training/agent/out/lora-q3-1.7b
python evals/agent/run_eval.py --model Qwen/Qwen3-1.7B --backend hf --adapter training/agent/out/lora-q3-1.7b
```

~2 hours, 2.4 GB peak GPU memory. Dev loss 3.39 -> 0.029.

| test set (100, unseen strategies and tickers) | base | fine-tuned |
|---|---|---|
| ran a Lab check | 0 | 72 |
| right baseline | 0 | 72 |
| right number of variants | 0 | 65 |
| right verdict | 0 | 50 |
| reported its own verdict honestly | 0 | 61 |
| **all correct** | **0** | **43** |

By family (all correct): Bollinger 9/11, momentum 9/11, SMA cross 7/11, above-SMA 7/12, buy-and-hold 4/11, turn-of-month 3/10, RSI 3/11, Donchian 1/12, weekday 0/11.

What went wrong, from the transcripts:
- **Memorized templates.** Train loss reached 0.009. On unseen strategy types it reuses training code (a weekday question gets the training `seasonal` function; RSI and buy-and-hold get `long_short`). Nine families are not enough variety.
- **No real debugging.** It only learned the `Close` -> `close` fix; other errors get the same code resubmitted.
- **Invented numbers.** On a NOT REPORTABLE result it answered NOT SIGNIFICANT with a made-up p-value. The data has no NOT REPORTABLE examples, so the answer template filled the gap.

Next: many more families and code styles, varied real errors with real fixes, NOT REPORTABLE cases, 1 epoch.
