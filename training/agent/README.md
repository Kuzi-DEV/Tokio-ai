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

## Fine-tune v2 (2026-11-20): v2 data (30 families, 1,194 trajectories), 1 epoch

```bash
python training/agent/finetune.py --model Qwen/Qwen3-1.7B --epochs 1 --data training/agent/data/v2 --out training/agent/out/lora-q3-1.7b-v2
python evals/agent/run_eval.py --model Qwen/Qwen3-1.7B --backend hf --adapter training/agent/out/lora-q3-1.7b-v2 --tag ft2-qwen3-1.7b
```

~2.3 hours. Dev loss 3.40 -> 0.042.

| test set (100, unseen strategies and tickers) | base | v1 | v2 |
|---|---|---|---|
| ran a Lab check | 0 | 72 | 90 |
| right baseline | 0 | 72 | 90 |
| right number of variants | 0 | 65 | 75 |
| right verdict | 0 | 50 | 66 |
| reported its own verdict honestly | 0 | 61 | 90 |
| **all correct** | **0** | **43** | **52** |
| code errors (all questions) | | 73 | 27 |

By family (all correct, v1 -> v2): above-SMA 7 -> 12/12, Bollinger 9 -> 11/11, weekday 0 -> 6/11, SMA cross 7 -> 8/11, momentum 9 -> 9/11, buy-and-hold 4 -> 4/11, RSI 3 -> 1/11, turn-of-month 3 -> 1/10, Donchian 1 -> 0/12. 33 questions right in both, 19 only in v2, 10 only in v1.

Honest reporting rose from 61 to 90, and v2 no longer invents a verdict: when it reports, it repeats what the Lab said. The 48 misses are mostly about the code, not the reporting:
- **Wrong rule for an unseen indicator (about 25).** RSI questions get the training CCI code with the result named `rsi`. "Buy and hold vs cash" gets a momentum rule. Donchian uses a rolling max that includes today, so `close > max` is never true and the signal is flat (NOT REPORTABLE, 14 cases). Each test family is unseen by design, so this is the transfer gap the test set measures.
- **It learned the injected bugs.** Every assistant turn was a training target, including first attempts with a planted error. In 12 test transcripts the model writes the planted lookahead (`.shift(-1).fillna(0.0)`) itself. Planted bugs should stay in the context but not be trained on.
- **Copied explanations and invented tool output.** A flat Donchian signal is explained as "a constant position has no timing to test". After a `NameError` it once answered with a "corrected p = 0.000" that no tool printed. Neither was the final scored answer here, but both are the same habit.

Next (v3): train only on the turns with correct code, keep planted errors as context. When NOT REPORTABLE comes from a rule that should trade, teach the model to check the signal and retry instead of explaining it away. Add generic patterns that are not test rules, such as "prior N-day high" = `shift(1)` before the rolling max, and holding vs cash.
