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
