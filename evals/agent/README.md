# Agent test set

100 questions for the web terminal's agent, with reference answers, and a
scorer that runs any OpenRouter model through the exact tools the browser
uses (the Python shim and system prompt are read from
`docs/terminal/index.html`).

```
python evals/agent/make_questions.py                 # rebuild questions.jsonl (fetches + freezes prices)
python evals/agent/run_eval.py --model oracle        # scorer self-test: must be 100/100
set OPENROUTER_API_KEY=sk-or-...
python evals/agent/run_eval.py --model google/gemma-4-31b-it:free --limit 20
```

**Questions:** 9 strategy families (SMA crossover, trend filter, RSI and
Bollinger mean reversion, momentum, breakout, day of week, turn of month,
buy-and-hold) x 12 tickers, with parameter grids. Each reference is the
family's signal grid run through a fresh `lab()` on prices frozen at
2026-10-30.

**Scores per question:** `ran` (made a Lab check on the symbol), `vs_ok` (right
baseline), `variants_ok` (tested exactly the variants asked for),
`verdict_ok` (its lab's verdict matches the reference), `honest_ok` (its
answer states what its own lab said), `all_ok`.

**Caveat:** 86 of the 100 reference verdicts are NOT SIGNIFICANT, so a model
that always answers "not significant" looks good on `verdict_ok`. The
summary splits every metric by reference class; judge models on the 14
SIGNIFICANT questions and on `variants_ok` / `honest_ok`.

Prices are cached locally in `cache/` (not committed: Yahoo data).
