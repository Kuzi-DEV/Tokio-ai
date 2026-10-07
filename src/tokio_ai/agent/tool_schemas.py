def to_openai_format(tools: list[dict]) -> list[dict]:
    """Adapt this project's tool defs (name/description/input_schema) to the
    OpenAI-compatible function-calling shape used by NVIDIA NIM and friends."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


TOOLS = [
    {
        "name": "get_price_history",
        "description": (
            "Fetch daily OHLCV price history for a stock ticker from Yahoo "
            "Finance. No API key needed. Returns a list of daily bars with "
            "date/open/high/low/close/adj_close/volume."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
                "range": {
                    "type": "string",
                    "description": "History window: 1y, 5y, 10y, 20y. Avoid requesting more than 20y.",
                    "default": "10y",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_sec_filings",
        "description": (
            "List recent SEC filings for a ticker (e.g. 10-K, 10-Q, 8-K) via "
            "EDGAR. No API key needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "form_type": {
                    "type": "string",
                    "description": "e.g. '10-K', '10-Q', '8-K'. Omit for all types.",
                },
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "top_performing_stocks",
        "description": (
            "Rank stocks by trailing return over a real S&P 500 snapshot -- "
            "use this for open-ended questions like 'what are the best "
            "performing stocks' when the user hasn't named a ticker or "
            "sector. Never invent a list of tickers or sectors from memory; "
            "this is the only source of truth for that. Optionally filter "
            "by GICS sector (Information Technology, Health Care, "
            "Financials, Consumer Discretionary, Communication Services, "
            "Industrials, Consumer Staples, Energy, Utilities, Real Estate, "
            "Materials) if the user specifies one -- otherwise omit it and "
            "screen the whole index."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "GICS sector name to filter to. Omit to screen the whole S&P 500.",
                },
                "period": {
                    "type": "string",
                    "enum": ["1mo", "3mo", "6mo", "1y", "ytd"],
                    "default": "3mo",
                },
                "top_n": {"type": "integer", "default": 10},
            },
            "required": [],
        },
    },
    {
        "name": "test_return_pattern",
        "description": (
            "Test whether a simple technical condition on a stock predicts its "
            "forward return. Handles the fetch + bucketing + statistical test "
            "all in one call -- use this instead of manually pulling price "
            "history and eyeballing it whenever the question is shaped like "
            "'does X predict what happens next'. Prefer this over "
            "test_hypothesis for anything involving raw price data; only use "
            "test_hypothesis directly when you already have two numeric "
            "groups from elsewhere. Uses the same engines as tokio_ai.check(): "
            "Hodrick standard errors (overlapping windows handled exactly) "
            "with a circular-shift randomization test as a second opinion. "
            "If the tool reports that the second opinion disagrees, say so."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
                "feature": {
                    "type": "string",
                    "enum": ["daily_return", "gap_pct", "volume_ratio"],
                    "description": (
                        "daily_return: that day's close-over-close return. "
                        "gap_pct: that day's open vs. the prior day's close. "
                        "volume_ratio: that day's volume vs. its trailing 20-day average."
                    ),
                },
                "op": {"type": "string", "enum": [">", ">=", "<", "<="]},
                "threshold": {
                    "type": "number",
                    "description": "e.g. 0.02 for a 2% daily return threshold",
                },
                "horizon_days": {
                    "type": "integer",
                    "description": "how many trading days forward to measure the return",
                },
                "range": {
                    "type": "string",
                    "description": "history window to pull, e.g. 10y, 20y",
                    "default": "10y",
                },
            },
            "required": ["symbol", "feature", "op", "threshold", "horizon_days"],
        },
    },
    {
        "name": "test_pattern_grid",
        "description": (
            "Test a whole grid of thresholds and horizons for one technical "
            "condition at once, corrected for having searched the grid "
            "(tokio_ai.check_many: Romano-Wolf step-down, family-wise error "
            "held at 5%). Use this whenever the question is 'which threshold / "
            "horizon works best' or the user wants several variants compared: "
            "one grid call is honest, several single test_return_pattern calls "
            "followed by picking the best one is p-hacking. Report the grid "
            "verdicts (p grid), and say how many variants would have looked "
            "significant alone but did not survive the grid."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. SPY"},
                "feature": {
                    "type": "string",
                    "enum": ["daily_return", "gap_pct", "volume_ratio"],
                },
                "op": {"type": "string", "enum": [">", ">=", "<", "<="]},
                "thresholds": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "every threshold to try, e.g. [-0.01, -0.02, -0.03]",
                },
                "horizons": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "every forward horizon in trading days, e.g. [1, 5, 20]",
                },
                "range": {"type": "string", "default": "10y"},
            },
            "required": ["symbol", "feature", "op", "thresholds", "horizons"],
        },
    },
    {
        "name": "test_hypothesis",
        "description": (
            "Run a two-sided studentized permutation test comparing two groups of "
            "numbers (e.g. forward returns after a signal fires vs. a "
            "baseline). ALWAYS use this before claiming any pattern is real -- "
            "never eyeball a mean and call it significant. Enforces a minimum "
            "sample size and corrects for every other hypothesis tested this "
            "session, so results only get more conservative as more signals "
            "get tested in one conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short unique label for this hypothesis, e.g. 'AAPL_positive_surprise_20d'",
                },
                "group_a": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "e.g. forward returns in the 'signal fired' condition",
                },
                "group_b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "e.g. forward returns in the baseline/control condition",
                },
            },
            "required": ["name", "group_a", "group_b"],
        },
    },
]
