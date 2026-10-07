"""Entry points for the research agent, which needs the `agent` extra.

`pip install tokio-ai` installs only the statistics (numpy). The agent's
TUI and REPL need openai and textual; without them, say how to get them
instead of a traceback.
"""

from __future__ import annotations

import sys

_HINT = ('The TokIO research agent needs extra packages: pip install "tokio-ai[agent]"\n'
         "(The statistics -- check_backtest, tokio-ai-backtest -- work without them.)")


def _missing(e: ImportError) -> None:
    print(f"tokio-ai: {e.name or e} is not installed.\n{_HINT}", file=sys.stderr)
    raise SystemExit(2)


def tui() -> None:
    try:
        from .tui import main
    except ImportError as e:
        _missing(e)
    main()


def plain() -> None:
    try:
        from .cli import main
    except ImportError as e:
        _missing(e)
    main()
