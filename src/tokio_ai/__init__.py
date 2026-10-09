"""TokIO AI: an open-source financial research agent that runs every claim
through honest statistical testing before trusting it."""

__version__ = "0.6.0"

from .check import CheckResult, check  # noqa: E402  (needs __version__ defined first)
from .backtest import BacktestResult, check_backtest  # noqa: E402
from .family import FamilyResult, check_many  # noqa: E402
from .overfit import OverfitResult, probability_of_overfitting  # noqa: E402
from .contracts import ContractsResult, check_contracts  # noqa: E402
from .sharpe import SharpeStats, deflated_sharpe_ratio, min_track_record_length, probabilistic_sharpe_ratio  # noqa: E402
from .trades import TradeList, read_tradingview  # noqa: E402
from .lab import Lab, LookaheadError, load_prices  # noqa: E402
from .rigor.ledger import TestLedger  # noqa: E402

__all__ = ["check", "check_many", "check_backtest", "BacktestResult", "probability_of_overfitting", "OverfitResult", "check_contracts", "ContractsResult", "CheckResult", "FamilyResult", "TestLedger", "deflated_sharpe_ratio", "probabilistic_sharpe_ratio", "min_track_record_length", "SharpeStats", "read_tradingview", "TradeList", "Lab", "LookaheadError", "load_prices", "__version__"]
