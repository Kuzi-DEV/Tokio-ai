"""Tool-use loop against any OpenAI-compatible chat-completions endpoint.

Defaults to NVIDIA's free NIM catalog (integrate.api.nvidia.com) rather than
a paid API, on purpose: an open-source tool that requires a metered key to
even try is a real adoption barrier. Point OPENAI_BASE_URL/OPENAI_API_KEY at
any other OpenAI-compatible endpoint (OpenRouter, a local vLLM/Ollama
server, etc.) and this keeps working unmodified.
"""

from __future__ import annotations

import json
import os
from typing import Callable

from openai import OpenAI

from ..rigor.ledger import TestLedger
from ..tools.filings import recent_filings
from ..tools.hypothesis import test_hypothesis as _test_hypothesis
from ..tools.patterns import test_pattern_grid as _test_pattern_grid
from ..tools.patterns import test_return_pattern as _test_return_pattern
from ..tools.prices import fetch_daily_bars
from ..tools.screener import top_performers as _top_performers
from ..tools.strategy import StrategyBook
from ..tools.strategy import backtest_strategy as _backtest_strategy
from ..tools.strategy import final_test_strategy as _final_test_strategy
from .system_prompt import SYSTEM_PROMPT
from .tool_schemas import TOOLS, to_openai_format

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Picked empirically, not from a spec sheet. The previous default,
# nvidia/llama-3.3-nemotron-super-49b-v1.5, was retired by NVIDIA on
# 2026-08-26 and started returning HTTP 410 for every request -- the agent was
# dead on arrival for anyone who installed it after that, and no unit test
# could see it. Replacement chosen 2026-09-24 by running the same real tasks
# (a test_return_pattern call whose p-value must be quoted exactly; an SEC
# filings lookup whose dates were checked by hand) on 7 free-tier candidates:
# this one was 2/2 correct at 10-21s. openai/gpt-oss-20b once tested the
# opposite condition (rises instead of falls); nemotron-3.5-lightning timed
# out once and misreported a filing date; deepseek-v4.1-flash and
# glm-5.3-flash timed out; kimi-k2.6 and nemotron-nano-3 404'd for a free
# account. Expect this to happen again -- free-tier models get retired.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_TIMEOUT = 60.0  # reasoning models can take a while; don't time out mid-thought
MAX_TOOL_ROUNDS = 8  # hard cap so a confused loop can't spin forever

OPENAI_TOOLS = to_openai_format(TOOLS)

# "auto": tools run immediately, no confirmation (the default -- every tool
# in this project is read-only/analysis-only, no trades, no destructive
# actions, so friction-free is the right default). "confirm": every tool
# call is routed through confirm_callback first, which must return True/False
# and may block the calling thread (see tui.py for how the TUI answers this
# from its own event loop via App.call_from_thread).
PERMISSION_LEVELS = ("auto", "confirm")


class Agent:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        initial_usage: dict | None = None,
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            timeout=DEFAULT_TIMEOUT,
            max_retries=1,
        )
        self.model = model or os.environ.get("TOKIO_AI_MODEL", DEFAULT_MODEL)
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.ledger = TestLedger()
        self.strategies = StrategyBook()  # one Lab per symbol for this conversation

        initial_usage = initial_usage or {}
        self.usage = {
            "prompt_tokens": initial_usage.get("prompt_tokens", 0),
            "completion_tokens": initial_usage.get("completion_tokens", 0),
            "total_tokens": initial_usage.get("total_tokens", 0),
            "request_count": initial_usage.get("request_count", 0),
        }

        self.permission_level = "auto"
        # Callable[[tool_name, tool_input], bool]; only consulted when
        # permission_level == "confirm". None means "confirm" degrades to
        # auto-approve rather than crashing (better than blocking forever).
        self.confirm_callback: Callable[[str, dict], bool] | None = None

    def _record_usage(self, response) -> None:
        self.usage["request_count"] += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage["prompt_tokens"] += usage.prompt_tokens or 0
        self.usage["completion_tokens"] += usage.completion_tokens or 0
        self.usage["total_tokens"] += usage.total_tokens or 0

    def _execute_tool(self, name: str, tool_input: dict) -> str:
        if self.permission_level == "confirm" and self.confirm_callback is not None:
            if not self.confirm_callback(name, tool_input):
                return f"ERROR: user denied permission to run {name!r}"
        try:
            if name == "get_price_history":
                bars = fetch_daily_bars(tool_input["symbol"], tool_input.get("range", "10y"))
                return json.dumps([b.to_dict() for b in bars[-500:]])  # cap payload size
            if name == "get_sec_filings":
                filings = recent_filings(
                    tool_input["symbol"], tool_input.get("form_type"), tool_input.get("limit", 10)
                )
                return json.dumps([f.to_dict() for f in filings])
            if name == "test_hypothesis":
                return _test_hypothesis(
                    self.ledger, tool_input["name"], tool_input["group_a"], tool_input["group_b"]
                )
            if name == "top_performing_stocks":
                return json.dumps(
                    _top_performers(
                        sector=tool_input.get("sector"),
                        period=tool_input.get("period", "3mo"),
                        top_n=tool_input.get("top_n", 10),
                    )
                )
            if name == "test_pattern_grid":
                return _test_pattern_grid(
                    self.ledger,
                    tool_input["symbol"],
                    tool_input["feature"],
                    tool_input["op"],
                    tool_input["thresholds"],
                    tool_input["horizons"],
                    tool_input.get("range", "10y"),
                )
            if name == "test_return_pattern":
                return _test_return_pattern(
                    self.ledger,
                    tool_input["symbol"],
                    tool_input["feature"],
                    tool_input["op"],
                    tool_input["threshold"],
                    tool_input["horizon_days"],
                    tool_input.get("range", "10y"),
                )
            if name == "backtest_strategy":
                return _backtest_strategy(
                    self.strategies, tool_input["symbol"], tool_input["name"], tool_input["code"],
                    tool_input.get("params"), tool_input.get("vs", "cash"),
                )
            if name == "final_test_strategy":
                return _final_test_strategy(
                    self.strategies, tool_input["symbol"], tool_input["name"], tool_input["code"],
                    tool_input.get("params"), tool_input.get("vs", "cash"),
                )
            return f"ERROR: unknown tool {name!r}"
        except Exception as e:  # tool errors go back to the model as text, never crash the loop
            return f"ERROR: {e}"

    def send(self, user_text: str) -> str:
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                    messages=self.messages,
                )
                self._record_usage(response)
                msg = response.choices[0].message

                if not msg.tool_calls:
                    self.messages.append({"role": "assistant", "content": msg.content or ""})
                    return msg.content or ""

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )

                for tc in msg.tool_calls:
                    try:
                        tool_input = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}
                    result_text = self._execute_tool(tc.function.name, tool_input)
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                    )
        except Exception:
            # Roll the conversation back to before this turn -- otherwise a
            # failed API call (network blip, timeout) leaves a dangling
            # unanswered user message in history, and the next send() would
            # stack a second user message right after it instead of
            # retrying cleanly.
            del self.messages[checkpoint:]
            raise

        bailout = "[stopped: too many tool calls in a row without a final answer]"
        # Record this in history too -- otherwise the conversation ends
        # mid-tool-exchange (a "tool" message with no assistant response
        # wrapping it up), which the next turn's user message would follow
        # directly, an invalid shape for most chat-completions APIs.
        self.messages.append({"role": "assistant", "content": bailout})
        return bailout
