"""Budget governor.

Caps the run by USD, tokens, and tool-call count. The orchestrator checks the
budget before dispatching each task and records usage after, so a runaway plan
stops instead of burning unbounded spend.
"""
from __future__ import annotations

from dataclasses import dataclass
from ..llm.base import Usage


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_usd: float = 5.0
    max_tokens: int = 2_000_000
    max_tool_calls: int = 200

    spent_usd: float = 0.0
    spent_tokens: int = 0
    tool_calls: int = 0

    def check(self) -> None:
        if self.spent_usd > self.max_usd:
            raise BudgetExceeded(f"USD budget exceeded: {self.spent_usd:.2f} > {self.max_usd}")
        if self.spent_tokens > self.max_tokens:
            raise BudgetExceeded(f"Token budget exceeded: {self.spent_tokens} > {self.max_tokens}")
        if self.tool_calls > self.max_tool_calls:
            raise BudgetExceeded(f"Tool-call budget exceeded: {self.tool_calls} > {self.max_tool_calls}")

    def record(self, usage: Usage) -> None:
        self.spent_usd += usage.usd
        self.spent_tokens += usage.input_tokens + usage.output_tokens
        self.tool_calls += max(usage.calls, 1)

    def summary(self) -> dict:
        return {"usd": round(self.spent_usd, 4), "tokens": self.spent_tokens,
                "tool_calls": self.tool_calls}
