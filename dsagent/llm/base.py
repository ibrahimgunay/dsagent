"""Provider-agnostic LLM interface.

Every sub-agent depends on this interface, never on a concrete provider, so we
can run the entire system offline against `StubLLM` in tests/CI and against
`OpenAIClient` in production by swapping one object at the composition root.

`complete_json` is the workhorse: agents ask for a structured object and pass an
`intent` tag. Production clients ignore the tag (they instruct the model to emit
JSON and parse it); the offline stub uses the tag to return deterministic,
schema-valid fixtures so the orchestration loop is fully exercised without a
network.
"""
from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    usd: float = 0.0


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    raw: dict | None = None


class LLMClient(abc.ABC):
    """Minimal surface the agents rely on."""

    model: str = "unknown"

    @abc.abstractmethod
    def complete(self, system: str, prompt: str, *,
                 max_tokens: int = 1500, intent: str = "") -> LLMResponse: ...

    def complete_json(self, system: str, prompt: str, *,
                      intent: str = "", max_tokens: int = 1500) -> tuple[dict, Usage]:
        """Return (parsed_object, usage). Robust to code fences / preamble."""
        sys2 = (system + "\n\nRespond with a single valid JSON object and nothing "
                "else. No markdown, no code fences, no commentary.")
        resp = self.complete(sys2, prompt, max_tokens=max_tokens, intent=intent)
        return _extract_json(resp.text), resp.usage


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # salvage the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"LLM did not return JSON: {text[:200]!r}")
