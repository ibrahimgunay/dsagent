"""Production LLM client for the Anthropic Messages API.

Uses only the standard library (urllib) so the package keeps zero hard runtime
deps. Reads the key from ANTHROPIC_API_KEY. This is the path that runs in
production; it is import-safe offline (no network at import time) and only
touches the network when `complete` is called.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from .base import LLMClient, LLMResponse, Usage

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_VERSION = "2023-06-01"

# Approx blended price per 1M tokens (USD) for budget tracking; tune per model.
_PRICE = {"input": 3.0, "output": 15.0}


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: str | None = None, timeout: int = 60) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.timeout = timeout

    def complete(self, system: str, prompt: str, *,
                 max_tokens: int = 1500, intent: str = "") -> LLMResponse:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Use StubLLM for offline runs, or "
                "export the key for production.")
        body = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(_ENDPOINT, data=body, method="POST", headers={
            "x-api-key": self.api_key,
            "anthropic-version": _VERSION,
            "content-type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Anthropic API error {e.code}: {e.read().decode()[:300]}")

        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        u = data.get("usage", {})
        it, ot = u.get("input_tokens", 0), u.get("output_tokens", 0)
        usage = Usage(input_tokens=it, output_tokens=ot, calls=1,
                      usd=(it * _PRICE["input"] + ot * _PRICE["output"]) / 1e6)
        return LLMResponse(text=text, usage=usage, raw=data)
