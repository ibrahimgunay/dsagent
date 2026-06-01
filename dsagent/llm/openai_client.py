"""OpenAI client for the Chat Completions API.

Same `LLMClient` surface as the Anthropic client; stdlib urllib only, so no
extra deps. Reads OPENAI_API_KEY. Uses JSON response mode when the caller's
system prompt asks for JSON (which `complete_json` does), since native JSON mode
is more reliable than prompt-only instructions.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from .base import LLMClient, LLMResponse, Usage

_ENDPOINT = "https://api.openai.com/v1/chat/completions"
# Approx blended price per 1M tokens (USD); tune per model.
_PRICE = {"input": 2.5, "output": 10.0}


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None,
                 timeout: int = 60) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout

    def complete(self, system: str, prompt: str, *,
                 max_tokens: int = 1500, intent: str = "") -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Use StubLLM offline or "
                               "export the key for live runs.")
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }
        if "json" in system.lower():
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            _ENDPOINT, data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenAI API error {e.code}: {e.read().decode()[:300]}")

        text = data["choices"][0]["message"]["content"]
        u = data.get("usage", {})
        it, ot = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        usage = Usage(input_tokens=it, output_tokens=ot, calls=1,
                      usd=(it * _PRICE["input"] + ot * _PRICE["output"]) / 1e6)
        return LLMResponse(text=text, usage=usage, raw=data)
