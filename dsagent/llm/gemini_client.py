"""Google Gemini client for the generateContent API.

Same `LLMClient` surface; stdlib urllib only. Reads GOOGLE_API_KEY (or
GEMINI_API_KEY). Maps the system prompt to `systemInstruction` and requests
`application/json` output when the caller asks for JSON.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from .base import LLMClient, LLMResponse, Usage

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# Approx blended price per 1M tokens (USD); tune per model.
_PRICE = {"input": 1.25, "output": 5.0}


class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None,
                 timeout: int = 60) -> None:
        self.model = model
        self.api_key = (api_key or os.environ.get("GOOGLE_API_KEY")
                        or os.environ.get("GEMINI_API_KEY", ""))
        self.timeout = timeout

    def complete(self, system: str, prompt: str, *,
                 max_tokens: int = 1500, intent: str = "") -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY/GEMINI_API_KEY not set. Use StubLLM "
                               "offline or export the key for live runs.")
        gen_cfg = {"maxOutputTokens": max_tokens}
        if "json" in system.lower():
            gen_cfg["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg,
        }
        url = f"{_BASE}/{self.model}:generateContent?key={self.api_key}"
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="POST",
            headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Gemini API error {e.code}: {e.read().decode()[:300]}")

        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        u = data.get("usageMetadata", {})
        it, ot = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
        usage = Usage(input_tokens=it, output_tokens=ot, calls=1,
                      usd=(it * _PRICE["input"] + ot * _PRICE["output"]) / 1e6)
        return LLMResponse(text=text, usage=usage, raw=data)
