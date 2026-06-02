from .base import LLMClient, LLMResponse, Usage
from .stub import StubLLM
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient

# Default models per provider (override with model=...)
_DEFAULT_MODEL = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
}


def make_client(provider: str = "stub", model: str | None = None) -> LLMClient:
    """Factory: one string picks the backend. Agents never change.

    provider in {"stub", "openai", "gemini"}. Any other provider is a drop-in:
    implement the LLMClient interface and register it here.
    """
    p = provider.lower()
    if p == "stub":
        return StubLLM()
    cls = {"openai": OpenAIClient, "gemini": GeminiClient}.get(p)
    if cls is None:
        raise ValueError(f"unknown provider {provider!r}; choose stub/openai/gemini")
    return cls(model=model or _DEFAULT_MODEL[p])


__all__ = ["LLMClient", "LLMResponse", "Usage", "StubLLM",
           "OpenAIClient", "GeminiClient", "make_client"]
