from .base import LLMClient, LLMResponse, Usage
from .stub import StubLLM
from .anthropic_client import AnthropicClient
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient

# Default models per provider (override with model=...)
_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
}


def make_client(provider: str = "stub", model: str | None = None) -> LLMClient:
    """Factory: one string picks the backend. Agents never change.

    provider in {"stub", "anthropic", "openai", "gemini"}.
    """
    p = provider.lower()
    if p == "stub":
        return StubLLM()
    cls = {"anthropic": AnthropicClient, "openai": OpenAIClient,
           "gemini": GeminiClient}.get(p)
    if cls is None:
        raise ValueError(f"unknown provider {provider!r}; "
                         f"choose stub/anthropic/openai/gemini")
    return cls(model=model or _DEFAULT_MODEL[p])


__all__ = ["LLMClient", "LLMResponse", "Usage", "StubLLM", "AnthropicClient",
           "OpenAIClient", "GeminiClient", "make_client"]
