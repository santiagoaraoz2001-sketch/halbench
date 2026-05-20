"""Chat backends for `halbench run`.

Each backend is a callable that accepts a prompt and returns a response dict.
The default (and v1's only) backend is OpenRouter — covers 50+ models with
one auth token. Native adapters (Anthropic, OpenAI, Google, xAI, vLLM, …)
can be added without touching the runner.
"""
from halbench.backends.base import ChatBackend, ChatResponse
from halbench.backends.openrouter import OpenRouterBackend

__all__ = ["ChatBackend", "ChatResponse", "OpenRouterBackend", "get_backend"]


def get_backend(name: str, **kwargs) -> ChatBackend:
    """Factory for backends by short name."""
    name = name.lower()
    if name in ("openrouter", "or"):
        return OpenRouterBackend(**kwargs)
    raise ValueError(
        f"Unknown backend {name!r}. Available: openrouter. "
        "Submit a PR adding your provider — see backends/base.py."
    )
