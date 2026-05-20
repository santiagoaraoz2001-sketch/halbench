"""Chat backend abstract base.

To add a new provider, subclass ChatBackend, implement chat(), and register
in backends/__init__.py::get_backend.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChatResponse:
    """Single chat completion result."""
    response_text: str
    finish_reason: str = "stop"
    usage_in_tokens: int = 0
    usage_out_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None


class ChatBackend(ABC):
    """A provider that runs single-turn chat completions for one model.

    Implementations should be resilient to transient errors (HTTP 5xx, rate
    limits) via internal retry. Hard errors (auth, model not found) should
    surface as ChatResponse(error=...) rather than raising.
    """

    name: str
    model_id: str

    @abstractmethod
    def chat(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> ChatResponse:
        """Run a single-turn chat. Override in subclass."""
        ...
