"""Chat / tool-calling providers. See ``base.py`` for what a provider must do."""

from __future__ import annotations

from typing import Any

from .base import (HARD_TIMEOUT_QA, HARD_TIMEOUT_WRITE, THINKING_QA,
                   THINKING_WRITE, LLM)

__all__ = ["LLM", "get_llm", "HARD_TIMEOUT_QA", "HARD_TIMEOUT_WRITE",
           "THINKING_QA", "THINKING_WRITE"]


def get_llm(backend: str, model: str, **kwargs: Any) -> LLM:
    """Build a provider by name. NO DEFAULT BACKEND, and no default model.

    Both are deliberate. A default backend would mean a config with a typo'd
    provider name still runs -- against something else. A default model is worse:
    the model id decides the numbers, so guessing one produces a complete,
    plausible, differently-scoring result set with nothing in the logs to say why.

    Imports are lazy so that neither the DashScope SDK nor anything else is needed
    to load a provider the caller did not ask for.
    """
    if backend == "dashscope":
        from .dashscope import DashScopeLLM
        return DashScopeLLM(model, **kwargs)
    if backend in ("openai", "openai_compat"):
        from .openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(model, **kwargs)
    raise ValueError(
        f"unknown llm backend {backend!r}; known: dashscope, openai_compat "
        f"(any /v1/chat/completions endpoint -- vLLM, Ollama, DeepSeek, ...)")
