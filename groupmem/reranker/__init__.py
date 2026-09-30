"""Rerank providers -- the SELECTION step. ``base.py`` holds the method.

  dashscope : DashScope text-rerank. The backend the default benchmark uses.
  noop      : keep the fused order. This is the ablation arm, not a fallback.

No default backend: running the ablation by accident is a real accuracy loss
that shows up as a plausible score rather than as an error.
"""

from __future__ import annotations

from typing import Any

from .base import DEPTH, DOC_CHARS, Reranker

__all__ = ["Reranker", "DEPTH", "DOC_CHARS", "get_reranker"]


def get_reranker(backend: str, **kwargs: Any) -> Reranker:
    """Build a reranker by name. Unknown names raise rather than fall back."""
    if backend == "dashscope":
        from .dashscope import DashScopeReranker
        return DashScopeReranker(**{k: v for k, v in kwargs.items()
                                    if k == "model"})
    if backend == "noop":
        from .noop import NoopReranker
        return NoopReranker()
    raise ValueError(f"unknown rerank backend {backend!r}; "
                     "expected 'dashscope' or 'noop'")
