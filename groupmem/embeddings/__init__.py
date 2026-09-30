"""Embedding providers. See ``base.py`` for what an embedder must guarantee.

  dashscope_native : the dashscope SDK. THE DEFAULT BENCHMARK EMBEDDER.
  openai           : any ``POST {base_url}/embeddings`` endpoint -- OpenAI, vLLM,
                     Ollama, TEI, DashScope's compatible mode (needs `openai`).
  dashscope        : the same class as `openai`, kept as a name because the default
                     base_url is DashScope's compatible route.
  hashing          : offline, no key, no network. Plumbing only -- see hashing.py.

THERE IS NO DEFAULT BACKEND, ON PURPOSE: a hashing embedder does not fail, it
silently retrieves worse, so name the backend you want. Imports are inside the
branches so that picking one backend never drags in the other's dependency.
"""

from __future__ import annotations

from typing import Any

from .base import Embedder, l2_normalize

__all__ = ["Embedder", "l2_normalize", "get_embedder"]


def get_embedder(backend: str, **kwargs: Any) -> Embedder:
    """Build an embedder by name. Unknown names raise rather than fall back."""
    if backend == "dashscope_native":
        from .dashscope import DashScopeNativeEmbedder
        return DashScopeNativeEmbedder(**_only(kwargs, "model", "cache_dir",
                                               "batch_size", "dim"))
    if backend == "dashscope":
        from .dashscope import NATIVE_DIM
        from .openai_compat import OpenAICompatEmbedder
        kwargs.setdefault("dim", NATIVE_DIM)
        return OpenAICompatEmbedder(**_only(kwargs, "model", "cache_dir",
                                            "batch_size", "dim", "api_key",
                                            "base_url"))
    if backend in ("openai", "openai_compat"):
        from .openai_compat import OpenAICompatEmbedder
        return OpenAICompatEmbedder(**_only(kwargs, "model", "cache_dir",
                                            "batch_size", "dim", "api_key",
                                            "base_url"))
    if backend == "hashing":
        from .hashing import HashingEmbedder
        return HashingEmbedder(**_only(kwargs, "dim"))
    raise ValueError(
        f"unknown embedding backend {backend!r}; "
        "expected 'dashscope_native', 'openai' (any /v1/embeddings endpoint), "
        "'dashscope' or 'hashing'"
    )


def _only(kwargs: dict, *keys: str) -> dict:
    """The subset a backend's constructor accepts.

    Filtered rather than passed through because a config carries one embedder
    section for whichever backend is selected, and a key meant for another one
    (``base_url`` at the native SDK, say) would otherwise be a TypeError at
    construction -- from a config that is correct for the backend it names.
    """
    return {k: v for k, v in kwargs.items() if k in keys}
