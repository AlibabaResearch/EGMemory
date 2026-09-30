"""Offline hashing embedder: no network, no key, no model download.

Not for retrieval quality -- a signed bag-of-tokens hash has no semantics. It is
here so the parts of the system that are NOT the model (graph construction,
supersedes gating, fusion chains, RRF, the tool loop's plumbing, the in-memory
store) can be exercised without an API key.

RETRIEVAL QUALITY IS NOT MEANINGFUL WITH THIS EMBEDDER: any number under it
describes the plumbing, not the method. ``configs/defaults.EMBEDDING_MODEL`` names
the embedder the benchmark configuration uses, and the factory in ``__init__`` has
no default backend so nothing reaches for this one by accident.

Deterministic across processes and machines: md5 of the token, not ``hash()``,
which is per-process salted.
"""

from __future__ import annotations

import hashlib
import re
from typing import List

from .base import Embedder, l2_normalize

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class HashingEmbedder(Embedder):
    """Deterministic, offline, parameter-free hashing embedder."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in _TOKEN_RE.findall((t or "").lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[idx] += sign
            out.append(l2_normalize(vec))
        return out
