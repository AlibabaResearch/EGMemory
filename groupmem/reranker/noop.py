"""No selection model: keep the fused order exactly as it came out of fusion.

This is not a stand-in for the real reranker -- it is the ABLATION, the arm that
scores measurably lower. It exists so the pipeline runs with no rerank
credentials and so that "with vs without selection" is a config change rather
than a code edit.

It reports itself as ``noop`` instead of borrowing the fail-open ``passthrough``
counter, because ``passthrough`` means "the model was supposed to run and did
not", which is the one thing that must never be confused with this.
"""

from __future__ import annotations

from typing import List, Optional

from .base import Reranker


class NoopReranker(Reranker):
    """Identity permutation. Never fails, never reorders."""

    def rank(self, query: str, docs: List[str],
             top_n: int) -> Optional[List[int]]:
        self._bump("noop")
        return list(range(min(top_n, len(docs))))
