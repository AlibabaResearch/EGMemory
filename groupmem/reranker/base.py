"""What the SELECTION step needs of a reranker, plus the splicing that IS the method.

A cross-encoder replaces the mechanical scoring function with a model that reads
(query, passage) jointly. It is the SELECTION step: ``reorder_head`` reorders the
fused (score, mid) list BETWEEN `_fine_fuse` and `_select_indices` in
``memory/retrieval.py``, and `_select_indices` then takes the first k off the
head, so the cross-encoder decides every slot the agent sees. The anchored hop
pass in search_memory is untouched.

The query side is the agent's own search string, NOT the QA question.

DOC_CHARS is a guard, not a policy: it bounds the request against a pathologically
long passage that would otherwise 400. It is set well above every observed passage
length, so nothing on this corpus is clipped.

FAIL-OPEN IS THE CONTRACT, WHICH IS WHY ``stats()`` IS LOAD-BEARING. A backend
returns ``None`` from ``rank`` on any error and ``reorder_head`` hands back the
original fused order, so a transient 429/500 degrades one call toward baseline
instead of killing a run. The cost is that a run whose reranker silently no-op'd
(missing key, exhausted quota) scores like the pre-rerank pipeline and looks like
a valid result. Check `passthrough`, `fail` and `no_key` before believing any
number a run prints.

A backend implements ``rank`` and inherits everything else. The counters live
here rather than in the provider because ``reorder_head`` -- shared, so two
backends cannot drift on the splicing -- is what observes a passthrough.
"""

from __future__ import annotations

import collections
import threading
from typing import List, Optional, Tuple

# ---- frozen constants ------------------------------------------------------
# Environment variables during development, now frozen. The reasoning behind each
# is above. Which model does the selecting stays configurable
# (configs/defaults.RERANK_MODEL), because that is exactly the kind of
# substitution a reader may need to make.
DEPTH = 50               # head candidates handed to the model per call
DOC_CHARS = 4000         # per-passage clip guard (see above)


class Reranker:
    """Interface: implement ``rank``; return ``None`` to fail open."""

    def __init__(self) -> None:
        self._stats = collections.Counter()
        self._lock = threading.Lock()

    # ---- to implement ------------------------------------------------------
    def rank(self, query: str, docs: List[str],
             top_n: int) -> Optional[List[int]]:
        """Doc indices in relevance order (len <= top_n), or None on failure."""
        raise NotImplementedError

    # ---- provided ---------------------------------------------------------
    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._stats[key] += n

    def stats(self) -> dict:
        """Call/failure counters. Read this before trusting a run's accuracy."""
        with self._lock:
            return dict(self._stats)

    def probe(self) -> bool:
        """One throwaway call, to find out BEFORE the run whether rerank works.

        ``reorder_head`` fails open, which is right during a run and wrong at
        startup: a missing key or an unentitled model would let an entire run
        complete, score like the pre-rerank pipeline, and report as a genuine
        result. The counters this leaves behind are reset, so a probe never shows
        up as a real call.
        """
        ok = self.rank("probe", ["the first document", "the second document"], 2)
        with self._lock:
            self._stats.clear()
        return bool(ok)

    def reorder_head(self, store, query: str,
                     base: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        """Rerank the top ``DEPTH`` of ``base``, leaving the tail in place.

        The head is rewritten with synthetic descending scores that all sit above
        the best tail score, so the caller's ``sort``/slice sees exactly the
        model's permutation without the fused scores leaking back in.
        """
        if not base:
            return base
        d = min(DEPTH, len(base))
        if d < 2:
            return base
        # Stripped because the caller passes the agent's raw tool argument, and a
        # stray leading newline is a different string to the scorer.
        query = (query or "").strip()
        head, tail = base[:d], base[d:]
        docs = [store.memories[mid].index_text("rewritten")[:DOC_CHARS]
                for _, mid in head]
        order = self.rank(query, docs, d)
        if not order:
            self._bump("passthrough")
            return base
        # Any candidate the model omitted keeps its relative order after the ones
        # it ranked (top_n=d normally returns all of them; this is a guard).
        ranked = list(order) + [i for i in range(d) if i not in set(order)]
        floor = max((s for s, _ in tail), default=0.0)
        span = max(1.0, abs(floor)) * 1000.0
        # The score must decrease with the NEW rank, not with the original
        # position: the caller consumes this list in score order, which would
        # otherwise silently undo the permutation.
        new_head = [(floor + span - r, head[i][1]) for r, i in enumerate(ranked)]
        self._bump("reordered")
        return new_head + tail
