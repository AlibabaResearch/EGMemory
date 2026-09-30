"""Multi-signal fusion retriever over the memory graph (read side).

One retrieval path, two ranking stages, one graph post-processor:

  SCOPE -- `_candidate_ids`: an explicit scope (phase / topic / author, which the
    agent passes when it copies a value off a passage tag) restricts the pool to
    the hard INTERSECTION of those buckets; with no scope the pool is the whole
    store, so recall is never silently reduced.
  FUSION -- `_fine_fuse`: dense top-200 and BM25 top-200 within that pool, fused
    by Reciprocal Rank Fusion. This produces the CANDIDATE order.
  SELECTION -- `reranker.reorder_head`: a cross-encoder re-reads (query, passage)
    jointly over the top 50 of that order and rewrites it. This is the pass that
    decides which passages the agent actually sees. `_select_indices` then takes
    the first k of that order, and that is all it does.

The graph neither re-weights nor tops up this order: the retrieval-side injection
earlier versions ran (superseded penalty, current-version top-up, topic-head and
co-occurrence injection) is gone. Currency and structure reach the answer through
the explicit read tools and the version tags rendered on every passage, so the
edges and `_heads_by_topic` are kept; only the retrieval-side injection is not.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..reranker import Reranker
from ..schema import MemoryStore


# ---- frozen retrieval constants --------------------------------------------
# All of these were environment variables during development, now frozen.
RRF_K0 = 60              # RRF denominator offset
FINE_TOPN = 200          # per-channel retrieval depth
# PRIMARY_RATIO (0.7) and SUPERSEDED_PENALTY (0.5) used to sit here and are gone
# with the expansion pass. PRIMARY_RATIO never had a meaning of its own:
# reserving ceil(0.7*k) slots existed only to stop hit #1's expansions from
# crowding out hit #2, so with no expansions the layered fill collapses into
# "take the first k".


class FusionRetriever:
    def __init__(
        self,
        store: MemoryStore,
        id_by_index: List[str],
        embedder,
        lexical,
        reranker: Reranker,
    ) -> None:
        # The dense channel is delegated, not held: node vectors never enter
        # Python. Any ``vector_stores.VectorIndex`` satisfies this -- pgvector for
        # the benchmark runs, in-memory for a keyless local run -- but SOMETHING has
        # to satisfy it, because a store loaded without one would fuse a single
        # channel and still return an order, i.e. degrade silently.
        if getattr(store, "vindex", None) is None:
            raise ValueError(
                "FusionRetriever needs a store with a .vindex (VectorIndex); "
                "use vector_stores.pgvector.load_light_store or "
                "vector_stores.inmemory.load_store. Node embeddings are not kept "
                "in Python.")
        # Required, not defaulted: which backend does the SELECTING is the single
        # largest read-side lever, so it is named at the call site rather than
        # inherited from a default nobody reads.
        self.store = store
        self.id_by_index = id_by_index
        self.index_by_id = {mid: i for i, mid in enumerate(id_by_index)}
        self.embedder = embedder
        self.lexical = lexical
        self.reranker = reranker

        # Current heads of change chains = a node that supersedes an older
        # decision and is itself not superseded. For knowledge_update questions
        # the answer lives here. This index is NOT consulted during retrieval any
        # more (that injection priced null and is gone); it exists solely to serve
        # the get_current_decision tool, which is the one graph consumer that DID
        # price positive.
        self._chain_heads = {
            mid for mid, m in store.memories.items()
            if m.supersedes and not m.superseded_by
        }
        # Keyed by the agent-facing LABEL, because its single consumer is
        # ReadTools.get_current_decision looking up the topic string it copied
        # off a passage. There used to be a second index over the same heads
        # (`_heads_by_group`, keyed by `topic_group`) feeding the retrieval-side
        # injection; it went with the expansion pass, and it had been a
        # duplicate anyway -- `topic_group` is not among the columns pg_store
        # selects, so it is None on every loaded node (verified: 0 of 30117 on
        # Manufacturing) and `topic_group or topic` always fell back to the
        # label.
        #
        # Built by walking `store.order`, NOT by iterating the `_chain_heads`
        # set. A topic with several heads is reported in this list's order and
        # the tool's cap cuts the tail, so the order is observable. Set
        # iteration over str keys follows per-process hash randomisation, which
        # would make that tail depend on PYTHONHASHSEED: two runs differing in
        # nothing else could return a different id list. Ingestion order is
        # stable across processes and puts the earliest-seen head first.
        self._heads_by_topic: Dict[str, List[str]] = {}
        for mid in store.order:
            if mid not in self._chain_heads:
                continue
            _m = store.memories[mid]
            if _m.topic:
                self._heads_by_topic.setdefault(_m.topic, []).append(mid)

    # ---- candidate selection ----
    def _candidate_ids(self, scope: Optional[Dict[str, str]]) -> List[str]:
        """Scope pre-filter via the channel/topic/phase/author indexes.

        With a scope hint, restrict scoring to the INTERSECTION of the supplied
        buckets instead of scanning all N nodes, which improves precision and
        scales. With no hint, scan the full insertion order so recall is never
        silently reduced.
        """
        if not scope:
            return self.store.order
        buckets: List[set] = []
        ch = scope.get("channel")
        tp = scope.get("topic")
        ph = scope.get("phase")
        au = scope.get("author")
        if ch:
            buckets.append(set(self.store.by_channel.get(ch, [])))
        if tp:
            buckets.append(set(self.store.by_topic.get(tp, [])))
        if ph:
            buckets.append(set(self.store.by_phase.get(ph, [])))
        if au:
            buckets.append({mid for mid, m in self.store.memories.items()
                            if m.author == au})
        if not buckets:
            return self.store.order
        keep = set.intersection(*buckets) if len(buckets) > 1 else buckets[0]
        # preserve insertion order for determinism; fall back to full scan if
        # the intersection came up empty (avoid silently returning nothing).
        ordered = [mid for mid in self.store.order if mid in keep]
        return ordered or self.store.order

    def _dense_rank(self, qvec, candidate_ids, topn: int) -> List[str]:
        """Rank candidate ids by pure cosine (no graph boosts), best-first. Runs
        SERVER-SIDE in pgvector; embeddings never enter Python. A full-store
        candidate list is passed as None so the index can skip the id filter."""
        full = (candidate_ids is self.store.order
                or len(candidate_ids) == len(self.store.order))
        return self.store.vindex.search(qvec, topn,
                                        None if full else list(candidate_ids))

    def _bm25_rank_subset(self, query: str, pool_ids, topn: int) -> List[str]:
        """BM25 ranking restricted to a candidate pool."""
        if self.lexical is None:
            return []
        sm = self.lexical.score_map(query)
        scored = [(sm.get(mid, 0.0), mid) for mid in pool_ids if sm.get(mid, 0.0) > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mid for _, mid in scored[:topn]]

    @staticmethod
    def _rrf(ranked_lists: List[List[str]]) -> Dict[str, float]:
        """Reciprocal Rank Fusion: score(id) = Σ 1/(RRF_K0 + rank+1) over
        channels. Every channel weight in this configuration is 1.0 -- dense and
        BM25 are fused at parity in both the main retrieve and the hop ranking --
        so there is no per-channel weight argument. (A third proximity route in
        the hop ranking used to fuse here too; it priced null on both benchmarks
        as a ranking signal and is gone -- see _hop_rank.)"""
        fused: Dict[str, float] = {}
        for lst in ranked_lists:
            for rank, mid in enumerate(lst):
                fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K0 + rank + 1)
        return fused

    def _fine_fuse(self, query: str, qvec, pool):
        """RRF over dense + bm25 within the pool. This is the fusion that decides
        the returned order.

        Scores are normalised to the top hit. That is cosmetic for the ordering
        (a monotone rescale) but not for the next stage: reranker.reorder_head
        rewrites the head by synthesising scores above the tail's floor, so it
        needs the tail on a bounded scale.
        """
        pool_list = list(pool)
        dense_list = self._dense_rank(qvec, pool_list, FINE_TOPN)
        lists: List[List[str]] = [dense_list]
        if self.lexical is not None:
            lists.append(self._bm25_rank_subset(query, pool_list, FINE_TOPN))
        fused = self._rrf(lists)
        if not fused:
            return []
        mx = max(fused.values()) or 1.0
        base = [(fs / mx, mid) for mid, fs in fused.items()]
        base.sort(key=lambda x: x[0], reverse=True)
        return base

    def _select_indices(self, base, k: int) -> List[int]:
        """Take the first k distinct ids off the final order, as indices into the
        original message list.

        Nothing is re-scored and nothing is topped up: the order this consumes is
        already final (fusion, then the cross-encoder), and the graph expansion
        that used to interleave into it priced null -- see the module docstring.

        De-duplication is still required. `base` can name the same node twice
        because reranker.reorder_head re-appends any candidate the model omitted
        from its ranking, and dropping the guard would let one node consume two
        of the k slots.
        """
        indices: List[int] = []
        seen = set()
        for _, mid in base:
            if mid in seen:
                continue
            seen.add(mid)
            idx = self.index_by_id.get(mid)
            if idx is not None:
                indices.append(idx)
            if len(indices) >= k:
                break
        return indices

    def retrieve(self, query: str, k: int,
                 scope: Optional[Dict[str, str]] = None,
                 dense_query: Optional[str] = None) -> List[int]:
        # Two-query decoupling: the lexical (bm25) channel wants keywords, the
        # dense channel wants a phrasing close to how memories are STORED
        # (narrative rewritten sentences). Forcing one string to serve both
        # trades one channel off against the other. `query` drives bm25; an
        # optional `dense_query` (narrative) drives the embedding. When absent,
        # dense falls back to `query`.
        qvec = self.embedder.embed([dense_query or query])[0]
        pool = self._candidate_ids(scope)
        base = self._fine_fuse(query, qvec, pool)
        # SELECTION: the cross-encoder scores against `query` -- the agent's own
        # search string, not `dense_query` and not the QA question; see
        # reranker.py for why the alternatives are not used. Fail-open, so an
        # API error degrades this call to the fused order rather than
        # aborting the run -- which is why reranker.stats() has to be checked
        # before a run's accuracy is reported.
        base = self.reranker.reorder_head(self.store, query, base)
        return self._select_indices(base, k)
