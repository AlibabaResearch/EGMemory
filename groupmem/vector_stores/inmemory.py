"""In-memory vector store: the same graph and the same fusion, with no database.

``memory/retrieval.py`` never holds a node vector -- it asks a ``VectorIndex`` for
a ranked id list and fuses that with BM25. That indirection was for pgvector's
sake, and it also means the dense channel is the only thing tying the system to
Postgres, so this file lets the method run unchanged against a dict: a reader can
clone the repo and retrieve without provisioning anything.

NOT THE DEFAULT BENCHMARK PATH -- the benchmark numbers come from
``pgvector.load_light_store`` against a real table (``build_store`` in ``base.py``
is shared, so the two cannot disagree about what a row means). Search here is exact
brute-force cosine; pgvector with an ivfflat/hnsw index is approximate, so expect
near-identical retrieval, not bit-identical.

WHERE IT DOES NOT SCALE. O(N*d) per query, single-threaded, all vectors resident:
fine for one domain, not for a service.

Rows are the same dicts ``pgvector`` selects (``GM_COLS``), plus ``embedding``.
Supply the vectors, or supply an embedder and they are computed from the same
``index_text("rewritten")`` the table stores -- which must match, because a store
embedded with one model and queried with another does not error, it just ranks wrong.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..schema import MemoryStore
from .base import VectorIndex, build_store

try:                                    # optional: pure-Python fallback below
    import numpy as _np
except Exception:                       # noqa: BLE001
    _np = None


def read_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    """One row per line -> the row list ``load_store`` wants. The portable format.

    TWO THINGS THIS DOES THAT A BARE ``json.loads`` PER LINE DOES NOT, both of them
    reproducing what the pgvector table does rather than adding policy:

      collapse by memory_id, LAST occurrence wins -- the log is append-only and a
        fused row is rewritten on every hop of its chain, so the same memory_id
        legitimately appears several times. Postgres resolves that with ``ON
        CONFLICT (memory_id) DO UPDATE``; ``MemoryStore.add`` resolves a duplicate
        the OTHER way (first wins, it returns early), so reading the log without
        collapsing would silently keep each chain's OLDEST fused row -- a store
        that loads clean and describes superseded state.
      sort by ord -- the pgvector loader selects ``ORDER BY ord``, and build_store's
        last-write-wins on ``superseded_by`` depends on that order, fused rows
        included (their ord is FUSION_ORD_BASE + the chain root's, so they land
        after every message row).

    No schema check: ``build_store`` accesses the columns it needs and fails on a
    missing one, which is what catches a store written by an older writer instead
    of loading it half-populated.
    """
    by_id: "Dict[str, Dict[str, Any]]" = {}
    for i, line in enumerate(open(path, "r", encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        mid = row.get("memory_id")
        if mid is None:
            raise ValueError(f"{path}:{i+1} has no 'memory_id'")
        by_id[mid] = row
    return sorted(by_id.values(), key=lambda r: (r.get("ord") or 0))


class InMemoryIndex(VectorIndex):
    """Exact cosine top-k over vectors held in this process.

    Vectors are stored L2-normalised, so cosine is a dot product and the ranking
    needs no per-query division. Rows whose vector is empty are simply absent --
    the same state pgvector's ``embedding IS NOT NULL`` filter produces, and the
    reason it is a filter and not a zero vector: a zero vector has cosine 0 with
    everything, which is a legitimate score and would let an unembedded node
    outrank a genuinely irrelevant one.

    A NON-FINITE vector raises instead of being dropped. Missing is a state this
    system has a defined answer for; NaN is corruption, and it would otherwise
    propagate into every score as a comparison that is neither greater nor less,
    i.e. an order that depends on the sort's implementation.
    """

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        ids: List[str] = []
        rows: List[List[float]] = []
        for mid, vec in vectors.items():
            if vec is None or len(vec) == 0:
                continue
            n = math.sqrt(sum(float(x) * float(x) for x in vec))
            if not math.isfinite(n):
                raise ValueError(f"vector for {mid!r} is not finite (NaN/inf); "
                                 f"the embedder or the row is corrupt")
            if n < 1e-12:
                continue
            ids.append(mid)
            rows.append([float(x) / n for x in vec])
        self.ids = ids
        self.pos = {mid: i for i, mid in enumerate(ids)}
        if _np is not None:
            self._mat = _np.asarray(rows, dtype="float32") if rows else None
            self._rows = None
        else:
            self._mat = None
            self._rows = rows

    def search(self, qvec, k: int,
               candidate_ids: Optional[Sequence[str]] = None) -> List[str]:
        if not self.ids or k <= 0:
            return []
        n = math.sqrt(sum(float(x) * float(x) for x in qvec))
        if not math.isfinite(n):
            raise ValueError("query vector is not finite (NaN/inf)")
        q = [float(x) / n for x in qvec] if n > 1e-12 else list(qvec)
        # A candidate list is a SCOPE, so an empty one means "nothing in scope"
        # and must return nothing. Returning the unfiltered top-k there would
        # silently widen the scope the caller asked to narrow.
        if candidate_ids is not None:
            sel = [self.pos[m] for m in candidate_ids if m in self.pos]
            if not sel:
                return []
        else:
            sel = None
        if self._mat is not None:
            qa = _np.asarray(q, dtype="float32")
            mat = self._mat if sel is None else self._mat[sel]
            # errstate, NOT a warnings filter, and scoped to this one call.
            # numpy 2.x built against Apple's Accelerate BLAS raises spurious
            # divide-by-zero/overflow/invalid flags out of matmul -- reproducible
            # on freshly generated random float32 with no non-finite element in
            # it, so it says nothing about these vectors. Both operands are
            # guaranteed finite at construction (above) and by the query check,
            # which is what makes suppressing it here honest rather than a way to
            # stop hearing about real corruption.
            with _np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                scores = mat @ qa
            kk = min(k, scores.shape[0])
            # argpartition then sort the slice: O(N) + O(k log k) instead of
            # sorting N. Ties broken by index so the order is deterministic --
            # the fusion downstream is rank-based, so a nondeterministic tie
            # would make the same query score differently across runs.
            part = _np.argpartition(-scores, kk - 1)[:kk] if kk < scores.shape[0] \
                else _np.arange(scores.shape[0])
            part = part[_np.lexsort((part, -scores[part]))]
            src = self.ids if sel is None else [self.ids[i] for i in sel]
            return [src[int(i)] for i in part]
        rows = self._rows or []
        idxs = range(len(rows)) if sel is None else sel
        scored = [(sum(a * b for a, b in zip(rows[i], q)), i) for i in idxs]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [self.ids[i] for _, i in scored[:k]]


def load_store(rows: Iterable[Mapping[str, Any]], keep_noise: bool = False,
               embedder=None) -> Tuple[MemoryStore, List[str]]:
    """Rows -> (store with ``.vindex``, id_by_index). The pgvector loader's twin.

    Identical to ``pgvector.load_light_store`` except for where the rows come from
    and where the vectors live; both call the same ``build_store``, so the graph,
    the derived fields, the edges and the insertion order are not reimplemented
    here and cannot drift.

    ``embedder`` is used only for rows that carry no ``embedding``. It embeds
    ``index_text("rewritten")`` -- the same text pgvector's ingest embeds -- in one
    batch, because a per-row call against a hosted endpoint turns a 31k-node load
    into an afternoon.

    A row with no vector is LEGAL and is dropped from the dense channel, because
    that is exactly what pgvector's ``embedding IS NOT NULL`` does and the real
    tables contain such rows (gm_manufacturing__release: 27 of 31699). It is
    reported rather than swallowed. What is NOT legal is EVERY row missing one: the
    dense channel would go dark, the fusion would quietly become single-channel
    BM25, and it would still return a plausible order -- so that raises.
    """
    rows = list(rows)
    store, ids = build_store(rows, keep_noise=keep_noise)
    by_id = {r.get("memory_id"): r for r in rows}

    vectors: Dict[str, Sequence[float]] = {}
    missing: List[str] = []
    for mid in ids:
        r = by_id.get(mid) or {}
        emb = r.get("embedding")
        if emb:
            vectors[mid] = emb
        else:
            missing.append(mid)
    if missing:
        if embedder is not None:
            texts = [store.memories[mid].index_text("rewritten")
                     for mid in missing]
            for mid, vec in zip(missing, embedder.embed(texts)):
                vectors[mid] = vec
        elif not vectors:
            raise ValueError(
                f"none of the {len(missing)} row(s) carry an 'embedding' and no "
                f"embedder was given. The dense channel would be empty and "
                f"retrieval would silently fall back to BM25 alone -- pass "
                f"embedder= to compute the vectors, or supply them in the rows.")
        else:
            print(f"[inmemory] {len(missing)} of {len(ids)} node(s) have no "
                  f"vector and are excluded from the dense channel (same as "
                  f"pgvector's `embedding IS NOT NULL`); they remain reachable "
                  f"via BM25 and the graph.", flush=True)

    store.vindex = InMemoryIndex(vectors)
    return store, ids
