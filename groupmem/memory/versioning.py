"""QA-side store assembly: Postgres table -> retrieval-ready store.

The write side runs OFFLINE and lands one table per domain (variant `release`),
so QA never executes a line of writer code. What is left to do at QA time is the
part that is deliberately NOT baked into the table: the SUPERSEDES edge tier
gate. The table carries every candidate edge the writer produced together with a
numeric `link_confidence` (1.0 = quote-verified, 0.5 = weaker), and the gate
decides here which of them get to act as authority. Keeping it here rather than
at write time makes the tier a one-line change instead of a full re-derive.

  - `link_confidence >= 0.9` -> HARD edge. `superseded_by` is filled in, so the
    node becomes a chain head and `get_current_decision` can walk the chain to it.
  - below the threshold -> DEMOTED. Zero authority: no `supersedes` marker, and
    its changed-payload fields are cleared so the passage cannot claim it
    overturned an earlier decision.

FUSED ROWS GO THROUGH THIS SAME GATE, on a tier they do not store: a `FUS_` row
supersedes a whole chain rather than one predecessor, so `pg_store` derives its
`link_confidence` as the MINIMUM over that chain's hops. An all-high chain gets a
hard fused head (carrying `status=CURRENT_DECISION` and the replaced-value tag);
a chain with one weak hop is demoted exactly as that hop would be.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..utils.lexical import build_lexical_from_store
from ..schema import Relation


HARD_EDGE_THRESHOLD = 0.9


def regate_supersedes(store, threshold: float = HARD_EDGE_THRESHOLD) -> None:
    """Re-attach SUPERSEDES edges by confidence tier, in place."""
    sup_edges = [(e.src, e.dst) for e in store.edges
                 if e.relation == Relation.SUPERSEDES]
    # reset all SUPERSEDES adjacency, then re-attach by tier
    store.edges = [e for e in store.edges if e.relation != Relation.SUPERSEDES]
    for m in store.memories.values():
        m.superseded_by = None
        m.decision_superseded_by = None
        m.supersedes = []
    cooccur: Dict[str, List[str]] = {}   # log-only: how wide the demoted tier is
    demoted: List[str] = []
    n_hard = 0
    for src, dst in sup_edges:
        src_m = store.memories.get(src)
        dst_m = store.memories.get(dst)
        if src_m is None or dst_m is None:
            continue
        if (getattr(src_m, "link_confidence", None) or 0.0) >= threshold:
            store.link(src, dst, Relation.SUPERSEDES)
            n_hard += 1
        else:
            cooccur.setdefault(dst, []).append(src)
            demoted.append(src)
    ncleared = 0
    for mid in demoted:
        m = store.memories.get(mid)
        if m is None or m.decision_type != "changed":
            continue
        m.decision_type = "original"
        m.decision_prior_value = None
        ncleared += 1
    n_heads = sum(1 for m in store.memories.values()
                  if m.supersedes and not m.superseded_by)
    print(f"  [assemble] supersedes {len(sup_edges)} -> hard={n_hard}, "
          f"demoted={len(demoted)} (cooccur targets={len(cooccur)}), "
          f"chain_heads={n_heads}, changed-payload cleared on {ncleared} nodes")
    # by_topic labels are bare (the read tools receive a topic string with no
    # channel context), so two channels reusing a generic label would silently
    # merge into one bucket. Measured 0 of 50 and 0 of 29 labels reused, but
    # report it rather than discover it as a score regression.
    t2ch: Dict[str, set] = {}
    for m in store.memories.values():
        if m.topic and m.channel:
            t2ch.setdefault(m.topic, set()).add(m.channel)
    cross = {t: len(cs) for t, cs in t2ch.items() if len(cs) > 1}
    if cross:
        worst = max(cross.items(), key=lambda kv: kv[1])
        print(f"  [assemble] WARNING {len(cross)}/{len(t2ch)} topic labels span "
              f">1 channel (worst: {worst[1]} channels, {worst[0]!r}) -- "
              f"by_topic buckets and the topic-scoped read tools will mix them")


def assemble(store, id_by_index,
             threshold: float = HARD_EDGE_THRESHOLD) -> Tuple:
    """Make a LOADED store retrieval-ready: gate the edges, build BM25. In place.

    Split out from ``load_qa_store`` so that where the rows came from -- Postgres,
    a JSONL log, a facade that just wrote them -- cannot change what "assembled"
    means. Every backend reaches the reader through this one function, so a second
    loader cannot ship with the gate applied at a different threshold or with BM25
    built over a different field.

    BM25 IS BUILT AFTER THE GATE, BUT DOES NOT DEPEND ON IT. ``index_text`` reads
    only ``content_rewritten`` (falling back to ``clean_content()``), and the gate
    touches neither -- it clears ``decision_type`` and ``decision_prior_value`` and
    re-attaches edges. Measured on the release Manufacturing table: 0 of 31685
    indexed strings differ across the gate, and ``store.order`` is unchanged. So
    this call order is for readability, not correctness; what the gate does change
    is the graph adjacency and the header tags derived from it.
    """
    regate_supersedes(store, threshold)
    lexical = build_lexical_from_store(store, "rewritten")
    return store, id_by_index, lexical


def load_qa_store(domain: str, variant: str = "release",
                  threshold: float = HARD_EDGE_THRESHOLD,
                  keep_noise: bool = False) -> Tuple:
    """Load a domain's memory table and make it retrieval-ready.

    Returns (store, id_by_index, lexical). Vectors stay in pgvector
    (`store.vindex`); BM25 is built over the same `rewritten` text the vectors
    were derived from, so the two retrieval channels sit on one surface.

    `keep_noise` passes straight through to the loader: the noise rows are in the
    table either way, so this is a WHERE clause, not a rebuild. It is for corpora
    where the write prompt's `keep` judgement does not transfer -- see
    `pgvector.load_light_store`.

    THE PGVECTOR IMPORT IS INSIDE THIS FUNCTION, and that is load-bearing rather
    than tidiness: ``vector_stores/pgvector`` imports ``psycopg2`` at module level,
    so importing it up here made ``assemble`` -- the backend-neutral path that the
    in-memory store and the facade go through -- require a Postgres driver to be
    installed in order to gate edges in a dict. It did, until this moved.
    """
    from ..vector_stores import pgvector as pg_store

    store, idx = pg_store.load_light_store(domain, variant=variant,
                                           keep_noise=keep_noise)
    return assemble(store, idx, threshold)
