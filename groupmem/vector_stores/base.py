"""What a store backend has to be, and the part that is the same for all of them.

TWO THINGS LIVE HERE:

  ``VectorIndex``  the ONLY thing retrieval asks of a backend -- ``search``. A
                   backend is a top-k over ids, not a database, so this is a
                   two-method surface and not an ORM.

  ``build_store``  rows -> MemoryStore. This is METHOD, not storage: the six
                   derived read-side fields, the edge construction, the fused
                   row's weakest-hop tier, the last-write-wins ordering. Every
                   backend calls THIS function and differs only in where its rows
                   come from and what answers ``search``, so there is never a
                   second implementation of the graph rules.

A ROW IS A MAPPING KEYED BY ``GM_COLS``. That the column names are Postgres's is
history, not a dependency: the names are the field contract, and a backend that
stores rows some other way maps into them before calling here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..schema import Memory, MemoryStore, Relation, _parse_ts

# THE ROW CONTRACT. Ordered, because the SQL backend builds positional tuples from
# it; it lives here rather than in ``pgvector`` because it is what EVERY backend
# maps into, and a backend that has no database still has to name its fields. The
# DDL types stay with the backend that needs types.
GM_COLS = [
    "memory_id", "node_kind", "channel", "ch_ord", "ord",
    "author", "role", "timestamp", "reply_to",
    "content", "rewritten",
    "keep", "verdict", "slot", "decision_quote",
    "topic", "phase", "phase_name",
    "supersedes", "sources", "link_conf",
]

# The numeric tier values ``build_store`` writes into ``link_confidence`` and
# ``memory.versioning.HARD_EDGE_THRESHOLD`` reads back. The ordering
# LINK_CONF_HIGH > HARD_EDGE_THRESHOLD > LINK_CONF_LOW is what makes a "high" edge
# hard and a "low" edge demoted; changing either side without the other breaks the
# gate. A fused row with no resolvable hop gets 0.0 -- below both, so it demotes.
LINK_CONF_HIGH = 1.0
LINK_CONF_LOW = 0.5


def fusion_row(m: Memory, ord_: int) -> Dict[str, Any]:
    """Column dict for one ``FUS_*`` node from ``fusion.apply_fusion``.

    A synthesized row states no verdict and no pointer: its whole chain is in
    ``sources`` (which is where its supersedes edges come back from), and
    ``build_store`` reads node_kind=="fusion" as decision_type "changed". It
    carries no ``link_conf`` either, which is faithful -- a fused row never made
    a point-to-point claim, so there was nothing to grade. Its TIER is therefore
    derived at load, not stored: ``build_store`` takes the minimum over the
    chain's own hops. Leaving it NULL and letting the assembly gate read that as
    "demoted" was a bug, not a policy -- it silently retired every fused row from
    headship, which is the one thing ``fusion.py`` exists to grant.

    ``sources`` IS A LIST HERE -- the neutral form, and the form ``build_store``
    reads. A backend that needs it serialized serializes it on the way out
    (pgvector's ``fusion_row`` wraps this one for its JSONB column). Getting that
    backwards is silent: ``build_store`` does ``list(d["sources"] or [])``, so a
    JSON STRING becomes a list of single characters and the chain's edges quietly
    vanish.
    """
    return {
        "memory_id": m.memory_id, "node_kind": "fusion",
        "channel": m.channel, "ch_ord": None, "ord": ord_,
        "author": m.author, "role": m.role,
        "timestamp": m.timestamp.isoformat() if m.timestamp.year > 1 else None,
        "reply_to": None,
        "content": m.content, "rewritten": m.content_rewritten,
        "keep": True, "verdict": None, "slot": None,
        "decision_quote": m.decision_new_value,
        "topic": m.topic, "phase": None, "phase_name": m.phase_name,
        "supersedes": None,
        "sources": list(m.sources or []),
        "link_conf": None,
    }


class VectorIndex:
    """Top-k over node ids. The one thing ``FusionRetriever`` needs.

    ``search`` must return ids ordered MOST SIMILAR FIRST, must never return an
    id the store did not load (the retriever's reranker indexes the result
    unguarded -- a stale id there cost 3561 KeyErrors across 1345 LoCoMo
    questions once), and must honour ``candidate_ids=None`` as "the whole store"
    rather than "no candidates".
    """

    def search(self, qvec: Sequence[float], k: int,
               candidate_ids: Optional[Sequence[str]] = None) -> List[str]:
        raise NotImplementedError

    def close(self) -> None:      # optional; a backend holding no handle passes
        return None


def derive_phase_timeline(store: MemoryStore) -> None:
    """Fill phase_start / phase_order from timestamps, per channel.

    The writer NAMES phases but cannot order them: it sees one message at a
    time. So ordering is derived at load -- phase_start = its earliest member's
    timestamp, phase_order = the rank of phases by that start, scoped per
    channel because channels run concurrently and their phase numbering is
    independent. Two persisted columns could go stale; this cannot.

    Nodes that already carry a non-negative `phase_order` are left alone (a
    dataset that ships phase ordinals keeps them).
    """
    chan_phase_start: Dict[str, Dict[str, Any]] = {}
    for mid in store.order:
        m = store.memories[mid]
        if (m.phase_order is not None and m.phase_order >= 0) or not m.phase_name:
            continue
        starts = chan_phase_start.setdefault(m.channel or "", {})
        cur = starts.get(m.phase_name)
        if cur is None or m.timestamp < cur:
            starts[m.phase_name] = m.timestamp
    chan_order: Dict[str, Dict[str, int]] = {}
    for ch, starts in chan_phase_start.items():
        ranked = sorted(starts.items(), key=lambda kv: kv[1])
        chan_order[ch] = {ph: i for i, (ph, _) in enumerate(ranked)}
    for mid in store.order:
        m = store.memories[mid]
        if (m.phase_order is not None and m.phase_order >= 0) or not m.phase_name:
            continue
        ch = m.channel or ""
        if m.phase_name in chan_phase_start.get(ch, {}):
            m.phase_start = chan_phase_start[ch][m.phase_name]
            m.phase_order = chan_order[ch][m.phase_name]


def build_store(rows: Iterable[Mapping[str, Any]],
                keep_noise: bool = False) -> Tuple[MemoryStore, List[str]]:
    """Rows (in ``ord`` order) -> (MemoryStore, id_by_index).

    THE ROW IS THE NODE. The write side used to produce records and a second
    function rebuilt a store from them plus the corpus; now there is one mapping
    and nothing to keep in sync. The row's columns are the version contract: a
    store written by an older writer fails on the field access rather than
    loading half-populated.

    Six read-side fields are DERIVED here rather than stored, each because it
    restates a column the writer already wrote:

      is_noise          not keep
      is_decision_point verdict IS NOT NULL, or a fusion row (which states no
                        verdict -- its whole claim is the chain in ``sources``)
      decision_type     "changed" for a revision or a fused row, else "original"
                        on a decision, else None
      decision_new_value  decision_quote (the same span, under the read-side name)
      link_confidence   the link_conf tier, numerically: high -> 1.0, low -> 0.5,
                        so ``memory.versioning``'s threshold can re-split what the
                        writer graded. On a FUSED row, whose claim is a chain and
                        not a pair, it is the MINIMUM over that chain's hops --
                        see the fusion branch below
      decision_prior_value  FOLLOWS THE POINTER: the superseded row's own
                        decision_quote. One fact, one place. Where the target
                        carries no quote the revision simply has no prior value --
                        the same state a demoted edge reaches at assembly time.

    EDGES COME FROM THE COLUMNS: ``reply_to``, a fusion row's ``sources``, and a
    raw row's ``supersedes``. Insertion order is reproduced too, because
    ``superseded_by`` is last-write-wins for a node with several superseders: all
    replies_to first, then supersedes in ``ord`` order -- and a fused row's ``ord``
    is ``fusion.FUSION_ORD_BASE`` plus its chain root's, so FUS_ rows sort after
    every message row and one pass covers both phases in the original sequence.
    That last-write-wins is also why ``fusion._chains`` does not read
    ``superseded_by``: attaching a fused row overwrites it on every hop.

    Noise rows may stay in the caller's storage (the writer needs them to resume)
    and are dropped HERE by default, so ``ord`` has gaps. Nothing reads it as an
    index; the only contract is relative order.
    """
    store = MemoryStore()
    sup: Dict[str, str] = {}
    for d in rows:
        keep = True if d["keep"] is None else bool(d["keep"])
        if not keep and not keep_noise:
            continue
        kind = d["node_kind"] or "raw"
        verdict = d["verdict"]
        # THE decision flag, derived: a raw row is a decision iff it reached a
        # verdict, and a fusion row always is even though it states none.
        is_dec = verdict is not None or kind == "fusion"
        mem = Memory(
            memory_id=d["memory_id"], content=d["content"] or "",
            author=d["author"] or "", role=d["role"] or "",
            channel=d["channel"], phase_name=d["phase_name"], topic=d["topic"],
            timestamp=_parse_ts(d["timestamp"]),
            msg_node=d["memory_id"],
            reply_to=d["reply_to"],
            is_noise=not keep,
            is_decision_point=is_dec,
            decision_type=("changed" if (verdict == "revision" or kind == "fusion")
                           else ("original" if is_dec else None)),
            decision_new_value=d["decision_quote"],
            embedding=None,  # vectors stay in the backend, not in the graph
        )
        mem.content_rewritten = d["rewritten"]
        mem.node_kind = kind
        # `sources` arrives DECODED (a list), not as JSON text: the pgvector
        # backend stores it as JSONB and psycopg2 decodes on the way in, so a
        # backend that persists it as a string parses before calling here.
        mem.sources = list(d["sources"] or [])
        # A pointer without a revision verdict cannot happen -- `_commit` only
        # ever sets one on the branch that also sets the verdict -- but the edge
        # is built off BOTH so a hand-edited row cannot invent one.
        if d["supersedes"] and verdict == "revision":
            sup[d["memory_id"]] = d["supersedes"]
            mem.link_confidence = (LINK_CONF_HIGH if d["link_conf"] == "high"
                                   else LINK_CONF_LOW)
        store.add(mem)

    # `store.link` returns early when either end is missing, which IS the noise
    # guard: a message whose parent was filtered out gets no edge.
    for mid in store.order:
        rt = store.memories[mid].reply_to
        if rt:
            store.link(mid, rt, Relation.REPLIES_TO)
    for mid in store.order:
        m = store.memories[mid]
        if m.node_kind == "fusion":
            # A FUSED ROW'S TIER IS ITS WEAKEST HOP'S. The row supersedes its
            # whole chain, so `versioning.regate_supersedes` has to gate it like
            # any other edge -- but the tier cannot be read off the row, because
            # it carries no `link_conf` of its own (see `fusion_row`): it never
            # made a point-to-point claim. Its claim is the CHAIN, so what the
            # chain is worth is its weakest link.
            #
            # Exempting fused rows instead, or taking the max, would hand every
            # low-tier link exactly the authority the threshold exists to refuse,
            # via a row the writer never graded.
            #
            # `sources` is oldest-first, so [0] is skipped: the root supersedes
            # nothing and is graded on nothing. An unresolvable hop (one filtered
            # out as noise) leaves the chain unverifiable, which DEMOTES -- it
            # does not default to trusted.
            confs = [getattr(store.memories.get(s), "link_confidence", None)
                     for s in m.sources[1:]]
            m.link_confidence = (min(confs) if confs
                                 and all(c is not None for c in confs) else 0.0)
            for dst in m.sources:
                store.link(mid, dst, Relation.SUPERSEDES)
            # A fused row's prior value is its chain HEAD's -- the newest hop's,
            # which the raw pass below has already resolved, because FUS_ rows
            # sort last by `ord`.
            head = store.memories.get(m.sources[-1]) if m.sources else None
            if head is not None:
                m.decision_prior_value = head.decision_prior_value
        elif mid in sup:
            store.link(mid, sup[mid], Relation.SUPERSEDES)
            prior = store.memories[sup[mid]].decision_new_value
            if prior:
                m.decision_prior_value = prior

    derive_phase_timeline(store)
    return store, list(store.order)
