"""Fusion nodes: one retrievable row per supersedes chain.

One row per supersedes CHAIN, holding "current decision + evolution timeline"
rendered from the chain's structured deltas (prior_value / new_value extracted
per-hop at write time). Zero LLM, zero API: every string here is a concatenation
and the vector is the mean of hop vectors the writer already committed.

Consequences, all via EXISTING retrieval machinery (no retriever changes):
  - the fused node is the chain's only head -> gets decision_boost, topic-head
    injection and is the target of current-version injection;
  - the source messages get the superseded 0.5x penalty + SUPERSEDED_BY status
    tags, but stay indexed (temporal questions still need old values);
  - fused ids are APPENDED to id_by_index, so original message indices are
    unchanged.

THE FUSED ROW IS KEYED ON THE CHAIN'S ROOT, NOT ITS HEAD, which is why this file
can be re-run: a chain grows, and keying on the root means growth REWRITES one
row in place and the table converges (keying on the head would orphan a stale
duplicate on every revision).

There are TWO callers and ONE rendering. ``apply_fusion`` is the store-shaped one
(``bench/finalize.py``, whole domain at a time); ``build_fused`` is the
field-shaped one the writer calls per message. Both go through ``build_fused``,
so there is no second place that knows what a fused row contains.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..schema import Memory, MemoryStore, Relation

FUSION_PREFIX = "FUS_"

# Fused rows must sort AFTER every message row, because `load_light_store`
# rebuilds edges in `ord` order and `superseded_by` is last-write-wins: a fused
# row has to be the last thing to claim its hops. Corpus positions are message
# counts (~3e4 at the largest), so the offset both guarantees that and leaves
# `ord` a pure function of the root -- which is what lets the per-message writer
# and finalize compute the same value for the same row without consulting each
# other.
FUSION_ORD_BASE = 10_000_000


def fused_id(root: str) -> str:
    """The chain's row id, from the one node on it that cannot change."""
    return FUSION_PREFIX + root


def fused_ord(root_ord: int) -> int:
    return FUSION_ORD_BASE + max(0, int(root_ord))


@dataclass
class Hop:
    """One node of a chain, as the render needs it -- no store, no DB.

    ``indexed`` is the text retrieval sees for that hop (``rewritten``, or the
    verbatim body where the rewrite came back empty); the caller resolves that
    fallback, because the two callers read it from different places (a Memory's
    ``index_text`` vs. a record's columns).
    """
    mid: str
    indexed: str = ""
    author: str = ""
    role: str = ""
    channel: Optional[str] = None
    phase_name: Optional[str] = None
    topic: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.min)
    new_value: Optional[str] = None
    prior_value: Optional[str] = None

    @property
    def date(self) -> str:
        return (self.timestamp.date().isoformat()
                if self.timestamp.year > 1 else "?")

    @property
    def flat(self) -> str:
        return (self.indexed or "").replace("\n", " ")


def _parse_ts(s: Any) -> datetime:
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.min


def hop_from_record(rec: Dict[str, Any],
                    prior_value: Optional[str] = None) -> Hop:
    """A Hop from a writer RECORD -- the per-message path.

    The ``indexed`` fallback mirrors ``Memory.index_text("rewritten")`` exactly
    (rewritten if it has content, else the unescaped verbatim body) because the
    other caller reads it through that method, and the two paths have to render
    the same row for the same chain or the reconciliation pass would rewrite
    every row it inspects.

    ``prior_value`` is passed in rather than read off the record: a record does
    not store the value it replaced (``pg_store`` resolves it by following the
    ``supersedes`` pointer at load), and on a chain the answer is simply the
    previous hop's own value.
    """
    rw = (rec.get("rewritten") or "").strip()
    return Hop(mid=rec["msg_node"],
               indexed=rw or html.unescape(rec.get("content") or "").strip(),
               author=rec.get("author") or "", role=rec.get("role") or "",
               channel=rec.get("channel"), phase_name=rec.get("phase_name"),
               topic=rec.get("topic"), timestamp=_parse_ts(rec.get("timestamp")),
               new_value=rec.get("decision_quote"), prior_value=prior_value)


def mean_unit(vecs: Sequence[Sequence[float]]) -> List[float]:
    """A fused row's vector: the component-wise mean of its hops', re-normalised.

    THE WHOLE WRITE PATH EMBEDS IN ONE PLACE (the per-message commit), so this is
    how a fused row gets a vector without an API call. Both are legitimate
    document vectors under the ``<=>`` cosine operator the reader uses -- cosine
    normalises, so the shorter mean vector is not penalised -- but they are NOT
    the same vector: ``mean(embed(a), embed(b))`` discards the cross-attention
    ``embed(a + b)`` would have applied. That is a behavioural choice, not an
    identity.

    The re-normalisation is redundant for cosine ranking; it is here so every row
    in the table has ``|v| = 1`` like the writer's rows do, and a future
    inner-product operator cannot quietly demote every fused row for being short.

    THE TWO CALLERS DO NOT GET BITWISE-IDENTICAL VECTORS. The per-message path
    averages the hop vectors it still holds in memory; finalize averages the same
    vectors read back out of the table, where ``vec_literal`` stored them at 6
    decimals. So a re-render can move a component by ~1e-6, and the change
    detection in ``apply_fusion`` deliberately does not look at the vector -- it
    compares the strings and the source list, because those are what a stale row
    actually gets wrong.
    """
    n = len(vecs)
    acc = [0.0] * len(vecs[0])
    for v in vecs:
        for i, x in enumerate(v):
            acc[i] += x
    acc = [x / n for x in acc]
    norm = math.sqrt(sum(x * x for x in acc))
    return acc if norm == 0.0 else [x / norm for x in acc]


def render_content(hops: List[Hop]) -> str:
    """Human-readable provenance render, stored in ``content`` ONLY.

    The read side never reads ``content`` -- it indexes and displays
    ``index_text("rewritten")`` -- so this template is free: it costs disk and
    buys a chain you can read straight out of the table when something looks
    wrong. Putting it in the indexed string instead would charge for it twice,
    once in the vector and once in the agent's context window.
    """
    head = hops[-1]
    current = head.new_value or head.flat[:300]
    lines = [f"[CURRENT DECISION] {current}",
             f"Decision evolution for topic '{head.topic or '?'}':"]
    for i, h in enumerate(hops, 1):
        if i == 1 and not h.new_value:
            lines.append(f"{i}. {h.date} {h.author or '?'}: {h.flat[:200]} "
                         f"(initial, {h.mid})")
            continue
        val = h.new_value or h.flat[:200]
        seg = f"{i}. {h.date} {h.author or '?'}: "
        seg += f"changed to {val}" if i > 1 else f"{val} (initial)"
        if h.prior_value and i > 1:
            seg += f" (from: {h.prior_value})"
        seg += f" ({h.mid})"
        lines.append(seg)
    lines.append("Sources: " + ", ".join(h.mid for h in hops))
    return "\n".join(lines)


def render_index(hops: List[Hop]) -> str:
    """What retrieval actually sees: the chain's own rewritten texts, oldest
    first, and nothing else.

    Everything a template could add here is ALREADY A COLUMN -- topic, author,
    timestamp, decision_new_value, sources -- and the read side reads those as
    columns, not as prose. Scaffolding in this string would be paid for twice
    (vector + context window) and, worse, `(from: <old value>)` re-states the
    previous hop's value while the previous hop sits on the line above it, so
    every value would be embedded two or three times over.

    What survives is the one property fusion exists to buy: a whole chain in ONE
    retrieval unit, so the reranker spends one passage slot on it instead of the
    many it would take to reassemble the chain from raw rows.
    """
    return "\n".join(h.flat for h in hops)


def build_fused(hops: List[Hop]) -> Memory:
    """The fused node for one chain, oldest hop first. THE only renderer.

    Every scalar comes off the HEAD (it is the current decision) except the
    timestamp, which is the newest on the chain -- the same thing when hops are
    in time order and the honest answer when a corpus is not.
    """
    head = hops[-1]
    newest = max((h.timestamp for h in hops if h.timestamp.year > 1),
                 default=head.timestamp)
    fid = fused_id(hops[0].mid)
    return Memory(
        memory_id=fid,
        content=render_content(hops),   # A-class assembled by code: provenance
        author=head.author, role=head.role,
        channel=head.channel, phase_name=head.phase_name, topic=head.topic,
        timestamp=newest,
        msg_node=fid,
        is_decision_point=True,
        decision_type="changed",
        decision_prior_value=head.prior_value,
        decision_new_value=head.new_value,
        content_rewritten=render_index(hops),  # the ONLY string retrieval sees
        node_kind="fusion",
        sources=[h.mid for h in hops],
    )


def _chains(store: MemoryStore) -> List[List[str]]:
    """Enumerate supersedes chains (oldest -> newest), heads only.

    A chain's head is the raw node no OTHER RAW NODE supersedes. Headship is
    computed from the raw nodes' own ``supersedes`` lists rather than read off
    ``superseded_by``, and that is not a style choice:

      * ``superseded_by`` is a single field written by every ``link``, so it is
        last-write-wins. Attaching a fused node to its hops writes it on ALL of
        them, so after one fusion pass every hop -- including the chain's real
        head and every node in the middle -- claims to be superseded by the same
        ``FUS_`` row, and the raw pointer that said which node actually replaced
        which is gone.
      * Reading it therefore gives a wrong answer either way: taken at face value
        it hides EVERY chain on a table that already has fused rows; excusing a
        ``FUS_`` superseder instead makes every MID-CHAIN node a head too, so one
        root is rendered once per hop.

    Deriving it from the raw edges has neither failure mode, and it makes this
    function independent of how many times it has already run over the store.

    A ``FUS_`` node is never a head and never covers a hop: it supersedes its
    whole chain, so counting it would either fuse its own output or retire the
    head that produced it.
    """
    covered = {p for mid, m in store.memories.items()
               if not mid.startswith(FUSION_PREFIX)
               for p in m.supersedes
               if not p.startswith(FUSION_PREFIX)}
    heads = [mid for mid, m in store.memories.items()
             if m.supersedes and not mid.startswith(FUSION_PREFIX)
             and mid not in covered]
    chains: List[List[str]] = []
    for head in heads:
        # walk backward through supersedes (each node may supersede several;
        # follow the earliest-linked target which is the canonical prior)
        chain = [head]
        seen = {head}
        cur = head
        while True:
            prevs = [p for p in store.memories[cur].supersedes
                     if p in store.memories and p not in seen
                     and not p.startswith(FUSION_PREFIX)]
            if not prevs:
                break
            # canonical prior = the earliest by timestamp (stable, deterministic)
            prevs.sort(key=lambda x: (store.memories[x].timestamp, x))
            cur = prevs[0]
            seen.add(cur)
            chain.append(cur)
        if len(chain) >= 2:
            chains.append(list(reversed(chain)))  # oldest first
    return chains


def _hop_of(m: Memory) -> Hop:
    return Hop(mid=m.memory_id, indexed=m.index_text("rewritten"),
               author=m.author, role=m.role, channel=m.channel,
               phase_name=m.phase_name, topic=m.topic, timestamp=m.timestamp,
               new_value=m.decision_new_value,
               prior_value=m.decision_prior_value)


def apply_fusion(store: MemoryStore,
                 id_by_index: List[str]
                 ) -> Tuple[MemoryStore, List[str], List[str]]:
    """Bring every chain's fused node up to date. Returns the SAME store, an
    id_by_index with new fused ids appended (original prefix untouched), and the
    ids that were written or rewritten.

    CONVERGENT, not append-only: a chain that grew since the last pass has its
    row REPLACED (same id, because the id is the root), so running this twice in
    a row is a no-op and running it after more ingestion is a repair.
    """
    touched: List[str] = []
    added: List[str] = []
    for chain in _chains(store):
        hops = [_hop_of(store.memories[m]) for m in chain]
        fused = build_fused(hops)
        fid = fused.memory_id
        prev = store.memories.get(fid)
        if prev is not None:
            if (list(prev.sources) == chain
                    and prev.content_rewritten == fused.content_rewritten
                    and prev.content == fused.content):
                continue
            # Replace in place: `store.order` already holds fid, and dropping
            # this node's edges first is not optional -- `link` appends, so
            # re-linking a grown chain over the old edges would leave duplicate
            # SUPERSEDES edges and a `supersedes` list with repeats.
            store.edges = [e for e in store.edges if e.src != fid]
            store.memories[fid] = fused
        else:
            store.add(fused)
            added.append(fid)
        for mid in chain:
            store.link(fid, mid, Relation.SUPERSEDES)
        touched.append(fid)
    if added:
        id_by_index = list(id_by_index) + added
    return store, id_by_index, touched
