"""Append-only graph memory schema for group-conversation memory.

Design goals validated by this prototype:
  - every memory carries speaker/scope/version/source as first-class fields
  - relations (replies_to / supersedes / supplements) are edges, not mutations
  - memory state (confirmed / superseded) is DERIVED from edges, never set by hand
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set


class Relation(str, Enum):
    REPLIES_TO = "replies_to"      # thread structure (from reply_to)
    SUPERSEDES = "supersedes"      # newer decision overrides an older one
    SUPPLEMENTS = "supplements"    # same topic, complementary view


# NO `State` ENUM. A confirmed/superseded enum and a `Memory.state` property
# derived from it used to live here, and nothing read either one: every consumer
# tests `mem.superseded_by` directly, which is the edge the enum was derived
# from. format_passage renders status from the same field.


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.min


@dataclass
class Edge:
    src: str            # memory_id
    dst: str            # memory_id
    relation: Relation


@dataclass
class Memory:
    """A single append-only memory node."""

    memory_id: str
    content: str
    # ---- speaker grounding ----
    author: str = ""
    role: str = ""
    # ---- scope ----
    channel: Optional[str] = None
    phase_name: Optional[str] = None
    # The topic LABEL is agent-facing: read tools hand it to the model and accept
    # it back as a scope value, so it has to stay meaningful. Relabelling it with
    # opaque ids cut get_timeline calls 12 -> 1 and cost 13 temporal points.
    #
    # A separate retrieval-only partition (`topic_group`) used to live next to it,
    # feeding the retriever's chain-head injection. It is gone with that
    # injection, and it had never been loaded in the first place -- it is not one
    # of the columns pg_store selects, so it was None on every node at QA time and
    # the retriever's `topic_group or topic` always fell back to this label.
    topic: Optional[str] = None
    # ---- phase timeline (structure-aware retrieval) ----
    phase_order: int = -1                 # ordinal of the phase within its channel
    phase_start: datetime = field(default_factory=lambda: datetime.min)
    # ---- time ----
    timestamp: datetime = field(default_factory=lambda: datetime.min)
    # ---- source / provenance ----
    msg_node: Optional[str] = None
    reply_to: Optional[str] = None
    is_noise: bool = False
    is_decision_point: bool = False
    decision_type: Optional[str] = None   # original / changed / ...
    # decision-change payload: `new` is this message's own value quote, `prior`
    # is read off the row it supersedes (pg_store follows the pointer at load
    # time). The pair is what the fused row renders and what a
    # knowledge-update question is answered from.
    decision_prior_value: Optional[str] = None
    decision_new_value: Optional[str] = None
    # ---- vector ----
    embedding: Optional["list"] = None
    # ---- graph adjacency (memory_id lists) ----
    replies_to_parent: Optional[str] = None
    reply_children: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None       # points to newer memory
    supersedes: List[str] = field(default_factory=list)
    supplements: List[str] = field(default_factory=list)
    # ---- written by the LLM write path; all defaulted ----
    content_rewritten: Optional[str] = None   # rewritten text (dates resolved,
                                              # references de-referenced, ...)
    node_kind: str = "raw"                    # "raw" | "fusion"
    sources: List[str] = field(default_factory=list)  # fusion node: its hop chain
    # Mechanical check of this node's supersedes edge (quote overlap). None means
    # the edge was not graded, which the read side treats as trusted; graded edges
    # are gated on a threshold there.
    link_confidence: Optional[float] = None
    # Business supersession is independent of the retrieval graph: fusion intake
    # must not mark the newest source message as superseded. Derived only from the
    # raw message's SUPERSEDES edge; not written to the database.
    decision_superseded_by: Optional[str] = None

    def clean_content(self) -> str:
        return html.unescape(self.content or "").strip()

    def index_text(self, source: str = "verbatim") -> str:
        """Text the retrieval indexes (dense/BM25) are built on.
        ``rewritten`` falls back to verbatim when no rewrite exists, so a
        mixed store (e.g. rewrite failed on some messages) never indexes
        empty strings."""
        if source == "rewritten":
            rw = (self.content_rewritten or "").strip()
            if rw:
                return rw
        return self.clean_content()


class MemoryStore:
    """Append-only graph store. No update/delete of node content."""

    def __init__(self) -> None:
        self.memories: Dict[str, Memory] = {}
        self.edges: List[Edge] = []
        # scope indexes for fast pre-filtering
        self.by_channel: Dict[str, List[str]] = {}
        self.by_phase: Dict[str, List[str]] = {}
        self.by_topic: Dict[str, List[str]] = {}
        # ordering: memory_id -> insertion position (for msg index mapping)
        self.order: List[str] = []

    # ---- write path (append only) ----
    def add(self, mem: Memory) -> str:
        if mem.memory_id in self.memories:
            return mem.memory_id
        self.memories[mem.memory_id] = mem
        self.order.append(mem.memory_id)
        if mem.channel:
            self.by_channel.setdefault(mem.channel, []).append(mem.memory_id)
        if mem.phase_name:
            self.by_phase.setdefault(mem.phase_name, []).append(mem.memory_id)
        if mem.topic:
            self.by_topic.setdefault(mem.topic, []).append(mem.memory_id)
        return mem.memory_id

    def link(self, src: str, dst: str, relation: Relation) -> None:
        """Create an edge. src/dst must already exist."""
        if src not in self.memories or dst not in self.memories:
            return
        self.edges.append(Edge(src, dst, relation))
        s, d = self.memories[src], self.memories[dst]
        if relation is Relation.REPLIES_TO:
            # src replies to dst  => dst is parent
            s.replies_to_parent = dst
            d.reply_children.append(src)
        elif relation is Relation.SUPERSEDES:
            # src supersedes dst  => dst.superseded_by = src
            s.supersedes.append(dst)
            d.superseded_by = src
            if s.node_kind != "fusion":
                d.decision_superseded_by = src
        elif relation is Relation.SUPPLEMENTS:
            s.supplements.append(dst)

    # ---- read helpers ----
    def get(self, memory_id: str) -> Optional[Memory]:
        return self.memories.get(memory_id)

    def thread_members(self, memory_id: str, depth: int = 2) -> Set[str]:
        """Expand the evidence neighborhood: reply chain (up ancestors + down
        descendants) PLUS cross-thread `supplements` links (link_related), so a
        multi-hop question can follow a depends_on/answers/supplements edge to a
        message in another thread/topic."""
        members: Set[str] = {memory_id}
        # up (reply ancestors)
        cur = self.memories.get(memory_id)
        up = 0
        while cur and cur.replies_to_parent and up < depth:
            members.add(cur.replies_to_parent)
            cur = self.memories.get(cur.replies_to_parent)
            up += 1
        # down (BFS over reply children + supplements neighbors)
        queue = [(memory_id, 0)]
        seen = {memory_id}
        while queue:
            mid, d = queue.pop(0)
            if d >= depth:
                continue
            m = self.memories.get(mid)
            if not m:
                continue
            for c in list(m.reply_children) + list(m.supplements):
                if c not in seen:
                    seen.add(c)
                    members.add(c)
                    queue.append((c, d + 1))
        return members

    def current_version(self, memory_id: str) -> str:
        """Follow superseded_by chain to the latest confirmed memory."""
        seen = set()
        cur = memory_id
        while cur in self.memories and self.memories[cur].superseded_by:
            if cur in seen:
                break
            seen.add(cur)
            cur = self.memories[cur].superseded_by
        return cur

    def stats(self) -> Dict[str, int]:
        rel_counts: Dict[str, int] = {}
        for e in self.edges:
            rel_counts[e.relation.value] = rel_counts.get(e.relation.value, 0) + 1
        superseded = sum(1 for m in self.memories.values() if m.superseded_by)
        roots = sum(1 for m in self.memories.values()
                    if m.replies_to_parent is None and m.reply_children)
        return {
            "memories": len(self.memories),
            "edges": len(self.edges),
            "channels": len(self.by_channel),
            "phases": len(self.by_phase),
            "topics": len(self.by_topic),
            "thread_roots": roots,
            "superseded_nodes": superseded,
            **{f"edge_{k}": v for k, v in rel_counts.items()},
        }
