"""The decision REGISTRY: what a channel currently holds, and how it is quoted.

Zero LLM, and the whole of the writer's state. `agent_write.MemoryWriter` owns
one of these per channel and is the only caller.

  ground_quote     a field is only kept if it is a verbatim span of THIS message
  Slot             one settled value: its subject, its current text, and the
                   message currently HOLDING it
  DecisionRegistry the channel's live slots + their retriever (every slot
                   competes; ordering on lexical.fuse, no relevance floor)
  _fmt_registry    the [REGISTRY] block stage-1 reads
  _fmt_thread      the [THREAD] block, the reply chain of the current message

Registry semantics (matches the gold edge convention "latest valid statement"):
  restate  -> pointer moves to this message;
  new      -> register slot;
  revision -> supersedes = slot holder BEFORE update.
Which of the three a message gets is decided by `agent_write._commit`; this
module only provides the operations and keeps them consistent.

There is no store builder here, and there is none anywhere on the write side:
the rows the writer commits ARE the read-side store, loaded by
pg_store.load_light_store, where a `supersedes` pointer is a hard graph edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..utils import lexical

# ---------------------------------------------------------------- grounding

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def token_overlap(a: str, b: str) -> float:
    """|tokens(a) ∩ tokens(b)| / |tokens(a)| -- how much of `a` is grounded in
    `b`. Directional on purpose: the writer asks whether a quote it produced is
    supported by the source message, not whether the two are similar.

    Its own tokenizer, not `lexical.tokenize`: this is a GROUNDING check on a
    [0,1] scale, and it must keep measuring exactly what `ground_quote`'s 0.9 was
    set against, independently of changes to the retrieval surface.
    """
    ta = set(_TOKEN_RE.findall((a or "").lower()))
    if not ta:
        return 0.0
    tb = set(_TOKEN_RE.findall((b or "").lower()))
    return len(ta & tb) / len(ta)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def ground_quote(quote: Optional[str], content: str,
                 min_overlap: float = 0.9) -> Optional[str]:
    """Return the quote iff it is verbatim-grounded in content, else None.

    Accepts exact (whitespace-normalized) substring, or >=min_overlap token
    overlap for minor punctuation drift.
    """
    if not quote or not isinstance(quote, str):
        return None
    q = quote.strip().strip('"').strip()
    if not q:
        return None
    if _norm(q) in _norm(content):
        return q
    if token_overlap(q, content) >= min_overlap:
        return q
    return None


# ---------------------------------------------------------------- registry

@dataclass
class Slot:
    slot_id: int
    subject: str
    quote: str            # current value, verbatim from holder message
    holder: str           # msg_node currently holding the value
    vec: Optional[List[float]] = None
    revisions: int = 0
    history: List[str] = field(default_factory=list)   # holder mids, oldest first
    topic: str = ""       # workstream label of the message that opened the slot;
                          # surfaced in [REGISTRY] so later messages reuse it
    phase: str = ""       # progress label of the slot's CURRENT holder, updated
                          # on revise/restate. Surfaced in [REGISTRY] so later
                          # messages COPY an existing phase label instead of
                          # minting one: invisible labels fragment the label
                          # space, which silently disables the read side's
                          # by_phase bucket.


class DecisionRegistry:
    """Per-channel decision state. Retrieval = BM25 fused with cosine, unfiltered."""

    def __init__(self, embed_fn: Callable[[List[str]], List[List[float]]]):
        self.slots: List[Slot] = []
        self._embed = embed_fn
        # mid -> that message's REWRITTEN-text vector, injected by the writer
        # (MemoryWriter wires it to its own _cvec). A slot's vector is the vector
        # of the message currently holding the value, built on the rewritten
        # text -- the same representation the read side indexes and searches.
        # Nothing embeds verbatim text: no vector in the store derives from the
        # original message body.
        self.vec_of: Optional[Callable[[str], Optional[List[float]]]] = None
        # BM25 statistics over the SLOT documents, for the lexical channel of
        # `retrieve`. Maintained incrementally in `_reindex` rather than
        # recomputed per query: retrieval runs ~1.2x per message while slots
        # change once, so a rebuild would turn a per-slot cost into a per-query
        # one. Channel-local, because the writer is per channel.
        self._doc: Dict[int, List[str]] = {}      # slot_id -> tokens
        self._df: Dict[str, int] = {}
        self._ndocs = 0
        self._dl_sum = 0

    def _reindex(self, slot: "Slot") -> None:
        """Re-tokenize one slot's document and update the BM25 tables.

        Called on register and on revise (which rewrites `quote`); restate only
        moves the holder, so the document text is unchanged and it does not call
        this.
        """
        old = self._doc.pop(slot.slot_id, None)
        if old is not None:
            self._ndocs -= 1
            self._dl_sum -= len(old)
            for w in set(old):
                self._df[w] = self._df.get(w, 1) - 1
        t = lexical.tokenize(slot.subject + " " + slot.quote)
        self._doc[slot.slot_id] = t
        self._ndocs += 1
        self._dl_sum += len(t)
        for w in set(t):
            self._df[w] = self._df.get(w, 0) + 1

    @staticmethod
    def _cos(a, b) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        return sum(a[i] * b[i] for i in range(n))

    def _holder_vec(self, holder: str) -> Optional[List[float]]:
        """Rewritten-text vector of the message currently holding the value.

        Resolved at RETRIEVE time, not stored on the slot: a slot's holder is
        set inside `_commit`, which runs while the writer's own `process` is
        still on the stack, so the holder's rewritten vector does not exist yet
        at register/revise time. Looking it up lazily also means the vector
        follows the value automatically when a revision moves the holder.

        None is normal, not an error: a message whose rewrite came back empty
        gets no vector at all (rather than falling back to embedding its
        verbatim body), and `_cos` treats a missing vector as 0.0, so such a
        slot is simply ranked on the lexical term alone.
        """
        if self.vec_of is None or not holder:
            return None
        try:
            return self.vec_of(holder)
        except Exception:      # noqa: BLE001 -- ranking aid, never fatal
            return None

    def retrieve(self, text: str, qvec: Optional[List[float]],
                 k: int = 6) -> List[Slot]:
        """Top-k slots for this message: BM25 fused with cosine, no relevance floor.

        `text` is the verbatim message body, and it is what the SPARSE channel
        scores with. Scoring the agent's keyword query there instead is no better,
        because BM25's IDF already discounts the filler a rewrite would drop -- and
        the body does not depend on a per-run LLM artefact.

        THERE IS NO ADMISSION FLOOR. A floor would be inert here: `k` truncates the
        result afterwards anyway, and any two texts from the same channel already
        score well above a plausible cutoff, so a floor almost never removes a slot
        -- it only risks returning EMPTY, which callers would misread. Precision is
        therefore the agent's choice among these k. An empty return means only that
        the channel has no slots yet, not "nothing here matches".

        With qvec=None (the agent searched with an empty query) there is no dense
        channel to fuse and the order is BM25 alone; otherwise the two channels go
        through groupmem/utils/lexical.fuse -- weighted RRF, w_lex=2.0, the read
        side's fusion as well.
        """
        if not self.slots:
            return []
        q_toks = lexical.tokenize(text)
        avgdl = (self._dl_sum / self._ndocs) if self._ndocs else 1.0
        lex_scored, cos_scored = [], []
        for s in self.slots:
            lex_scored.append((lexical.bm25(
                q_toks, self._doc.get(s.slot_id) or [],
                self._df, self._ndocs, avgdl), s.slot_id))
            if qvec is not None:
                cos_scored.append(
                    (self._cos(qvec, self._holder_vec(s.holder)), s.slot_id))
        if qvec is None:
            order = [sid for _sc, sid in
                     sorted(lex_scored, key=lambda t: -t[0])][:k]
        else:
            order = lexical.fuse(lex_scored, cos_scored, k)
        by_id = {s.slot_id: s for s in self.slots}
        return [by_id[i] for i in order if i in by_id]

    def register(self, subject: str, quote: str, holder: str,
                 topic: str = "", phase: str = "") -> Slot:
        s = Slot(slot_id=len(self.slots), subject=subject, quote=quote,
                 holder=holder, vec=None,
                 history=[holder], topic=topic or "", phase=phase or "")
        self.slots.append(s)
        self._reindex(s)
        return s

    def get(self, slot_id) -> Optional[Slot]:
        try:
            sid = int(slot_id)
        except (TypeError, ValueError):
            return None
        if 0 <= sid < len(self.slots):
            return self.slots[sid]
        return None

    def restate(self, slot: Slot, holder: str, phase: str = "") -> None:
        slot.holder = holder
        slot.history.append(holder)
        if phase:                    # phase follows the CURRENT holder
            slot.phase = phase

    def revise(self, slot: Slot, new_quote: str, holder: str,
               phase: str = "") -> str:
        """Returns the superseded holder mid (BEFORE the update)."""
        old_holder = slot.holder
        slot.quote = new_quote or slot.quote
        slot.holder = holder
        if phase:
            slot.phase = phase
        slot.revisions += 1
        slot.history.append(holder)
        self._reindex(slot)           # the document text just changed
        # vector is not stored: retrieve() resolves it from the new holder
        return old_holder


# ---------------------------------------------------------------- rendering

def _fmt_registry(slots: List[Slot]) -> str:
    if not slots:
        return "(no related slots — registry has nothing on this yet)"
    rows = []
    for s in slots:
        # `topic` is shown so the model can REUSE an existing workstream label
        # instead of minting a new one. Without it the label space fragments to
        # roughly one topic per message, which disables topic-scoped lookups
        # like get_current_decision: the revision and the decision it overturns
        # land in different topics, so the topic's chain head stays the STALE
        # value.
        topic = getattr(s, "topic", "") or ""
        # `phase` is shown for the same reason as `topic`: the writer can only
        # reuse a label it can see. Unlike topic it tracks the current holder, so
        # the visible vocabulary stays weighted toward recent stages.
        phase = getattr(s, "phase", "") or ""
        rows.append(f"slot_id={s.slot_id} | subject: {s.subject} | "
                    + (f"topic: {topic} | " if topic else "")
                    + (f"phase: {phase} | " if phase else "")
                    + f"current_value_quote: \"{s.quote}\" | holder_msg: {s.holder} "
                    f"| revisions: {s.revisions}")
    return "\n".join(rows)


# [THREAD] QUOTES ARE NOT TRUNCATED. A character cap on them is a trap: decision
# quotes are long (median 170 characters, p90 349), and a decision's date/owner/
# number tends to sit at the END of the sentence, which is exactly what a
# mid-quote cut removes.
#
# THERE IS NO CHANNEL-WINDOW FALLBACK for a message that opens a thread; it sees
# "(start of thread)". Showing the most recent messages of the same CHANNEL
# instead pushes over-judgement on the domains that already over-judge, and the
# block has to disclaim itself in the prompt ("the current message does NOT
# answer them") because the model otherwise reads channel neighbours as what the
# speaker was replying to.

def _fmt_thread(chain: List[Dict[str, Any]]) -> str:
    if not chain:
        return "(start of thread)"
    rows = []
    for r in chain[-4:]:
        # `verdict IS NOT NULL` is the decision flag. The records carry no
        # separate is_decision: it was the same predicate stored twice.
        flag = "DECISION" if r.get("verdict") else "msg"
        q = (r.get("decision_quote") or r.get("content") or "")
        rows.append(f"{r['msg_node']} @{r.get('author')} {r.get('timestamp')} "
                    f"[{flag}] \"{q}\"")
    return "\n".join(rows)
