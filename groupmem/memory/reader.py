"""Read side: the memory graph exposed as tools, driven by a frozen model.

The graph is exposed as read tools and a frozen strong model chooses which to
call, per question, in a short function-calling loop.

FOUR tools, which is the read-side surface:
  - search_memory(query, k, phase?, topic?, author?, dense_query?)
                                 : fused recall -> cross-encoder rerank ->
                                   automatic anchored hop
  - lookup_phrase(phrase, k)     : literal/BM25 exact-wording lookup
  - get_current_decision(topic)  : current head of a supersedes chain
  - resolve_date(msg_id, phrase) : deterministic relative-date arithmetic

Six tools -- get_thread, get_timeline, get_user_history, hop, list_deadlines,
lookup_assignments -- are deliberately NOT here.

`hop` the TOOL is gone; `_hop_rank` the RANKING is not: search_memory runs it
automatically on its own top-3 hits, so the neighbourhood re-ranking survives
without costing the agent a round.

`memory_id == msg_node` in this store, so ids surfaced in passages (the
`msg_node=` tag) can be fed straight back into the tools that take one.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from .passage import format_passage
from ..utils.dates import resolve_relative_date
from ..configs.defaults import READ_AUTO_HOP_K, READ_MAX_PASSAGES, RERANK_TOPN


# ---- read-side constants ---------------------------------------------------
# Every one of these was an environment variable during development, now frozen.
# The reasoning that fixed each value is kept at the use site.
#
# The three PASSAGE COUNTS keep their env override (defaults unchanged, in
# config.py) because "how many passages does the agent get to see" is a
# legitimate thing to re-price on a new corpus. AUTO_HOP_ANCHORS is not one of
# them: it is how far the hop reaches, not how much the agent sees.
# Frozen to the chain-local scope rendering, with no experimental switch.
DECISION_RENDERER = "chain_scope_v1"
AUTO_HOP_ANCHORS = 3            # top-N hits used as hop anchors inside search_memory
SEARCH_TOPN = RERANK_TOPN       # passages the cross-encoder keeps, per search_memory call
AUTO_HOP_K = READ_AUTO_HOP_K    # extra passages the anchored pass appends to those
# Hard cap for every OTHER tool return. Deliberately not raised to 15: the
# 15 passages search_memory delivers are SEARCH_TOPN + AUTO_HOP_K passed as an
# explicit cap, which bypasses this one. Raising it here would also widen
# lookup_phrase and get_current_decision.
MAX_PASSAGES = READ_MAX_PASSAGES


def _squash(s: str) -> str:
    """Collapse whitespace so a phrase spanning a line break still matches.
    Used only for literal phrase lookup, never for scoring."""
    return re.sub(r"\s+", " ", s or "").strip()


# ------------------------- tool schemas (OpenAI/dashscope function format) ----
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Fusion retrieval over the group-chat memory graph. Your "
                       "entry point: call this first with the question (or a key "
                       "phrase) to get the most relevant passages. Each passage "
                       "shows its msg_node id, topic, phase_name, phase_start, "
                       "author and version status. To narrow a second search, pass "
                       "phase / topic / author (copy the exact value from a "
                       "passage's phase_name= / topic= / user= tag) -- results are "
                       "then restricted to that phase/topic/author, which sharply "
                       "improves precision for temporal and multi-hop questions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keyword search text "
                      "(drives exact-lexical/BM25): the concrete terms -- names, "
                      "module, value, date -- you expect in the passage."},
            "dense_query": {"type": "string", "description": "OPTIONAL semantic "
                      "variant for the vector channel. Memories are stored as "
                      "NARRATIVE sentences stating what was decided/changed. "
                      "Phrase this as a short declarative statement of the fact "
                      "you expect (e.g. 'the team set X to Y'), NOT as a question. "
                      "Leave empty to reuse `query` for both channels."},
            "k": {"type": "integer", "description": "number of passages (default 10)"},
            "phase": {"type": "string", "description": "optional: restrict to this phase_name"},
            "topic": {"type": "string", "description": "optional: restrict to this topic"},
            "author": {"type": "string", "description": "optional: restrict to this author/user id"},
        }, "required": ["query"]}}},
    # `dense_query` is advertised in the schema and in NO system prompt, which is
    # deliberate: it is rarely used, so the dense channel is keyword-driven in
    # practice. Requiring it is not something an observational trace can price --
    # the few calls that carry it come late, after earlier searches failed, so
    # their lower delivery reflects question difficulty, not harm from the field.
    {"type": "function", "function": {
        "name": "lookup_phrase",
        "description": (
            "EXACT-WORDING lookup. Finds passages containing a specific phrase, "
            "by literal substring match first and BM25 term overlap as a "
            "fallback. No semantic/vector matching and no fusion -- so a passage "
            "that contains your words cannot be crowded out by passages that "
            "merely sound related.\n"
            "USE THIS WHEN search_memory keeps returning passages that are about "
            "the right subject but never state the specific thing you need -- a "
            "short rule, an ordering, a status word, a named exception. Vector "
            "search represents a whole passage as one point, so a 4-word detail "
            "inside a 200-character passage barely moves that point and such a "
            "passage can rank below dozens that are merely on-topic. Word "
            "matching does not have that problem.\n"
            "Pass the words you expect to appear IN THE PASSAGE, not the "
            "question. Try the wording the team would have used ('labels first, "
            "logs second'), and if that returns nothing, try fewer or different "
            "words. It is cheap; several attempts are fine."),
        "parameters": {"type": "object", "properties": {
            "phrase": {"type": "string",
                       "description": "the exact words expected in the passage"},
            "k": {"type": "integer",
                  "description": "number of passages (default 10)"},
            "phase": {"type": "string",
                      "description": "optional: restrict to this phase_name"},
            "topic": {"type": "string",
                      "description": "optional: restrict to this topic"},
        }, "required": ["phrase"]}}},
    {"type": "function", "function": {
        "name": "get_current_decision",
        "description": "Return the CURRENT decision(s) for a topic by following "
                       "supersedes edges to the chain head. Use whenever the answer "
                       "depends on the latest decision (it may have changed).",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "a topic string from a passage's topic= tag"},
        }, "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "resolve_date",
        "description": "Compute an ABSOLUTE date (YYYY-MM-DD) from a message's "
                       "timestamp plus a RELATIVE phrase. Use for date questions: find "
                       "the message that states the deadline, then call this with its "
                       "msg_node and the exact relative phrase (e.g. 'by Wednesday', "
                       "'EOD Friday', 'tomorrow', 'end of the week', 'in 3 days'). "
                       "Returns the anchor date (with weekday) and the resolved date. "
                       "Do NOT do weekday arithmetic yourself.",
        "parameters": {"type": "object", "properties": {
            "msg_id": {"type": "string", "description": "msg_node of the message stating the deadline"},
            "phrase": {"type": "string", "description": "the relative time phrase to resolve"},
            "anchor": {"type": "string", "description": "base date: 'msg' (message timestamp, default) or 'phase' (the message's phase_start, for phrasings measured from the phase kickoff)"},
        }, "required": ["msg_id", "phrase"]}}},
]


class ReadTools:
    """Read-only tool implementations over a shared store + retriever.

    All methods return a formatted string (the tool result content). Safe for
    concurrent use: nothing here mutates the store or the retriever.

    Passage bodies come from the rewritten text, matching what the dense and
    BM25 indexes were built on -- matching against a different source than the
    one indexed would report hits the retriever cannot reach.
    """

    def __init__(self, store, retriever, max_passages: int = MAX_PASSAGES):
        self.store = store
        self.retr = retriever
        self.max_passages = max_passages
        self.text_source = "rewritten"
        self._seen: set = set()   # msg_nodes already returned this question

    def tool_schemas(self) -> List[Dict]:
        return TOOL_SCHEMAS

    def new_session(self) -> None:
        """Reset the per-question 'already shown' set, so dedup is scoped to a
        single QA episode. Called once per question by run_qa_agentloop."""
        self._seen = set()

    # ---- helpers ----
    def _passage(self, mem) -> str:
        return format_passage(mem)

    def _fmt(self, ids, cap: Optional[int] = None) -> str:
        cap = cap or self.max_passages
        out = []
        for i, mid in enumerate(ids[:cap], 1):
            mem = self.store.memories.get(mid)
            if mem:
                out.append(f"[{i}] {self._passage(mem)}")
        return "\n\n".join(out) if out else "No matching memory."

    def _fmt_scored(self, ids, cap: Optional[int] = None) -> str:
        """Format retrieved passages in chronological order, deduplicated
        against what this question already saw.

        No per-passage similarity number is printed: on balance it does not steer
        selection either way. High similarity does not imply the passage answers
        the question (unanswerable questions retrieve equally-similar passages),
        so it is not an abstention signal either; the grounding test (can you
        quote the exact answer span?) is what gates abstention. Dropping it also
        skips one embed call per tool call -- and took the `query` parameter with
        it, since nothing left in here reads the question.

        There is deliberately NO cap on cumulative passages per question. Capping
        evidence hurts, especially on term_ambiguity questions, which ask WHICH of
        many similar statements applies: there the surplus passages are the
        disambiguating context, not distractors. Do not retry context reduction on
        this question type.
        """
        cap = cap or self.max_passages
        ids = list(ids)
        if not ids:
            return ("No matching memory -- nothing retrieved. Abstain unless "
                    "another tool surfaces a passage that states the answer.")
        # DEDUP: drop passages already shown earlier this question -- do NOT resend
        # them to the model (saves context; the agent has seen them already).
        new_ids = [mid for mid in ids if mid not in self._seen]
        dropped = len(ids) - len(new_ids)
        if not new_ids:
            return (f"All {len(ids)} retrieved passages were already shown earlier "
                    "this question -- no new evidence. STOP re-searching and decide "
                    "now (answer if grounded, else abstain).")
        ids = new_ids[:cap]
        # present the selected NEW hits in CHRONOLOGICAL order (oldest first, newest
        # last) for EVERY question -- it lets the agent read the conversation
        # sequence and see which statement came later (e.g. the current decision).
        # Selection is still by relevance.
        ids = sorted(ids, key=lambda m: (self.store.memories[m].timestamp
                                         if m in self.store.memories else 0))
        supp = f", {dropped} already-shown suppressed" if dropped else ""
        out = [f"[retrieval] {len(ids)} new passage(s){supp}. Passages are in "
               f"CHRONOLOGICAL order (oldest first, newest last). Only "
               f"answer from these passages at the strength they support: quote "
               f"one that directly states the answer; if the question asks what "
               f"the chat implies (a judgment, prediction, or reading of a "
               f"person), ground the answer in passages directly about the "
               f"subject and label it inferred; else abstain. If a search "
               f"returns no new passages, STOP re-searching and decide now."]
        for i, mid in enumerate(ids, 1):
            mem = self.store.memories.get(mid)
            if not mem:
                continue
            out.append(f"[{i}] {self._passage(mem)}")
            self._seen.add(mid)
        return "\n\n".join(out)

    def _sorted_by_time(self, ids) -> List[str]:
        return sorted(ids, key=lambda m: (self.store.memories[m].timestamp
                                          if m in self.store.memories else 0))

    # ---- tools ----
    def search_memory(self, query: str, k: int = 10, phase: str = "",
                      topic: str = "", author: str = "",
                      dense_query: str = "") -> str:
        # The passage count is a pipeline constant, not the agent's guess: once a
        # cross-encoder does the selecting, k stops being a recall dial. The `k`
        # argument survives in the tool schema -- removing it would change the
        # prompt the agent sees -- but it is ignored here, so an arm cannot
        # silently mix two counts depending on whether the model chose to pass one.
        #
        # SEARCH_TOPN + AUTO_HOP_K = 5 + 10 = 15 passages per call. The similarity
        # channel and the graph neighbourhood compete for one budget, and the hop's
        # value came from NARROWING the competition set rather than from any
        # proximity signal -- so cleaner anchors are worth more than more
        # similarity slots.
        k = SEARCH_TOPN
        scope = {}
        if phase:
            scope["phase"] = phase
        if topic:
            scope["topic"] = topic
        if author:
            scope["author"] = author
        idxs = self.retr.retrieve(query, k, scope=scope or None,
                                  dense_query=(dense_query or None))
        ids = [self.retr.id_by_index[i] for i in idxs
               if 0 <= i < len(self.retr.id_by_index)]
        if ids:
            # Automatic anchored second pass, merged into THIS return.
            #
            # SHRINKING THE COMPETITION SET is the operative ingredient: the global
            # ranking scores the wanted statement against the whole store, while
            # the anchored hop scores it against the union of the anchors' topic
            # buckets, a much smaller pool. Adding passages WITHOUT shrinking the
            # pool did nothing or hurt.
            #
            # Done here rather than by asking the agent, because the agent cannot
            # be reliably asked for it -- and the gain is ADDITIVE, so it must not
            # cost the agent a round.
            hopped = self._hop_rank(ids[:AUTO_HOP_ANCHORS], query,
                                    AUTO_HOP_K) or []
            seen = set(ids)
            ids = ids + [m for m in hopped if m not in seen]
            return self._fmt_scored(ids, cap=k + AUTO_HOP_K)
        return self._fmt_scored(ids, cap=k)

    def lookup_phrase(self, phrase: str, k: int = 10, phase: str = "",
                      topic: str = "") -> str:
        """Word-level lookup that bypasses the vector channel and the fusion.

        A single vector per passage cannot represent every local fact inside it: a
        passage literally containing a short phrase can still lose to passages that
        do not, because the rest of the text dominates the embedding, while BM25
        ranks it #1. Question types whose answer IS a short phrase (an ordering, a
        status word, a named exception) are unanswerable through the fused channel
        for that reason.

        It is a separate tool, NOT a weight change: raising BM25's RRF weight would
        shift ranking for every question, whereas here the model opts in per
        question and nothing else moves. Literal substring first (exactly what the
        caller asked for), BM25 as fallback (helps when the remembered wording is
        close but not identical).

        The tool ships undocumented -- the prompt names it ZERO times, so it is
        reached purely on the model's own judgement. WHERE IT ACTS is abstention:
        deciding that a specific wording is ABSENT needs a channel that can look
        for that wording, and a vector channel cannot answer "is this phrase here".
        It is not free (it costs the agent roughly half a round per question), and
        it is kept because the accuracy it buys is worth that.
        """
        k = max(1, min(int(k or 10), self.max_passages))
        phrase = (phrase or "").strip()
        if not phrase:
            return "lookup_phrase needs a phrase."

        def in_scope(mid):
            mem = self.store.memories.get(mid)
            if mem is None:
                return False
            if phase and (mem.phase_name or "") != phase:
                return False
            if topic and (mem.topic or "") != topic:
                return False
            return True

        needle = _squash(phrase).lower()
        hits = []
        for mid in self.store.order:
            if not in_scope(mid):
                continue
            # Same text the BM25 corpus and the dense index were built on, so a
            # hit here is guaranteed to be reachable content rather than some
            # other field the retriever never sees.
            txt = self.store.memories[mid].index_text(self.text_source)
            if needle and needle in _squash(txt).lower():
                hits.append(mid)
                # Stopping at k here picks the k OLDEST matches, not the k best:
                # store.order is ingestion order. Harmless for a long
                # discriminative phrase (few matches) but pathological for the
                # short status words these questions ask about, where it can
                # silently drop matching candidates. Ordering the pool by question
                # relevance instead was tried and changed no scores -- reaching the
                # gold passage is not what these questions are losing on -- so the
                # cheap rule stays.
                if len(hits) >= k:
                    break

        mode = "exact substring"
        if not hits and self.retr.lexical is not None:
            # No literal match: fall back to term overlap over the same corpus.
            mode = "BM25 term overlap (no exact match found)"
            for mid in self.retr.lexical.rank(phrase, k * 4):
                if in_scope(mid):
                    hits.append(mid)
                    if len(hits) >= k:
                        break
        if not hits:
            return (f"No passage contains {phrase!r}"
                    + (f" within phase={phase!r}" if phase else "")
                    + (f" topic={topic!r}" if topic else "")
                    + ". Try fewer or different words.")
        return (f"[{mode}] {len(hits)} passage(s) for {phrase!r}\n"
                + self._fmt_scored(hits, cap=k))

    def _decision_candidate(self, mid: str, exact: bool) -> str:
        """Expand only the stored chain-head value: no extra evidence, no inferred
        cross-chain replacement, no truncation of the body."""
        mem = self.store.memories[mid]
        head = mem
        verified = exact and bool(mem.supersedes) and not mem.decision_superseded_by
        if mem.node_kind == "fusion":
            # sources is write order; do not re-sort by time to guess the chain head.
            # A weak or missing source does not regain "current value" authority just
            # because the fusion row still exists in the store.
            sources = [self.store.memories.get(s) for s in mem.sources]
            verified = (verified and len(sources) >= 2
                        and len(set(mem.sources)) == len(mem.sources)
                        and all(s is not None and s.node_kind == "raw" for s in sources)
                        and all(s in mem.supersedes for s in mem.sources))
            if verified:
                verified = all(old.memory_id in new.supersedes
                               for old, new in zip(sources, sources[1:]))
            head = sources[-1] if sources else None
            verified = verified and head is not None and not head.decision_superseded_by
        lines = [f"[decision candidate={mid}; NOT a topic-wide replacement]"]
        if verified and head is not None:
            lines.append(f"[stored chain head source={head.memory_id}; "
                         "graph-linked, NOT semantically verified]")
            if head.decision_new_value:
                lines.append("[HEAD VALUE -- only this chain's decision field]\n"
                             + head.decision_new_value)
            else:
                lines.append("[No stored head-value quote; read the evidence below.]")
        else:
            lines.append("[UNVERIFIED CANDIDATE -- no confirmed chain head is "
                         "established here. Read the evidence; do not infer replacement.]")
        lines.append("[FULL EVIDENCE -- may contain both historical and current "
                     "statements; not every sentence is current]\n" + self._passage(mem))
        return "\n".join(lines)

    def _scoped_decisions(self, ordered: List[str], exact: bool) -> str:
        """Keep the original top-k; only make the coverage and the boundaries between
        chains explicit."""
        selected = ordered[:self.max_passages]
        header = (f"[decision candidates: showing {len(selected)} of {len(ordered)}, "
                  "NEWEST FIRST. This is a topic index, NOT one evolving decision. "
                  "A later candidate does not supersede another merely because it "
                  "shares a topic or has a later timestamp. Select evidence for the "
                  "SAME subject, decision field and requested scope. Different "
                  "fields can coexist; a later change does not revoke unmentioned "
                  "requirements. CURRENT_DECISION is chain-local, not proof that "
                  "this candidate answers the question. If none directly matches, "
                  "use search_memory(topic=..., query=...) or lookup_phrase.]")
        if not exact:
            header += ("\n[NO VERIFIED CHAIN HEADS from the topic index; fallback "
                       "decision points are candidates only, even when this list is short.]")
        if len(ordered) > len(selected):
            header += ("\n[PARTIAL TOPIC INDEX -- older candidates remain searchable; "
                       "absence from this return is not evidence of cancellation.]")
        return header + "\n\n" + "\n\n".join(
            f"[{i}] {self._decision_candidate(mid, exact)}"
            for i, mid in enumerate(selected, 1))

    def get_current_decision(self, topic: str) -> str:
        """The current head(s) of the decision chains under a topic.

        This FANS OUT, it does not walk: every chain head the topic carries is a
        candidate, and the newest `max_passages` are returned. Heads beyond the cap
        remain reachable through `search_memory`, so this is a ceiling on THIS
        TOOL, not on the system.
        """
        heads = getattr(self.retr, "_heads_by_topic", {}).get(topic, [])
        exact = bool(heads)
        if not heads:
            # fall back: current_version of any decision node in the topic
            cand = [self.store.current_version(m)
                    for m in self.store.by_topic.get(topic, [])
                    if self.store.memories[m].is_decision_point]
            heads = list(dict.fromkeys(cand))
        if not heads:
            return f"No current decision found for topic '{topic}'."
        # NEWEST FIRST: "current decision" must surface the latest heads; an
        # ascending sort + cap truncation would show the OLDEST ones.
        ordered = list(reversed(self._sorted_by_time(heads)))
        return self._scoped_decisions(ordered, exact)

    def resolve_date(self, msg_id: str, phrase: str, anchor: str = "msg") -> str:
        """Deterministically compute an absolute date from an anchor date plus a
        relative phrase. Offloads error-prone weekday/month arithmetic from the
        LLM. `anchor` selects the base date: "msg" = the message's timestamp
        (default), "phase" = the message's phase_start (for "N weeks after
        kickoff"-style phrasings measured from the phase start). The arithmetic
        lives in dates.resolve_relative_date (shared with the write side)."""
        if msg_id not in self.store.memories:
            return f"Unknown msg_id '{msg_id}'."
        mem = self.store.memories[msg_id]
        base = mem.phase_start if anchor == "phase" else mem.timestamp
        if not base or base.year <= 1:
            return f"Message '{msg_id}' has no usable {anchor} anchor date."
        anchor_dt = base
        aw = f"{anchor_dt.date().isoformat()}({anchor_dt.strftime('%a')})"
        target = resolve_relative_date(anchor_dt, phrase)
        if target is None:
            return (f"anchor={aw}; could not resolve relative phrase {phrase!r} "
                    f"deterministically — reason from the anchor date yourself.")
        tw = f"{target.date().isoformat()}({target.strftime('%a')})"
        return f"anchor={aw} + {phrase!r} -> {tw}"

    # ---- neighbourhood ranking (no longer a tool; used by search_memory) ----
    def _anchor_context(self, aid: str) -> set:
        """Structural neighborhood of one anchor across MULTIPLE edge types --
        for provenance / upward source-tracing. Gathers:
          - reply chain: ancestors (up) + descendants + supplements (thread_members)
          - sibling replies (same parent)
          - decision/version chain: forward to the current version (superseded_by
            hops) AND backward to everything this node's chain SUPERSEDES.
        Returns the set (anchor excluded)."""
        mem = self.store
        ctx = set(mem.thread_members(aid, depth=3))
        m = mem.memories[aid]
        if m.replies_to_parent and m.replies_to_parent in mem.memories:
            ctx.update(mem.memories[m.replies_to_parent].reply_children)
        # forward along superseded_by -> current version (trace what replaced it)
        cur, guard = aid, set()
        while cur in mem.memories and cur not in guard:
            guard.add(cur); ctx.add(cur)
            nxt = mem.memories[cur].superseded_by
            if not nxt:
                break
            cur = nxt
        # backward: everything this node (and its chain) SUPERSEDES (older sources)
        stack, seen = [aid], set()
        while stack:
            x = stack.pop()
            if x in seen or x not in mem.memories:
                continue
            seen.add(x)
            for old in mem.memories[x].supersedes:
                ctx.add(old)
                stack.append(old)
        ctx.discard(aid)
        return ctx

    def _hop_rank(self, anchors, query: str, k: int) -> Optional[List[str]]:
        """Neighborhood ranking: pool = union of the anchors' TOPIC buckets PLUS
        the anchors' multi-edge provenance neighborhood (reply chain + siblings +
        decision/version chain). The pool is then ranked by ordinary dense+bm25
        fusion. Returns ids best-first, or None when the anchors have no
        neighborhood.

        The provenance neighborhood earns its keep as a RECALL source, not as a
        ranking prior: dropping it from the pool loses passages that are reachable
        no other way, but adding it as a separately-weighted third RRF list on top
        of the pool does nothing. So it feeds the pool and stops there; there is no
        third list any more.

        This used to be reachable two ways -- an explicit `hop` tool and the
        automatic pass in search_memory. The tool is gone, the ranking stays, so
        there is still exactly one ranking rule, not two."""
        anchor_set = set(anchors)
        scope = set()
        prox = set()
        for aid in anchors:
            m = self.store.memories[aid]
            prox.update(self._anchor_context(aid))
            if m.topic:
                scope.update(self.store.by_topic.get(m.topic, []))
        pool = list((scope | prox) - anchor_set) or list(prox - anchor_set)
        if not pool:
            return None
        topn = max(k * 4, 40)
        try:
            qvec = self.retr.embedder.embed([query])[0]
            dense_list = self.retr._dense_rank(qvec, pool, topn)
            bm25_list = self.retr._bm25_rank_subset(query, pool, topn)
            fused = self.retr._rrf([dense_list, bm25_list])
            return sorted(fused, key=fused.get, reverse=True)[:k]
        except Exception:  # noqa: BLE001 — fall back to time order if embed fails
            return self._sorted_by_time([x for x in prox] or pool)[:k]

    # ---- dispatch ----
    def dispatch(self, name: str, args: Dict) -> str:
        try:
            if name == "search_memory":
                if not (args.get("query") or "").strip():
                    return "search_memory needs a non-empty query."
                return self.search_memory(
                    args.get("query", ""), args.get("k", 10),
                    args.get("phase", "") or "", args.get("topic", "") or "",
                    args.get("author", "") or "", args.get("dense_query", "") or "")
            if name == "lookup_phrase":
                if not (args.get("phrase") or "").strip():
                    return "lookup_phrase needs a phrase."
                return self.lookup_phrase(
                    args.get("phrase", ""), args.get("k", 10),
                    args.get("phase", "") or "", args.get("topic", "") or "")
            if name == "get_current_decision":
                if not (args.get("topic") or "").strip():
                    return "get_current_decision needs a topic."
                return self.get_current_decision(args.get("topic", ""))
            if name == "resolve_date":
                if not (args.get("msg_id") or "").strip() or not (args.get("phrase") or "").strip():
                    return "resolve_date needs both msg_id and phrase."
                return self.resolve_date(args.get("msg_id", ""), args.get("phrase", ""),
                                         args.get("anchor", "msg") or "msg")
            # A model that remembers a withdrawn tool name (get_thread, hop,
            # get_timeline, get_user_history, list_deadlines,
            # lookup_assignments) lands here. Refused rather than silently
            # ignored, so it cannot smuggle a tool back in.
            return (f"{name} is not available. Use the tools listed for you."
                    if name in _WITHDRAWN else f"Unknown tool '{name}'.")
        except Exception as e:  # noqa: BLE001
            return f"Tool '{name}' error: {type(e).__name__}: {e}"


_WITHDRAWN = {"get_thread", "get_timeline", "get_user_history", "hop",
              "list_deadlines", "lookup_assignments"}


def run_qa_agentloop(
    chat_fn: Callable[[List[Dict], Optional[List[Dict]]], Dict],
    system_prompt: str,
    question: str,
    asking_user: str,
    tools: ReadTools,
    n_max: int = 5,
    user_context: Optional[str] = None,
) -> Tuple[str, List[Dict]]:
    """Function-calling loop. `chat_fn(messages, tool_schemas) -> message dict`
    with keys {content, tool_calls?}. Returns (final_text, tool_trace).

    Stops when the model answers without a tool call, or after n_max rounds
    (then one forced final call without tools). N_max bounds cost/divergence.
    `user_context` (e.g. the asking user's profile passage) is prepended for
    EVERY question uniformly -- not gated on qtype.

    There is no `max_tokens` parameter. It used to be accepted here and was
    never read: `chat_fn` takes (messages, tools) only, so the cap it seemed to
    promise was always the one inside the client. That client now sends no cap
    at all -- a truncated turn ships half a sentence to the judge -- so the
    misleading parameter is gone rather than wired up.
    """
    asker_line = f"Asking user: {asking_user}\n\n" if asking_user else ""
    ctx_line = f"{user_context}\n\n" if user_context else ""
    tools.new_session()   # reset per-question 'already shown' dedup set
    user = (f"{asker_line}{ctx_line}Question:\n{question}\n\n"
            "Use the tools to gather grounded evidence, then answer.")
    messages: List[Dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    trace: List[Dict] = []
    schemas = tools.tool_schemas()

    def _final(msg: Dict) -> str:
        """Return the final text with the model's deliberation prepended when the
        last turn is a bare `Final: X`. `split_reasoning_and_final` then splits on
        the `Final:` line, so agent_reasoning gets populated without any change to
        the record schema. Under ENABLE_THINKING=1 the deliberation arrives in a
        separate field and would otherwise be discarded."""
        body = msg.get("content") or ""
        think = (msg.get("reasoning") or "").strip()
        if not body.strip():
            return think
        if not think:
            return body
        # Keep the provider-separated body and reasoning apart, so unmarked reasoning
        # is not sent to scoring.
        if not re.search(r"^\s*final(?: answer)?:", body, re.I | re.M):
            body = f"Final: {body}"
        return f"{think}\n{body}"

    for step in range(n_max):
        _t0 = time.perf_counter()
        msg = chat_fn(messages, schemas)
        llm_s = time.perf_counter() - _t0
        tcs = msg.get("tool_calls") or []
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if tcs:
            assistant["tool_calls"] = tcs
        messages.append(assistant)
        if not tcs:
            # marker entry: per-question finish_reason of the answer turn.
            # finish_reason=length here means the answer was cut mid-sentence
            # and accepted anyway -- analysis filters tool=='__final__' out of
            # tool chains and n_steps.
            trace.append({"step": step, "tool": "__final__",
                          "finish_reason": msg.get("finish_reason") or "",
                          "ret": "", "llm_s": round(llm_s, 3)})
            return _final(msg), trace
        # per-step deliberation: this is where the agent's own critical checks live
        # (e.g. concluding a handle is not a person). Recorded for attribution only.
        think = "\n".join(x for x in ((msg.get("reasoning") or "").strip(),
                                     (msg.get("content") or "").strip()) if x)
        for i, tc in enumerate(tcs):
            fn = (tc.get("function") or {}).get("name", "")
            raw_args = (tc.get("function") or {}).get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            _t1 = time.perf_counter()
            result = tools.dispatch(fn, args)
            tool_s = time.perf_counter() - _t1
            # the trace matches the tool result the model received, kept in full to
            # support offline attribution.
            entry = {"step": step, "tool": fn, "args": args,
                     "ret": result,
                     # latency attribution: llm_s is the round's model call (paid
                     # once per step, recorded on the first of any parallel
                     # calls), tool_s the retrieval/dispatch it triggered.
                     "llm_s": round(llm_s, 3) if i == 0 else 0.0,
                     "tool_s": round(tool_s, 3)}
            if think and i == 0:      # once per step, not once per parallel call
                entry["think"] = think
            trace.append(entry)
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "name": fn, "content": result})

    # tool budget exhausted -> force a final answer with no tools
    messages.append({"role": "user",
                     "content": "Tool limit reached. Give your Final answer now "
                                "using the evidence gathered above."})
    _t0 = time.perf_counter()
    msg = chat_fn(messages, None)
    trace.append({"step": n_max, "tool": "__final__",
                  "finish_reason": msg.get("finish_reason") or "", "ret": "",
                  "forced": True, "llm_s": round(time.perf_counter() - _t0, 3)})
    return _final(msg), trace
