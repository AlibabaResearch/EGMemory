"""The ONLY LLM stage: one agent looks up the registry, writes the fields, done.

Per message, exactly one agentic call:

  agent { search_registry(query) ... , write_fields(<all fields>) } -> _commit

and `_commit` at the bottom of this module is pure code from there on. One class,
`MemoryWriter`, because there is one writer; there is no second LLM stage.

The supersedes pointer is a registry rule, not a learned decision (see
`registry`), and the value it feeds is the fused row: `fusion.py` renders one
retrievable `[CURRENT DECISION] ... Decision evolution ...` row per chain, and
THE FUSED ROW IS WRITTEN HERE, in the same transaction as the message that
extended the chain (`_extend_chain` below, upserted by bench/run_ingest.py).
`bench/finalize.py` reconciles the same rows over a whole domain through the same
renderer.

The system prompt is the main write prompt VERBATIM, with a retrieval preamble
prepended and the "return one JSON object" line rewritten to point at the tool;
building it by concatenation keeps the two from drifting apart.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .fusion import build_fused, hop_from_record, mean_unit
from .registry import (DecisionRegistry, _fmt_registry, _fmt_thread,
                             ground_quote)

# Quote-grounding thresholds, named here instead of being repeated as literals at
# the call sites, so the set is visible in one place. What each one is for:
#   GROUND_VALUE   value_quote. Almost always an exact substring, so the fuzzy
#                  fallback only rescues punctuation drift.
#   GROUND_PRIOR   prior_ref. The one threshold that gates in practice: it
#                  catches paraphrased or cross-source "quotes" (the prompt's
#                  QUOTE DISCIPLINE forbids quoting [REGISTRY]/[THREAD]). A
#                  bogus prior_ref is rendered read-side as
#                  replaced_old_value_no_longer_current, and it now also decides
#                  the link tier (see `link_tier`).
GROUND_VALUE = 0.9        # value_quote / decision_quote (ground_quote default)
GROUND_PRIOR = 0.8        # prior_ref / new_quote

# NO SELF-CONTAINED REVISIONS. There is deliberately no path that records a
# change without a pointer when the message itself carries both the replaced
# value (oq) and the replacement (nq) but matched no registry slot. Such
# messages are same-breath corrections ("3pm", "today", "EHS owner") whose old
# value never held anywhere, and emitting a change payload for them tells the QA
# agent that a decision had once been in force when it never was. The slot
# requirement in `_commit` IS that rule: no slot, no pointer, written as `new`.


# Back-reference cues: a real revision refers to the OLD plan as a settled fact
# being walked back. GENERIC pragmatic markers only -- corpus-specific
# alternations are deliberately left out, so the trigger stays
# corpus-independent. Recall-only and ADDITIVE: a hit can only turn a
# restate/new that already matched a slot into a revision, so a low-precision cue
# is harmless here. It must NOT be used to grade an edge (see `link_tier`).
BACKREF = re.compile(
    r"no longer|the original|originally|earlier (decision|call|plan|agreement|lock)|"
    r"\brevisit\b|\breopen\b|\brevert\b|instead of|"
    r"we (previously|earlier)|walk(ing)? back|overturn", re.I)


def backref_hit(*texts) -> bool:
    return any(bool(BACKREF.search(t)) for t in texts if t)


def link_tier(oq: str) -> str:
    """The edge's link_conf tier: "high" iff the message NAMES what it replaces.

    `oq` is the grounded `prior_ref` -- the span of THIS message pointing at the
    value being replaced, already checked verbatim against the body at
    GROUND_PRIOR. Naming the replaced value is the only evidence strong enough to
    earn a high tier, and it is decided here in code, with no LLM call.

    IT MUST NOT FALL BACK TO BACKREF. BACKREF is the second-chance TRIGGER above:
    grading with it would hand the highest tier to exactly the weakest edges -- the
    ones the main call declined to call revisions at all -- which is the ranking
    inverted. What a tier is WORTH is decided by its consumer
    (versioning.HARD_EDGE_THRESHOLD); this function only has to be honest about the
    evidence.
    """
    return "high" if oq else "low"


def parse_plan_json(raw: str) -> Dict[str, Any]:
    """Best-effort JSON object out of an LLM reply.

    Tolerates a ```-fenced block and any prose around the object by taking the
    outermost brace pair. Returns {} rather than raising: callers treat a
    missing field as "the model said nothing about it", which is the same
    handling an unparseable reply needs.

    Both routes a reply can arrive by go through here -- the tool call's
    arguments and the bare-JSON salvage in `_run_field_agent` -- so a reply lands
    in the same shape either way.
    """
    if not raw:
        return {}
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
    lo, hi = s.find("{"), s.rfind("}")
    if lo == -1 or hi == -1 or hi <= lo:
        return {}
    try:
        obj = json.loads(s[lo:hi + 1])
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


# --------------------------------------------------------------- prompt material

# Stage-1 budgets, frozen. Neither cap actually binds: the loop stops because the
# agent is done, not because the cap fired.
MAX_LOOPS = 6            # stage-1 LLM turns, hard stop
MAX_SEARCHES = 3         # search_registry calls the agent may spend

# How many times a write_fields call may be rejected back to the model for
# failing the tool's own `required` list before the record is committed as it
# came. One: the point is to catch a dropped field, not to argue with the model.
MAX_REJECTS = 1

# The retrieval preamble. Deliberately terse: everything about WHAT to write
# already comes from the main write prompt, appended after this.
_PREAMBLE = """\
You write the memory record for ONE group-chat message, and you look up the
decision REGISTRY yourself before writing.

TOOLS
  search_registry(query)   query = a short SELF-CONTAINED query describing which
                           earlier memories matter for recording THIS message:
                           the decision/subject it discusses and, if it overrides
                           an earlier decision, that decision's subject and OLD
                           value. Write it so it stands on its own, as if the
                           reader has not seen the message: resolve pronouns and
                           "this/that/the above" into the thing they name, and
                           carry over the concrete identifiers the message uses
                           (systems, modules, files, fields, environments,
                           @handles, versions, values) -- those rare tokens are
                           what distinguish one decision thread from a
                           near-identical one. One line, no prose.
  write_fields(...)        the memory record. Its fields are specified below.

Search at least once before writing. If the rows that came back are unrelated to
this message, you may search once more from a different angle; at most %d
searches. The rows you retrieve are the ONLY registry state you get -- there is
no second pass, so the `matched_slot` and the topic you reuse can only come from
what your own queries surfaced.

The rest of this prompt specifies the fields. Read it as the definition of the
write_fields arguments.
""" % MAX_SEARCHES

# Field schema mirrors the write prompt's JSON object one-for-one, INCLUDING the
# per-field instructions.
#
# The descriptions belong in the schema, not only in the system prompt: under the
# tool protocol a long retrieval transcript sits between the prompt and the moment
# the model fills the argument, and instructions placed only up front get lost --
# `rewritten` comes back shorter with concrete particulars (dates, message ids)
# dropped. Putting them beside each argument keeps them in view where they are used.
_RW_DESC = (
    "Self-contained searchable restatement of THIS message: who said/decided "
    "what, with every concrete particular carried over. This text REPLACES the "
    "message for search and for downstream reading, so anything you drop is "
    "lost. Length follows the message: one clause for a one-line message, "
    "several sentences for a message that settles several things. Never "
    "summarise it down to its gist. CARRY OVER VERBATIM, every one of them: "
    "people and teams in the form the message uses (@handles stay @handles), "
    "plus who is assigned what; named systems, modules, files, tools, "
    "environments, identifiers, fields, links; every number, date, time, "
    "threshold, percentage, version and quantity, with its unit; each distinct "
    "item when several are listed -- do not collapse a list into a category "
    "word. MAKE IT STANDALONE: replace pronouns and 'this/that/the above' with "
    "what they refer to. Keep relative time expressions ('Friday', 'next "
    "week', 'EOD') in the words the message used -- do NOT convert them to "
    "calendar dates. Keep the reported stance intact: who is deciding versus "
    "proposing versus objecting, and any condition attached. Do not add facts "
    "the message does not state."
)

# The write_fields schema. THIS IS THE ONLY FIELD SET.
#
# The stored COLUMN names differ in places (`value_quote` lands in
# `decision_quote`, `prior_ref` becomes `link_conf` and never a stored value):
# that is the table's schema, not this one, and `_commit` maps between the two.
# Do not read the column names as a second field set.
_FIELDS = {
    "keep": {"type": "boolean",
             "description": "false ONLY for pure chit-chat/noise with zero "
                            "work content."},
    "topic": {"type": "string",
              "description": "2-5 word WORKSTREAM label: the ongoing line of "
                             "work this message belongs to, NOT a summary of "
                             "this message. REUSE, DO NOT MINT: if a registry "
                             "row you retrieved shows a `topic:` naming the same "
                             "line of work, copy it CHARACTER-FOR-CHARACTER, "
                             "even when this message covers a different facet, a "
                             "later stage, a revision or a narrower detail. Only "
                             "invent a label when no retrieved topic covers this "
                             "work at all. A topic spans MANY messages (dozens); "
                             "if a label could only ever apply to this one "
                             "message it is too specific. A message that CHANGES "
                             "a decision keeps the topic of the decision it "
                             "changes."},
    "phase": {"type": "string",
              "description": "2-4 word phase label for where this work CURRENTLY "
                             "stands. Phase tracks PROGRESS, not identity -- it "
                             "advances as work moves forward. DO NOT reuse an "
                             "earlier phase when work has moved on. Contrast with "
                             "`topic` which is the identity and stays the same."},
    # THE DECISION TEST LIVES HERE. It used to be a separate `is_decision`
    # boolean whose description carried it, with this field reading "null unless
    # is_decision" -- two fields stating one claim, and the boolean was read by
    # nobody: `_commit` derives the flag from this field being non-null.
    "verdict": {"type": ["string", "null"],
                "enum": ["new", "restate", "revision", None],
                "description": "NULL unless this message takes a POSITION on a "
                               "work question. Non-null for settling/locking a "
                               "decision, accepting/approving, committing to "
                               "act, recommending or endorsing an option, "
                               "stating what should or should not be done -- "
                               "even provisionally or with conditions. NULL for "
                               "pure questions, pure status reports, pure "
                               "information with no stance, and chit-chat. "
                               "WHICH verdict is about VALUES, not wording: "
                               "'restate' when this message leaves the operative "
                               "value unchanged, 'revision' only when it "
                               "replaces a value that was already in force, "
                               "'new' when no such value was in force yet. When "
                               "torn between restate and revision, choose "
                               "restate."},
    "value_quote": {"type": ["string", "null"],
                    "description": "Verbatim span of THIS message stating the "
                                   "value being fixed/decided. null when verdict "
                                   "is null. For a revision this is the NEW "
                                   "value (the replacement), not the old one."},
    "prior_ref": {"type": ["string", "null"],
                  "description": "Verbatim span of THIS message pointing at the "
                                 "value being REPLACED. Only for verdict=revision; "
                                 "null otherwise, and null when this message never "
                                 "mentions the old value explicitly."},
    "rewritten": {"type": "string", "description": _RW_DESC},
    "assignments": {"type": "array", "items": {"type": "object", "properties": {
        "assignee": {"type": "string"}, "item": {"type": "string"},
        "due": {"type": ["string", "null"]}}},
        "description": "Who is assigned what, and when it is due."},
}


# Fields exempt from the prompt-declaration check in MemoryWriter.__init__:
# the prompt words these as prose ("decide keep") rather than as a quoted JSON
# key, so requiring the quoted form would silently drop them.
_FIELDS_ALWAYS = {"keep", "verdict", "topic", "phase",
                  "rewritten", "value_quote", "prior_ref"}


# Nudge for _retry_empty. DELIBERATELY CONTENTLESS: it restates the calling
# contract and nothing else. Any hint about what to write ("remember to fill
# rewritten", "this looks like small talk") would make the retried records a
# differently-prompted population than the first-try ones, so they could no
# longer be pooled with the rest.
_EMPTY_NUDGE = ("Your previous reply was empty. Call write_fields now for the "
                "message in [CURRENT MESSAGE].")

# Rejection returned for a write_fields call that misses `rewritten`. Contentless
# for the same reason _EMPTY_NUDGE is: it names the field the schema already
# declares required and says nothing about what to put in it.
_REWRITE_REJECT = ("rejected: `rewritten` was missing or empty on a record you "
                   "kept. Call write_fields again with every required field.")


def _kept_without_rewrite(obj: Dict[str, Any]) -> bool:
    """True for a record the reader will render with no text to render.

    `required` in the write_fields schema is a HINT THE SERVER DOES NOT ENFORCE,
    and nothing on the client side looked either: a call that returned
    keep/verdict/topic/phase/decision_quote but omitted the `rewritten` key was
    committed with rewritten NULL, which then skipped embedding (`process` only
    embeds a non-empty rewrite), so the row was stored, filtered into the light
    store, and rendered with nothing to show. This is not truncation (`max_tokens`
    is unset) and not the network (chat calls raise on failure); it is the model
    simply leaving the key out, so the client has to catch it.

    `keep` mirrors `_commit`: absent means kept. A dropped record is exempt --
    the light store filters it out, so its missing text is never rendered.
    """
    if not obj or not bool(obj.get("keep", True)):
        return False
    rw = obj.get("rewritten")
    return not (rw.strip() if isinstance(rw, str) else "")


def _tools(fields=None):
    fields = fields or _FIELDS
    return [
        {"type": "function", "function": {
            "name": "search_registry",
            "description": "Look up decision slots this channel already holds.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "write_fields",
            "description":
                "Write this message's memory record. Ends the turn. Before you "
                "do: re-read [CURRENT MESSAGE] and [THREAD] above. The registry "
                "rows you just retrieved tell you WHICH decision this belongs "
                "to; the particulars you must write down come from the message "
                "and its reply chain, not from those rows. A short reply like "
                "'Checking with @analytics now.' carries almost nothing on its "
                "own -- name the date, the report, the message it answers, so "
                "the record stands alone. Writing 'earlier messages' or 'this "
                "issue' instead loses that information permanently, because this "
                "record REPLACES the message downstream.",
            "parameters": {"type": "object", "properties": fields,
                           "required": ["keep", "topic", "rewritten"]}}},
    ]


# ------------------------------------------------------------------- the writer

class MemoryWriter:
    """One per channel: the agentic stage-1 loop, then the code-only commit.

    Messages are fed in channel order by `bench/run_ingest.py`, one `process` call
    each; the registry accumulates across them, which is why a writer cannot be
    shared between channels and why a resumed run has to replay the registry
    (`write_store.restore_channel`) before feeding the next message.
    """

    def __init__(self, chat_fn: Callable[[str, str], str], system_prompt: str,
                 embed_fn: Callable[[List[str]], List[List[float]]],
                 chat_tools_fn: Callable[..., Dict[str, Any]],
                 phase_src: str = "record"):
        self.chat = chat_fn
        self.sys = system_prompt
        self.embed = embed_fn
        self.registry = DecisionRegistry(embed_fn)
        self.written: Dict[str, Dict[str, Any]] = {}   # mid -> committed record
        # SUPERSEDES CHAINS, keyed by their CURRENT head: the state `_extend_chain`
        # needs to hand the fused row its whole chain without reading the table
        # back. A chain only ever grows at the head, so one entry moves from the
        # old head to the new one per revision and the dict stays one entry per
        # live chain. Channel-local, like everything else on this writer, because
        # a supersedes pointer never crosses channels.
        self._chain: Dict[str, List[str]] = {}
        # The fused node for the chain the LAST processed message extended, if
        # any: (Memory, vector). The runner upserts it in that message's own
        # transaction, so a fused row is never more stale than its hops.
        self.last_fusion: Optional[Tuple[Any, List[float]]] = None

        # (messages, schemas) -> msg dict. REQUIRED, not defaulted: every LLM
        # turn the writer makes is a tool call, so a writer built without it
        # cannot process a single message -- a default would only move the
        # failure from construction to the first message.
        #
        # This is also the one place a reply cache would go. If one is added, two
        # constraints are load-bearing: the key must be namespaced per AGENT (a
        # second agent numbers its rounds from 0 too, so a shared tag collides on
        # (msg_node, kind, round)); and agent turns must be cached, because a turn
        # mutates the registry, the registry is rendered into every later prompt,
        # and the prompt hash is the cache key -- one uncached turn re-keys the
        # rest of the channel.
        self.chat_tools = chat_tools_fn

        # WHERE `phase_name` COMES FROM -- the only per-corpus decision the
        # pipeline still has, and it is made here because the writer stamps the
        # label the reader groups by (one table, so there is no later pass to
        # make it in).
        #   "record"  the LLM's own `phase` field. Required on the four synthetic
        #             domains and on EverMemBench: there "corpus" is B-class
        #             LEAKAGE, because the synthesizer allots phase_order in
        #             contiguous per-topic blocks, so copying it hands back the
        #             partition the writer is supposed to model.
        #   "corpus"  the message's own phase_name. Not leakage when the phase is
        #             a PUBLISHED, reader-visible division of the conversation
        #             rather than a generation artifact -- e.g. LoCoMo, which
        #             ships its dialogue as dated `session_N` blocks. There the
        #             LLM's phase field is mostly "n/a" anyway, which silently
        #             disables the by_phase bucket, `search_memory(phase=)`,
        #             `list_deadlines(phase=)` and `resolve_date(anchor="phase")`.
        # Either way the LLM's raw `phase` is ALSO stored, because the registry
        # replay on resume needs the label the slot was written with.
        if phase_src not in ("record", "corpus"):
            raise ValueError(f"phase_src must be 'record' or 'corpus', got "
                             f"{phase_src!r}")
        self.phase_src = phase_src

        self._cvec: Dict[str, List[float]] = {}   # mid -> REWRITTEN-text vector
        # Slot ranking reads those same rewritten-text vectors, keyed by the
        # slot's current holder. Nothing in the writer embeds a slot itself: a
        # per-slot vector would be a third representation, derived from neither
        # the rewritten text nor the read-side index, and it would cost one
        # embedding per register and per revise.
        self.registry.vec_of = lambda m: self._cvec.get(m)
        # NOTE: slot retrieval deliberately keeps the DecisionRegistry scorer
        # (a cosine + lexical-overlap blend). A higher-recall retriever (e.g. BM25)
        # would go straight to edge count with nothing to absorb it: no precision
        # gate stands downstream now -- a matched slot IS an edge.
        #
        # `triggered` counts messages that reached the revision rule at all;
        # `second_chance` the subset the main call did NOT call a revision (a
        # BACKREF cue on a matched slot did). committed + demoted == triggered.
        self.agent_stats = {"triggered": 0, "committed": 0, "demoted": 0,
                            "second_chance": 0}

        self._fields = _FIELDS
        # THE TOOL SCHEMA FOLLOWS THE PROMPT. A field the prompt does not declare
        # is not offered as a write_fields argument, so the two can never drift
        # into telling the model different things. This is what retires
        # `assignments`: none of its read-side consumers contributed anything
        # measurable, while a third of all records were paying to emit the array.
        # Retiring a field is therefore a prompt edit, and nothing here needs a
        # switch.
        #
        # Matched with quotes: the prompt's VERDICT RULES also say "assignments
        # move work forward", which must not count as declaring the field.
        self._fields = {k: v for k, v in self._fields.items()
                        if f'"{k}"' in self.sys or k in _FIELDS_ALWAYS}
        # The main prompt verbatim, retargeted from "emit JSON" to "call the
        # tool". `self.sys` is the write prompt as loaded by the runner.
        body = self.sys.replace(
            "OUTPUT \u2014 return ONLY one JSON object:",
            "FIELDS \u2014 these are the write_fields arguments:")
        self.stage1_sys = _PREAMBLE + "\n" + body
        # `fields_in_content` counts the salvage path in _run_field_agent: the
        # model returned the fields as bare JSON instead of calling write_fields.
        # Pre-declared rather than added on first use, because this is a plain
        # dict, not a Counter -- and kept as a COUNTER OF ITS OWN rather than
        # folded into no_fields, so the log distinguishes "recovered" from
        # "genuinely produced nothing".
        self.stage1_stats = {"messages": 0, "searches": 0, "turns": 0,
                             "no_search": 0, "budget_hit": 0, "no_fields": 0,
                             "fields_in_content": 0, "empty_reply": 0,
                             "empty_retry_ok": 0, "fusion": 0,
                             "fusion_novec": 0, "no_rewrite": 0,
                             "no_rewrite_final": 0}

    # ------------------------------------------------------------ stage 1
    def process(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        mid = msg.get("msg_node")
        content = msg.get("content") or ""
        chain = self._thread_chain(msg)

        obj, slots, nsearch = self._run_field_agent(msg, content, chain)
        if not nsearch:
            self.stage1_stats["no_search"] += 1

        # Embed the rewritten text HERE, before `_commit` runs, so that the vector
        # exists as early as it can.
        #
        # This position is safe for its one consumer: this message cannot yet be
        # any slot's holder (registry writes happen later, inside `_commit`), so
        # `registry.vec_of` cannot observe the difference. It cannot be moved
        # EARLIER: `rewritten` is what the field agent above produces, which is
        # also why stage-1's own retrieval has no rewritten vector available and
        # embeds its keyword query instead.
        rw = (obj.get("rewritten") or "").strip()
        if rw:
            try:
                self._cvec[mid] = self.embed([rw])[0]
            except Exception:      # noqa: BLE001
                pass

        rec = self._commit(msg, obj, content, slots)
        # The single place a committed record is published to later messages.
        self.written[mid] = rec
        # ... and the single place the chain it may have extended is re-fused.
        # AFTER `written`, because the render reads this message's own record out
        # of it like any other hop's.
        self.last_fusion = self._extend_chain(rec)
        self.stage1_stats["messages"] += 1
        return rec

    def _extend_chain(self, rec: Dict[str, Any], render: bool = True):
        """Track the chain this message extended; return its fused node, or None.

        Only a revision extends a chain, and it extends exactly one: `supersedes`
        names the node that held the value, which is by construction the chain's
        head (a revision points at the slot's CURRENT holder). So the new chain is
        the old one plus this message, and the entry moves to the new head --
        which reproduces, incrementally and without touching the table, the walk
        `fusion._chains` does over a loaded store.

        `render=False` is the resume path: replaying committed rows has to rebuild
        this dict, but re-rendering every historical chain on the way would be
        work whose result the table already holds.

        Returns None rather than raising when a hop has no vector. A fused row
        with no embedding would be invisible to the dense channel while looking
        present, and killing the channel over one un-rewritten message would be
        worse than leaving the row to `bench/finalize.py`, which reports the
        same condition as a hard stop because it can see the whole domain.
        """
        prev = rec.get("supersedes")
        if not prev:
            return None
        mid = rec["msg_node"]
        # A ROW THE READER DROPS IS NOT A HOP. `load_light_store` filters
        # `keep=false` rows out, and `store.link` then silently skips the edge
        # into the node that is no longer there -- so a chain running THROUGH a
        # noise message is two shorter chains on the read side. Ending the chain
        # at that boundary, rather than rendering across it, is what keeps this
        # path and `fusion._chains` enumerating the same chains on the same
        # corpus. Not a repair of anything observed: under this field set `keep`
        # is never false on the benchmark corpus -- it is here so the two renderers
        # cannot disagree if that ever stops being true.
        if not rec.get("keep", True):
            return None
        prevrec = self.written.get(prev)
        if prevrec is None or not prevrec.get("keep", True):
            return None
        chain = list(self._chain.pop(prev, [prev])) + [mid]
        self._chain[mid] = chain
        if not render:
            return None
        recs = [self.written.get(c) for c in chain]
        if any(r is None for r in recs):
            # An older hop belongs to a run this process did not replay, so the
            # chain cannot be rendered from here. finalize sees the whole table
            # and covers it.
            return None
        vecs = [self._cvec.get(c) for c in chain]
        if any(v is None for v in vecs):
            self.stage1_stats["fusion_novec"] += 1
            return None
        hops = [hop_from_record(
                    r, prior_value=(recs[i - 1].get("decision_quote")
                                    if i else None))
                for i, r in enumerate(recs)]
        self.stage1_stats["fusion"] += 1
        return build_fused(hops), mean_unit(vecs)

    def _run_field_agent(self, msg, content, chain):
        """Returns (fields, slots_seen, n_searches).

        The registry rows are collected as a union across queries -- so a second
        query exploring a different angle ADDS context instead of discarding what
        the first found -- and they are handed to `_commit`, which needs them to
        resolve `matched_slot`.
        """
        mid = msg.get("msg_node")
        # [THREAD] goes LAST. It is the source for resolving "this"/"the above"
        # into what they name, and by the time the model fills `rewritten` the
        # retrieval transcript already sits between the user turn and the answer.
        # Putting the chain closest to that boundary is the cheapest thing that
        # counteracts the distance: with the chain further up, particulars
        # available from the reply chain reach `rewritten` less often and short
        # records keep more unresolved references.
        user = ("[CURRENT MESSAGE]\n"
                f"msg_node: {mid} | author: {msg.get('author')} | "
                f"timestamp: {msg.get('timestamp')} | "
                f"reply_to: {msg.get('reply_to')}\n"
                f"text: {content}\n\n"
                "[THREAD] earlier stored messages on this reply chain -- use it "
                "to name what this message refers to\n" + _fmt_thread(chain))
        messages = [{"role": "system", "content": self.stage1_sys},
                    {"role": "user", "content": user}]

        seen: Dict[int, Any] = {}
        fields: Dict[str, Any] = {}
        nsearch = 0
        nreject = 0
        for _ in range(MAX_LOOPS):
            out = self.chat_tools(messages, _tools(self._fields))
            tcs = out.get("tool_calls")
            if not tcs:
                # The model ended its turn WITHOUT calling write_fields. It
                # usually did produce the fields -- as a bare JSON object in
                # `content` -- and dropping the turn threw that away, leaving a
                # record with no topic, no phase and no rewritten text.
                #
                # WHY IT HAPPENS. This stage-1 system prompt is the main write
                # prompt retargeted from "emit JSON" to "call the tool" (see the
                # module docstring), so the original "return ONLY one JSON
                # object" instruction is still in the text. On a message the
                # model judges to hold nothing worth storing it falls back to it.
                # It is a routing miss, not truncation and not a model failure:
                # the reply carries the fields, just not through the tool.
                #
                # WHY IT MATTERS MORE THAN THE COUNT SUGGESTS. `_commit` reads
                # keep as obj.get("keep", True), so an empty obj commits as
                # KEPT. On corpora where most messages are keep=false the
                # dropped-field rows therefore become nearly the whole kept set,
                # i.e. the writer's filter output is the bug rather than a
                # judgement.
                if not fields:
                    salvaged = parse_plan_json(out.get("content") or "")
                    if salvaged:
                        fields = salvaged
                        self.stage1_stats["fields_in_content"] += 1
                    else:
                        fields = self._retry_empty(messages)
                break
            messages.append({"role": "assistant",
                             "content": out.get("content") or "",
                             "tool_calls": tcs})
            done = False
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except Exception:      # noqa: BLE001
                    args = {}
                if name == "search_registry" and nsearch < MAX_SEARCHES:
                    q = (args.get("query") or "").strip()
                    got = self._retrieve_for(q, content)
                    nsearch += 1
                    self.stage1_stats["searches"] += 1
                    for s in got:
                        seen[s.slot_id] = s
                    ret = (_fmt_registry(got) if got else
                           "no matching slots (this is likely a NEW subject)")
                elif name == "search_registry":
                    ret = f"search budget spent ({MAX_SEARCHES}); write_fields now"
                elif name == "write_fields":
                    # `fields = args` FIRST, and kept even when the call is
                    # rejected: a model that refuses twice should still commit
                    # the topic and verdict it did produce rather than nothing.
                    # The rejection costs one extra turn out of MAX_LOOPS, which
                    # the search budget leaves free.
                    fields = args
                    if _kept_without_rewrite(args) and nreject < MAX_REJECTS:
                        nreject += 1
                        self.stage1_stats["no_rewrite"] += 1
                        ret = _REWRITE_REJECT
                    else:
                        ret = "ok"
                        done = True
                else:
                    ret = f"unknown tool {name}"
                messages.append({"role": "tool",
                                 "tool_call_id": tc.get("id"), "content": ret})
            self.stage1_stats["turns"] += 1
            if done:
                break
        else:
            self.stage1_stats["budget_hit"] += 1

        if not fields:
            # Agent ended without writing. Dropping the message would bias the
            # comparison (a variant cannot look good by silently skipping the
            # hard ones), so keep it with an empty record: `_commit` will store
            # the message with no decision, and the counter makes the rate
            # visible in the log instead of hiding it.
            self.stage1_stats["no_fields"] += 1
        elif _kept_without_rewrite(fields):
            # Committed without a rewrite anyway: the retry did not land, or the
            # fields arrived through the content-salvage path above, which is not
            # retried because that turn is already over. Counted so the residual
            # rate is in the log next to the rejections that were spent on it.
            self.stage1_stats["no_rewrite_final"] += 1
        return fields, [seen[k] for k in sorted(seen)], nsearch

    def _retrieve_for(self, query: str, content: str):
        """Registry retrieval driven by the AGENT's query.

        The query drives the DENSE side (it is what gets embedded). The SPARSE
        side scores the verbatim body, because BM25's IDF already discounts the
        filler a rewrite would drop, and keeping the body means the sparse
        channel does not depend on a per-run LLM artefact.

        Every slot in the channel competes; `retrieve` applies no relevance floor,
        so which slots are ELIGIBLE is not a tunable at all and the agent's choice
        among the returned k is the only precision stage.

        Verbatim text is still embedded nowhere: `content` is only tokenized.
        """
        qvec = None
        if query:
            try:
                qvec = self.embed([query])[0]
            except Exception:      # noqa: BLE001
                pass
        return self.registry.retrieve(content, qvec)

    def _retry_empty(self,
                     messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """One nudged retry after a turn that produced NOTHING usable.

        WHAT IT IS FOR. The model sometimes ends its turn with a completely
        empty reply -- `{"content": "", "finish_reason": "stop"}`, no
        tool_calls, no reasoning field -- after the registry lookup comes back
        "no matching slots". That is a SUCCESSFUL call that generated zero
        output tokens, so it is neither truncation (finish_reason would be
        "length") nor an API error (`_raw_ds_call_tools` raises on non-200), and
        `ds_call_tools`'s retry loop only fires on exceptions -- it returns an
        empty reply as-is. The message then committed with no fields at all.
        It is rare, but it silently produces an empty record.

        WHY IT IS HERE AND NOT IN ds_call_tools. That function is shared with
        the READ side, where the QA agent runs it once per tool round across
        every question; adding an empty-reply retry there would change the
        behaviour of the whole QA benchmark to fix a rare write-side case. Scoped
        to stage-1, the blast radius is exactly the thing that is broken.

        The nudge is appended as a USER turn and the empty assistant turn is NOT
        added to the history: a message with neither content nor tool_calls is
        not a valid assistant turn to send back. The nudge only re-states the
        contract -- it must not hint at what to write, or the retried records
        would be drawn from a different distribution than the first-try ones.
        """
        self.stage1_stats["empty_reply"] += 1
        msgs = messages + [{"role": "user", "content": _EMPTY_NUDGE}]
        out = self.chat_tools(msgs, _tools(self._fields))
        for tc in (out.get("tool_calls") or []):
            if tc["function"]["name"] != "write_fields":
                continue          # a second search_registry is not an answer
            try:
                got = json.loads(tc["function"].get("arguments") or "{}")
            except Exception:      # noqa: BLE001
                got = {}
            if got:
                self.stage1_stats["empty_retry_ok"] += 1
                return got
        got = parse_plan_json(out.get("content") or "")
        if got:
            self.stage1_stats["empty_retry_ok"] += 1
            self.stage1_stats["fields_in_content"] += 1
        return got

    # ------------------------------------------------------ context assembly
    def _thread_chain(self, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        chain, cur, seen = [], msg.get("reply_to"), set()
        # The cap matches `_fmt_thread`, which renders `chain[-4:]`. This walk
        # starts at the NEAREST ancestor and the list is reversed below, so those
        # four are exactly the four this walk collects first; walking further
        # only collects records that are then dropped.
        while cur and cur in self.written and cur not in seen and len(chain) < 4:
            seen.add(cur)
            rec = self.written[cur]
            chain.append(rec)
            cur = rec.get("reply_to")
        chain.reverse()
        return chain

    # --------------------------------------------------------- field mapping
    def _map_quotes(self, obj, content, retrieved):
        """Ground the quote fields and pick the registry slot.

        The ONLY place that knows the field set: `value_quote` (the value this
        message fixes -- the NEW value on a revision) plus `prior_ref` (the span
        pointing at the value being replaced).

        Returns (dq, oq, slot). There is ONE value span, not a before/after pair:
        a revision's `value_quote` IS its new value, and what it replaced is read
        off the superseded record rather than copied here. `oq` never becomes a
        stored value -- it is the revision TRIGGER (BACKREF) and the link TIER
        (`link_tier`), both of which run before any record exists.
        """
        dq = ground_quote(obj.get("value_quote"), content, GROUND_VALUE)
        oq = ground_quote(obj.get("prior_ref"), content, GROUND_PRIOR)
        # The slot comes from the writer's own retrieval, never from the model, so
        # there is no "did you really retrieve that slot" membership guard to
        # fail: it cannot be violated by construction.
        return dq, oq, (retrieved[0] if retrieved else None)

    def _apply_to_registry(self, verdict, slot, value, mid, subject, topic,
                           phase=""):
        """The single place that mutates the registry. Returns slot_used|None.

        `phase` is this message's progress label; the registry keeps it on the
        slot (following the current holder) and _fmt_registry shows it, so the
        next message sees the existing phase vocabulary and reuses it.

        CALL ORDER MATTERS FOR THE REVISION ARM: `registry.revise` moves
        `slot.holder` onto this message, so the superseded pointer has to be read
        off the slot BEFORE this method runs. `_commit` does exactly that and
        this method deliberately does not return it -- a second source for the
        same pointer is how the two would drift.
        """
        if verdict == "restate":
            if slot is None:
                return None
            self.registry.restate(slot, mid, phase=phase)
            return slot.slot_id
        if slot is not None and verdict == "revision":
            self.registry.revise(slot, value, mid, phase=phase)
            return slot.slot_id
        if value:                      # "new", or a revision with no slot
            s = self.registry.register(subject, value, mid, topic=topic,
                                       phase=phase)
            return s.slot_id
        return None

    # ---------------------------------------------------------------- commit
    def _commit(self, msg, obj, content, retrieved) -> Dict[str, Any]:
        """One main call's fields -> one write record. ZERO LLM from here on.

        A message whose verdict is `revision` and that matched a registry slot
        supersedes THAT SLOT'S CURRENT HOLDER -- the message which held the value
        until this one replaced it. The pointer is read off the registry BEFORE
        the update moves it, which is the whole rule.
        """
        mid = msg.get("msg_node")
        keep = bool(obj.get("keep", True))
        # `verdict` IS the decision state -- the field set states it once and the
        # record stores it once. There used to be an `is_decision` boolean beside
        # it, declared by the prompt, stored in a column and read by nobody: this
        # method already derived the flag from `verdict`, and so does the loader
        # (`verdict IS NOT NULL`). Gating the assignment on the model's own boolean
        # (`obj.get("verdict") if is_dec else None`) removed no failure mode and
        # added one -- `.get("is_decision", False)` turns ANY key or type anomaly
        # into a silent demotion of a decision (a misspelled key, a string where a
        # boolean was due), while the whitelist below validates `verdict` itself.
        # The prompt's definition of what counts as taking a position now lives on
        # this field, which is the field that has to carry it anyway.
        verdict = obj.get("verdict")
        if verdict not in ("new", "restate", "revision"):
            verdict = None

        dq, oq, slot = self._map_quotes(obj, content, retrieved)
        if verdict and dq is None:
            # There is no second chance to take: this field set carries one value
            # span, so there is no other quote to re-ground. A verdict with no
            # groundable value span is not a decision this store can hold.
            verdict = None

        supersedes = None
        slot_used = None
        link_conf = None
        if verdict:
            # THE REVISION RULE, in full. Two ways in:
            #   the main call said `revision`, or
            #   it said restate/new AND the text carries a BACKREF cue.
            # Both require a MATCHED SLOT, and requiring it is what makes the
            # rule sound rather than merely cheap: the slot IS the settled value
            # being overturned, and its holder IS the message that held it. The
            # cue alone fires often on messages with nothing to overturn.
            # NB neither arm needs a quote test: `verdict` is only still set here
            # if dq grounded (see the demotion above).
            triggered = verdict == "revision" or \
                        (verdict in ("restate", "new") and slot is not None
                         and backref_hit(content, oq))
            if triggered:
                self.agent_stats["triggered"] += 1
                if verdict != "revision":
                    # the second-chance path (main call demurred but the text
                    # walks back an earlier plan) -- counted separately, or the
                    # scorecards under-report it as 0
                    self.agent_stats["second_chance"] += 1
                if slot is None:
                    # A revision verdict with no slot: nothing settled to point
                    # at, so this is the opening statement of its own value, not
                    # a change. Recording the change off oq/nq alone is the NO
                    # SELF-CONTAINED REVISIONS note at the top of this module.
                    verdict = "new"
                    value = dq
                    self.agent_stats["demoted"] += 1
                else:
                    # READ THE POINTER BEFORE MUTATING. `_apply_to_registry`
                    # below calls registry.revise, which moves slot.holder onto
                    # THIS message; taken afterwards the edge would point at
                    # itself. `or None` because an empty holder is not a target.
                    supersedes = slot.holder or None
                    verdict = "revision"
                    value = dq
                    link_conf = link_tier(oq)
                    self.agent_stats["committed"] += 1
            else:
                value = dq
            # `verdict` is FINAL from here on: it is written once, not rewritten
            # on the way down with each stage doing its own registry write.
            if verdict == "restate" and slot is None:
                verdict = "new"
            # The slot label below is the first 60 chars of the decision quote --
            # the only source, since this field set carries no separate subject.
            # It reaches two consumers: the [REGISTRY] rows stage-1 reads, and the
            # slot's BM25 document (`subject + " " + quote`).
            #
            # It is never refreshed, so a revised slot's document keeps the
            # OPENING value as a prefix. That looks like a defect and is not: the
            # mechanism is token REACH, not ranking. For many revision messages
            # the stale prefix is their only lexical tie to the slot they revise,
            # because they cite the value being REPLACED -- drop it or refresh it
            # and the share of gold slots scoring exactly 0 in the lexical channel
            # jumps by an order of magnitude (a refreshed subject only repeats
            # tokens the quote already has, so it ties with having none at all).
            # Uncapping is worse still: it doubles the whole document, which
            # BM25's length normalization then penalizes.
            slot_used = self._apply_to_registry(
                verdict, slot, value, mid,
                (dq or "")[:60],
                (obj.get("topic") or "").strip(),
                (obj.get("phase") or "").strip())

        phase = obj.get("phase")
        return {
            "msg_node": mid,
            # The message's own structural fields ride along because the row this
            # record becomes IS the read-side node: there is no later pass that
            # rejoins the corpus, so anything the reader needs has to be here.
            "reply_to": msg.get("reply_to"),
            "author": msg.get("author"),
            "role": msg.get("role"),
            "timestamp": msg.get("timestamp"),
            "channel": msg.get("_channel"),
            "content": content,
            "keep": keep,
            "verdict": verdict,
            "slot": slot_used,
            "decision_quote": dq,
            # `supersedes` IS the whole change payload. Nothing here stores the
            # REPLACED value: `pg_store.load_light_store` follows this pointer to
            # the target's own decision_quote, so the fact lives once. A separate
            # column duplicating it would only risk drifting from the pointer, and
            # a NULL column is not free: it reads as a field the writer forgot.
            "supersedes": supersedes,
            "link_conf": link_conf,
            "topic": obj.get("topic"),
            "phase": phase,
            "phase_name": (msg.get("phase_name") if self.phase_src == "corpus"
                           else phase),
            "rewritten": obj.get("rewritten"),
        }
