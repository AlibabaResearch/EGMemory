"""How a memory renders into the passage the QA agent reads.

This is the only place the read side turns a stored node into text, so the tag
header below IS the contract between the two halves of the system: the writer
decides what a field means, and this decides whether the reader ever sees it.
The notes on the branches say why each tag is or is not rendered.

The corpus reader that used to live beside this function moved to
``bench/gmb_corpus.py``: reading GroupMemBench's JSON is the harness's job, and
the method should not import a benchmark to format a passage.
"""

from __future__ import annotations

from ..schema import Memory


VERSION_STATUS_RENDERER = "raw_revision_v1"


def format_passage(mem: Memory) -> str:
    """One retrieved node as the QA agent sees it: a tag header, then the body.

    The body is the REWRITTEN text (``index_text`` falls back to the unescaped
    verbatim body where no rewrite exists), which is also the text the dense and
    lexical indexes are built on -- so what ranked a passage is what the agent
    then reads. The verbatim content stays in the store and is never displayed.
    Fusion nodes additionally surface their kind, but no member ids -- see the
    note on the fusion branch below.
    """
    tags = [f"user={mem.author or '?'}"]
    if mem.role:
        tags.append(f"speaker_role={mem.role}")
    if mem.channel:
        tags.append(f"channel={mem.channel}")
    # PHASE TAGS ARE SHOWN. `phase_name` is the label the writer stamped; on the
    # synthetic domains it draws from a small set of distinct values, so it groups
    # rather than printing a near-unique string per passage. `phase_start` is not
    # stored -- it is derived at load as the earliest member of the
    # (channel, phase_name) group. Header tags are the only place injected volume
    # can be cut without touching a character of body text, but they stay on
    # because that is the default configuration, and dropping one is a QA run, not
    # an argument.
    if mem.phase_name:
        tags.append(f"phase_name={mem.phase_name}")
    if mem.phase_start and mem.phase_start.year > 1:
        tags.append(f"phase_start={mem.phase_start.date().isoformat()}")
    if mem.topic:
        tags.append(f"topic={mem.topic}")
    if mem.timestamp and mem.timestamp.year > 1:
        tags.append(f"timestamp={mem.timestamp.isoformat()}")
    if mem.reply_to:
        tags.append(f"reply_to={mem.reply_to}")
    if mem.msg_node:
        tags.append(f"msg_node={mem.msg_node}")
    # A fused node marks its kind, so the QA agent can treat it as the
    # consolidated current state.
    #
    # NO MACHINE-READABLE IDS ARE RENDERED. `sources=<member ids>` used to sit
    # here and the SUPERSEDED_BY tag used to carry its superseder's id; both are
    # gone. What the answers use is the chain SENTENCES and the FACT of a
    # supersession, not the ids.
    #
    # The tag WITHOUT an id is deliberate: it keeps the one bit that is used --
    # this statement was overridden, treat it as outdated -- while dropping the
    # identifier that was not.
    #
    # `sources` and `superseded_by` the FIELDS are untouched: the graph walks them
    # (schema.py resolve chain, retrieval.py head lookup, pgvector.py edge build)
    # and `get_current_decision` is built on them. Only the render changed.
    if getattr(mem, "node_kind", "raw") == "fusion":
        tags.append("kind=FUSED_DECISION")
    # version status derived from edges
    # superseded_by is still used to locate the fused retrieval unit; the status
    # trusts only the original revision relations. Genuinely old versions keep
    # SUPERSEDED_BY, and the tail hop of a fusion chain is no longer judged stale.
    if mem.decision_superseded_by:
        tags.append("status=SUPERSEDED_BY")
    elif mem.supersedes:
        tags.append("status=CURRENT_DECISION")
    # Decision-change payload: the prior value carries the knowledge-update
    # answer, and it MUST be labelled as replaced -- with a neutral
    # "prior_value=" tag next to status=CURRENT_DECISION, QA agents were observed
    # answering with the value that had already been superseded.
    if mem.decision_prior_value:
        tags.append("replaced_old_value_no_longer_current="
                    f"{mem.decision_prior_value}")
    # ASSIGNMENTS ARE NOT RENDERED, and the write side does not offer the field
    # (see MemoryWriter.__init__, which drops any field the prompt does not
    # declare). Not rendering wins for two reasons: a field shown in a passage is
    # read as authority, so a wrong row costs more than a missing one; and shorter,
    # less cluttered passages help abstention on their own.
    return f"[{' / '.join(tags)}]\n{mem.index_text('rewritten')}".strip()
