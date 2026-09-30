"""What the write side needs of a record store, and nothing else.

ONE MESSAGE IS ONE COMMIT -- the entire transaction model. A backend that cannot
make ``commit_message`` atomic over the message row AND the fused row of the chain
it extended is not a valid backend: an interrupted run must never leave a chain
whose newest hop exists and whose fused row still describes the prior state.

CHANNELS ARE INDEPENDENT AND ARE WRITTEN CONCURRENTLY, one writer and one handle
each. Nothing in the write side crosses channels, so the store hands out a
per-channel handle; a backend whose handle is not usable from its own thread has
to say so.

RESUME IS A READ OF THE ROWS, NOT A CHECKPOINT. ``restore`` replays a channel's
committed records through the real registry call sites, so a backend's records
must come back in ``ch_ord`` order, with their vectors.

THE RECORD CONTRACT AND THE REPLAY LIVE HERE, not in ``records.py`` (which imports
``psycopg2`` through ``vector_stores.pgvector``), so a JSONL store does not need a
Postgres driver installed to write a text file. Neither is backend-specific: the
key list is what the WRITER emits, and the replay drives the registry's own methods.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


class ChannelHandle:
    """One channel's writer-facing handle. Use as a context manager."""

    def commit_message(self, rec: Dict[str, Any], ch_ord: int, ord_: int,
                       vec: Optional[Sequence[float]] = None,
                       fused: Optional[Tuple[Any, int, Optional[Sequence[float]]]] = None
                       ) -> None:
        """Persist one message row, plus ``fused`` = (Memory, ord, vec), ATOMICALLY.

        ``ord_`` is the corpus position and ``ch_ord`` the position within the
        channel; both are passed in rather than computed, so read order is a fact
        about the corpus and the resume anchor is a fact about the channel.
        """
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "ChannelHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RecordStore:
    """A domain+variant scoped record store. Backends: ``pg``, ``jsonl``."""

    def progress(self) -> Dict[str, int]:
        """channel -> highest ch_ord already committed. ``{}`` if nothing is.

        This IS the checkpoint. A store that cannot answer it cannot be resumed,
        and a run that cannot be resumed will be restarted from zero at full API
        cost the first time it is interrupted.
        """
        raise NotImplementedError

    def channel(self, channel: str) -> ChannelHandle:
        raise NotImplementedError

    def restore(self, writer, channel: str) -> int:
        """Rebuild ``writer``'s per-channel state from the rows; messages done."""
        raise NotImplementedError

    def iter_records(self) -> Iterable[Dict[str, Any]]:
        """Every message record (node_kind='raw'), channel then ch_ord order."""
        raise NotImplementedError

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# the record contract
# --------------------------------------------------------------------------- #
# The writer's record contract: every key it emits is a column. `msg_node` is the
# only name that differs from its column (`memory_id`), because a record is about
# a MESSAGE while a row can also be a synthesized node.
REC_KEYS = [
    "msg_node", "channel", "reply_to", "author", "role", "timestamp",
    "content", "rewritten", "keep", "verdict",
    "slot", "decision_quote", "topic", "phase", "phase_name", "supersedes",
    "link_conf",
]
REC_KEY_COL = {"msg_node": "memory_id"}
COL_REC_KEY = {v: k for k, v in REC_KEY_COL.items()}
REC_COLS = [REC_KEY_COL.get(k, k) for k in REC_KEYS]


def record_from_row(row: Any) -> Dict[str, Any]:
    """A row (mapping or ``REC_COLS``-ordered sequence) -> the writer's field names.

    One direction of the mapping, in one place, so a backend cannot invent a
    second spelling of ``memory_id`` -> ``msg_node``.
    """
    if hasattr(row, "get"):
        return {k: row.get(REC_KEY_COL.get(k, k)) for k in REC_KEYS}
    return {COL_REC_KEY.get(c, c): v for c, v in zip(REC_COLS, row)}


# --------------------------------------------------------------------------- #
# the replay
# --------------------------------------------------------------------------- #
def replay_records(writer, rows, channel: str) -> int:
    """Rebuild a writer's state from ``[(record, vector)]`` in ch_ord order.

    THE ONLY REPLAY. Every backend's ``restore`` funnels through this function
    rather than reimplementing it, because the registry must be rebuilt through its
    own register/revise/restate: assigning ``writer.registry.slots`` a list of
    reconstructed Slot objects would bypass ``DecisionRegistry._reindex`` -- only
    reachable from those methods -- so a resumed channel would come back with empty
    doc-frequency counters, which ``lexical.ranked_ids`` drops entirely, leaving the
    writer's own retrieval fused on cosine alone for every restored slot for the
    rest of the run. A second path is where that class of bug lives.

    Zero LLM calls and no prompt hashing: the run continues from ch_ord+1.
    """
    if not rows:
        return 0

    for rec, vec in rows:
        mid = rec["msg_node"]
        writer.written[mid] = rec
        if vec is not None and hasattr(writer, "_cvec"):
            writer._cvec[mid] = vec
        _replay_registry(writer, rec, channel)
        # Chain state, for the fused rows the next messages will extend. Tracked
        # without rendering: the rows these historical chains produced are already
        # committed, and re-deriving them would only re-write what is there.
        writer._extend_chain(rec, render=False)
    return len(rows)


def _replay_registry(writer, rec: Dict[str, Any], channel: str) -> None:
    """Apply one committed record to the registry, as `_commit` originally did.

    Mirrors `_apply_to_registry`'s three branches, and reads the branch off the
    record rather than guessing:

      slot == len(slots)  the register branch, by construction -- `register`
                          assigns `slot_id = len(self.slots)`, so a record whose
                          slot equals the current length is the one that created
                          it. Records that registered nothing carry slot=None.
      verdict == revision the revise branch, on an existing slot.
      verdict == restate  the restate branch: moves the holder, value unchanged.

    The value is `decision_quote` in all three, and the slot's subject label is
    its first 60 characters -- exactly what `_commit` passes, because this field
    set declares no subject field. Any other (slot, verdict) combination means
    the row was written by a call site this function does not know about, and it
    raises rather than silently resuming with a registry that has drifted.
    """
    sid = rec.get("slot")
    if sid is None:
        return
    reg = writer.registry
    dq = rec.get("decision_quote") or ""
    mid = rec["msg_node"]
    phase = (rec.get("phase") or "").strip()
    verdict = rec.get("verdict") or ""
    if sid == len(reg.slots):
        reg.register(dq[:60], dq, mid,
                     topic=(rec.get("topic") or "").strip(), phase=phase)
        return
    if sid < len(reg.slots):
        slot = reg.slots[sid]
        if verdict == "revision":
            reg.revise(slot, dq, mid, phase=phase)
            return
        if verdict == "restate":
            reg.restate(slot, mid, phase=phase)
            return
    raise RuntimeError(
        f"cannot replay {mid} in {channel}: slot={sid} verdict={verdict!r} "
        f"against {len(reg.slots)} rebuilt slots")
