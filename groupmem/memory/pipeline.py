"""The write-side driver: feed one channel's messages through the writer, commit.

``MemoryWriter.process`` handles ONE message; everything that makes a run of 31k
messages survive is here -- resume, the moderation retry, and the atomic
message+fusion commit. It lives in the library rather than in ``bench/`` so that
anything, not only the harness, can ingest.

RESUME: the store's rows are the checkpoint (``RecordStore.progress``), the
writer's per-channel state is rebuilt by replaying those rows through the real
registry call sites, and the loop starts at ``ch_ord + 1`` -- zero LLM calls to
get back to where it was.

ONE WRITER AND ONE HANDLE PER CHANNEL. Channels share no state, so this function
is safe to run concurrently over different channels and does no locking; the
writer and handle are passed in already bound to the channel.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .fusion import fused_ord

# Substring of the provider error that means the request was refused by content
# moderation rather than failing. Matched on the message because that is all the
# endpoint gives; see ``ingest_channel`` for why this one error gets a stub row.
MODERATION_MARK = "inappropriate content"
MODERATION_RETRIES = 2
MODERATION_BACKOFF = 5.0


def ingest_channel(messages: Sequence[Mapping[str, Any]],
                   writer,
                   handle,
                   corpus_ord: Mapping[str, int],
                   start: int = 0,
                   on_progress: Optional[Callable[[int, int], None]] = None
                   ) -> List[Dict[str, Any]]:
    """Process ``messages[start:]`` in order, committing one row per message.

    ``corpus_ord`` maps msg_node -> corpus position, which is the read order and is
    a fact about the corpus rather than something a later pass computes.

    Returns the records, so a caller that wants them all in memory (the scorecard,
    an export) gets them without a second read of the store.
    """
    out: List[Dict[str, Any]] = []
    for i in range(start, len(messages)):
        m = messages[i]
        mid = m.get("msg_node")
        rec = _process_with_moderation_retry(writer, m, mid)
        out.append(rec)

        # One statement per message -- plus the fused row of the chain it extended,
        # if any, in the SAME transaction. The row IS the state, and it is the row
        # the reader loads. The registry used to be committed here too, next to a
        # row per LLM call; both are gone, and the store's ``restore`` replays the
        # registry from these rows instead.
        fused = writer.last_fusion
        fargs = None
        if fused is not None:
            fmem, fvec = fused
            # `ord` from the chain ROOT's corpus position, so the row lands in the
            # same place however many times the chain later grows, and after every
            # message row. bench/finalize.py computes it the same way from the
            # table.
            fargs = (fmem, fused_ord(corpus_ord.get(fmem.sources[0], 0)), fvec)
        handle.commit_message(rec, i, corpus_ord[mid],
                              writer._cvec.get(mid), fargs)
        if on_progress is not None:
            on_progress(i + 1, len(messages))
    return out


def _process_with_moderation_retry(writer, m: Mapping[str, Any],
                                   mid: Optional[str]) -> Dict[str, Any]:
    """``writer.process``, with the one provider error that must not kill a channel.

    A moderation refusal is not transient in the way a 429 is -- the same message
    will be refused again -- but it is also not a reason to abandon the remaining
    thousands of messages in the channel. So: retry a couple of times (the endpoint
    is not perfectly deterministic), then write a stub row and move on.
    """
    try:
        return writer.process(m)
    except RuntimeError as e:
        if MODERATION_MARK not in str(e):
            raise
    for _ in range(MODERATION_RETRIES):
        time.sleep(MODERATION_BACKOFF)
        try:
            return writer.process(m)
        except RuntimeError as e2:
            if MODERATION_MARK not in str(e2):
                raise

    # Carry the message's own structural fields: `channel` is NOT NULL, so a bare
    # stub trips the constraint and kills the channel, and the row IS the read-side
    # node, so anything missing here is missing from the store. keep=False keeps it
    # out of the loaded store and out of the scorecard; the content is preserved
    # verbatim so the skip is faithful rather than an empty hole.
    rec = {"msg_node": mid, "channel": m.get("_channel"),
           "reply_to": m.get("reply_to"),
           "author": m.get("author"),
           "role": m.get("role"),
           "timestamp": m.get("timestamp"),
           "phase_name": None,
           "content": m.get("content"),
           "keep": False, "verdict": None,
           "skip_reason": "moderation_400"}
    # `process` never returned, so `last_fusion` still holds whatever the PREVIOUS
    # message produced. Re-upserting it would be harmless (same id, same content)
    # but it would attribute another message's chain to this one.
    writer.last_fusion = None
    return rec
