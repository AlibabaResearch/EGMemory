"""Postgres record store: the ``RecordStore`` the benchmark stores were written with.

A THIN WRAPPER, ON PURPOSE. The SQL, the record->row mapping and the resume replay
stay in ``records.py``, which is the module that has been run over four domains and
whose comments are the record of why resume needs no separate state. This file only
gives that module the shape ``base.RecordStore`` describes: a per-channel handle
that owns one connection and commits once per message.

``synchronous_commit = off`` is set on each channel's connection. Safe because the
run is repeatable -- losing the last few commits to a power cut means redoing those
messages -- and it is what makes one-commit-per-message affordable.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from . import records
from .base import ChannelHandle, RecordStore


class PgChannel(ChannelHandle):
    def __init__(self, domain: str, variant: str, channel: str,
                 dbname: Optional[str] = None, user: Optional[str] = None) -> None:
        self.domain, self.variant, self.channel = domain, variant, channel
        kw = {k: v for k, v in (("dbname", dbname), ("user", user)) if v}
        self.con = records.connect(**kw)
        self.con.autocommit = False
        self.cur = self.con.cursor()
        self.cur.execute("SET synchronous_commit = off")

    def commit_message(self, rec: Dict[str, Any], ch_ord: int, ord_: int,
                       vec: Optional[Sequence[float]] = None,
                       fused: Optional[Tuple[Any, int, Optional[Sequence[float]]]] = None
                       ) -> None:
        records.upsert_record(self.cur, self.domain, self.variant, rec,
                              ch_ord, ord_, vec)
        if fused is not None:
            fmem, ford, fvec = fused
            records.upsert_fusion(self.cur, self.domain, self.variant,
                                  fmem, ford, fvec)
        # The one commit. Both statements or neither.
        self.con.commit()

    def close(self) -> None:
        self.con.close()


class PgRecordStore(RecordStore):
    def __init__(self, domain: str, variant: str = "release",
                 dbname: Optional[str] = None, user: Optional[str] = None,
                 ensure: bool = True, dim: int = 1024) -> None:
        """The table for one domain+variant, created if it is not there.

        ``ensure`` DEFAULTS TO TRUE because this object IS the table, and until it
        did, the only code that created one was ``bench/run_ingest.py`` -- so a
        ``GroupMemory`` configured with ``record_store={"backend": "pg"}`` (the
        release arm) failed on a fresh database while the JSONL default worked.
        ``ensure_table`` is idempotent and serialized behind an advisory lock, so
        paying it per construction costs one round trip.

        Pass ``ensure=False`` for a reader with no CREATE privilege; ``dim`` must
        match the embedder, since the column is ``vector(dim)`` and Postgres
        rejects a vector of the wrong length rather than truncating it -- 1024 is
        ``text-embedding-v4``'s, i.e. what the default stores hold.
        """
        self.domain, self.variant = domain, variant
        self.dbname, self.user = dbname, user
        if ensure:
            records.ensure_table(domain, variant, dim=dim,
                                 dbname=dbname, user=user)

    def progress(self) -> Dict[str, int]:
        return records.channel_progress(self.domain, self.variant,
                                        self.dbname, self.user)

    def channel(self, channel: str) -> PgChannel:
        return PgChannel(self.domain, self.variant, channel,
                         self.dbname, self.user)

    def restore(self, writer, channel: str) -> int:
        return records.restore_channel(writer, self.domain, self.variant,
                                       channel, self.dbname, self.user)

    def iter_records(self) -> Iterable[Dict[str, Any]]:
        return records.iter_records(self.domain, self.variant,
                                    self.dbname, self.user)
