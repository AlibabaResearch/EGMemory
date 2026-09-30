"""JSONL record store: the same rows, one per line, no database.

The write side's only hard dependency on Postgres is where the rows go, and
``base.RecordStore`` needs only one atomic commit per message, a per-channel
handle, and a resume anchor readable back off the rows -- all of which an
append-only file satisfies. So this lets someone run the writer end to end with
nothing provisioned, and it is the interchange format:
``vector_stores.inmemory.read_jsonl_rows`` reads exactly what this writes.

THE LOG IS APPEND-ONLY AND THE LAST LINE FOR AN ID WINS: a fused row is rewritten
on every hop of its chain, and appending means a crash loses at most the line
being written. The reader collapses by ``memory_id`` keeping the last occurrence,
the same resolution Postgres reaches via ``ON CONFLICT DO UPDATE``.

Where it is weaker than Postgres:
  * ``commit_message`` writes two lines (message + fused), so a crash between them
    can leave the fused row one hop stale; it self-heals on the next hop, but it
    is why the benchmark runs use Postgres.
  * ``progress`` scans the whole file: fine at tens of thousands of lines, linear
    beyond that.
  * No concurrent writers to ONE channel's file. Channels get separate files.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..vector_stores.base import GM_COLS, fusion_row
# From `base`, NOT from `records`: that module is the Postgres store and imports
# psycopg2, which this backend exists precisely not to need.
from .base import (ChannelHandle, RecordStore, REC_KEYS, REC_KEY_COL,
                   replay_records)

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _fname(channel: str) -> str:
    """One file per channel. The channel id is arbitrary text, so it is slugged
    for the filesystem and the original is in every row's ``channel`` column."""
    return _SAFE.sub("_", channel or "_") + ".jsonl"


def _row_dict(rec: Dict[str, Any], ch_ord: int, ord_: int,
              vec: Optional[Sequence[float]]) -> Dict[str, Any]:
    """A message record -> the row dict, via the SAME mapping the SQL path uses.

    ``REC_KEYS`` and ``REC_KEY_COL`` are imported rather than restated, so a column
    added to the writer's contract reaches both backends or neither.
    """
    d = {REC_KEY_COL.get(k, k): rec.get(k) for k in REC_KEYS}
    # keep defaults TRUE, matching the writer and the SQL path: a reply that parsed
    # to {} is a message with no fields, not a message to drop.
    d["keep"] = rec.get("keep", True)
    d["node_kind"] = "raw"
    d["ch_ord"] = ch_ord
    d["ord"] = ord_
    out = {c: d.get(c) for c in GM_COLS}
    out["embedding"] = list(vec) if vec is not None else None
    return out


class JsonlChannel(ChannelHandle):
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def commit_message(self, rec: Dict[str, Any], ch_ord: int, ord_: int,
                       vec: Optional[Sequence[float]] = None,
                       fused: Optional[Tuple[Any, int, Optional[Sequence[float]]]] = None
                       ) -> None:
        lines = [_row_dict(rec, ch_ord, ord_, vec)]
        if fused is not None:
            fmem, ford, fvec = fused
            frow = dict(fusion_row(fmem, ford))
            frow["embedding"] = list(fvec) if fvec is not None else None
            lines.append(frow)
        for row in lines:
            # default=str for the one field whose type depends on the corpus
            # loader (timestamp arrives as text from every converter, but
            # a datetime must not silently fail the whole channel).
            self.f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        # flush + fsync per message: without them the "one message one commit"
        # guarantee is the OS page cache's, not this store's, and a resume after a
        # hard kill would replay messages that were reported as written.
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()


class JsonlRecordStore(RecordStore):
    """Rows under ``<root>/<domain>__<variant>/<channel>.jsonl``."""

    def __init__(self, domain: str, variant: str = "release",
                 root: str = "gm_records") -> None:
        self.domain, self.variant = domain, variant
        self.dir = os.path.join(root, f"{domain}__{variant}")

    # ---- reads -------------------------------------------------------------
    def _files(self) -> List[str]:
        if not os.path.isdir(self.dir):
            return []
        return [os.path.join(self.dir, f) for f in sorted(os.listdir(self.dir))
                if f.endswith(".jsonl")]

    def _rows(self, path: str) -> List[Dict[str, Any]]:
        """Collapsed by memory_id, last line wins. See the module docstring."""
        by_id: Dict[str, Dict[str, Any]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    by_id[row["memory_id"]] = row
        return list(by_id.values())

    def progress(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for p in self._files():
            for row in self._rows(p):
                if row.get("node_kind") != "raw":
                    continue
                ch = row.get("channel")
                n = int(row.get("ch_ord") or 0)
                if ch is not None and n > out.get(ch, -1):
                    out[ch] = n
        return out

    def channel(self, channel: str) -> JsonlChannel:
        return JsonlChannel(os.path.join(self.dir, _fname(channel)))

    def restore(self, writer, channel: str) -> int:
        path = os.path.join(self.dir, _fname(channel))
        if not os.path.exists(path):
            return 0
        raw = [r for r in self._rows(path) if r.get("node_kind") == "raw"
               and r.get("channel") == channel]
        raw.sort(key=lambda r: int(r.get("ch_ord") or 0))
        pairs = [(self._rec_of(r), r.get("embedding")) for r in raw]
        # Through `base.replay_records`, NOT a local copy of the replay: the
        # registry must only ever be rebuilt via register/revise/restate.
        return replay_records(writer, pairs, channel)

    @staticmethod
    def _rec_of(row: Dict[str, Any]) -> Dict[str, Any]:
        return {k: row.get(REC_KEY_COL.get(k, k)) for k in REC_KEYS}

    def iter_records(self) -> Iterable[Dict[str, Any]]:
        for p in self._files():
            raw = [r for r in self._rows(p) if r.get("node_kind") == "raw"]
            raw.sort(key=lambda r: int(r.get("ch_ord") or 0))
            for row in raw:
                yield self._rec_of(row)

    def export_rows(self) -> List[Dict[str, Any]]:
        """Every row, message and fused, for ``inmemory.load_store``.

        The read-side counterpart of the write: same collapse, same ``ord`` sort as
        ``read_jsonl_rows``, so a store written here loads without a database.
        """
        by_id: Dict[str, Dict[str, Any]] = {}
        for p in self._files():
            for row in self._rows(p):
                by_id[row["memory_id"]] = row
        return sorted(by_id.values(), key=lambda r: (r.get("ord") or 0))
