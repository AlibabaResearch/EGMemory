"""Write side of the ONE table: the per-message upsert, and resume from it.

The table itself (name, columns, DDL) belongs to ``pgvector``, which also reads
it. What lives here is what only the WRITER needs: turning a record into a row,
finding out how far each channel got, and rebuilding a writer's in-memory state
from the rows it already committed.

``ord`` is the corpus position, passed in at write time; ``ch_ord`` is the
position within the channel and is the resume anchor.

RESUME NEEDS NO SEPARATE STATE: there is no registry table and no reply table.
Replaying a channel's rows in ch_ord order through the real registry rebuilds the
DecisionRegistry (`slot == len(slots)` is the register branch, the verdict names
the other two) -- see `restore_channel`.

THE WRITER'S VECTOR AND THE READER'S ARE THE SAME COLUMN: both are
embed(rewritten), so there is no copy step between two columns where two runs
could disagree.

Transaction model: every write function takes a cursor, never opens its own. The
caller owns the transaction so that one message == one commit.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..vector_stores.pgvector import (GM_COLS, connect, ensure_table, fusion_row,  # noqa: F401
                       row_tuple, table, vec_literal)
# The record contract and the replay are backend-neutral and live in `base`; this
# module is the POSTGRES record store.
from .base import (REC_COLS, REC_KEY_COL, REC_KEYS, record_from_row,
                   replay_records)


# --------------------------------------------------------------------------- #
# writes (caller owns the transaction)
# --------------------------------------------------------------------------- #

def upsert_record(cur, domain: str, variant: str, rec: Dict[str, Any],
                  ch_ord: int, ord_: int,
                  rw_vec: Optional[Sequence[float]] = None) -> None:
    """Upsert one message's row. ON CONFLICT so a restart never duplicates it."""
    d = {REC_KEY_COL.get(k, k): rec.get(k) for k in REC_KEYS}
    # keep defaults TRUE, matching the writer: a reply that parsed to {} is a
    # message with no fields, not a message to drop.
    d["keep"] = rec.get("keep", True)
    d["node_kind"] = "raw"
    d["ch_ord"] = ch_ord
    d["ord"] = ord_
    t = table(domain, variant)
    allcols = GM_COLS + ["embedding"]
    ph = ", ".join(["%s"] * len(allcols))
    upd = ", ".join(f"{c}=EXCLUDED.{c}" for c in allcols if c != "memory_id")
    cur.execute(f"INSERT INTO {t} ({', '.join(allcols)}) VALUES ({ph}) "
                f"ON CONFLICT (memory_id) DO UPDATE SET {upd}",
                row_tuple(d, rw_vec))


def upsert_fusion(cur, domain: str, variant: str, mem, ord_: int,
                  vec: Optional[Sequence[float]] = None) -> None:
    """Upsert one chain's fused row, on the CALLER'S cursor.

    Same transaction as the message that extended the chain, which is the whole
    point: the fused row and the hop that made it current commit together or not
    at all, so an interrupted run can never leave a chain whose last hop exists
    and whose fused row still describes the state before it.

    DO UPDATE, not DO NOTHING: the id is the chain's ROOT (``fusion.fused_id``),
    so a growing chain arrives here repeatedly under the same id with one more
    hop each time. That is what makes the row an UPDATE rather than a pile.
    """
    d = fusion_row(mem, ord_)
    t = table(domain, variant)
    allcols = GM_COLS + ["embedding"]
    ph = ", ".join(["%s"] * len(allcols))
    upd = ", ".join(f"{c}=EXCLUDED.{c}" for c in allcols if c != "memory_id")
    cur.execute(f"INSERT INTO {t} ({', '.join(allcols)}) VALUES ({ph}) "
                f"ON CONFLICT (memory_id) DO UPDATE SET {upd}",
                row_tuple(d, vec))


# --------------------------------------------------------------------------- #
# reads (resume + export)
# --------------------------------------------------------------------------- #

def channel_progress(domain: str, variant: str,
                     dbname: Optional[str] = None,
                     user: Optional[str] = None) -> Dict[str, int]:
    """channel -> highest ch_ord already committed. The resume anchor; there is
    deliberately no checkpoint table or file, because this IS the checkpoint."""
    kw = {k: v for k, v in (("dbname", dbname), ("user", user)) if v}
    t = table(domain, variant)
    con = connect(**kw)
    cur = con.cursor()
    cur.execute("SELECT to_regclass(%s)", (t,))
    if cur.fetchone()[0] is None:
        con.close()
        return {}
    cur.execute(f"SELECT channel, max(ch_ord) FROM {t} "
                f"WHERE node_kind='raw' GROUP BY channel")
    out = {ch: n for ch, n in cur.fetchall()}
    con.close()
    return out


def load_channel_records(domain: str, variant: str, channel: str,
                         dbname: Optional[str] = None,
                         user: Optional[str] = None
                         ) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
    """[(record, vector)] for one channel in ch_ord order.

    Everything the writer needs to rebuild its recomputable per-channel state
    (`written`, `_cvec`, the registry) comes from here, so resume costs zero LLM
    calls.
    """
    kw = {k: v for k, v in (("dbname", dbname), ("user", user)) if v}
    t = table(domain, variant)
    con = connect(**kw)
    cur = con.cursor()
    cur.execute(f"SELECT {', '.join(REC_COLS)}, embedding FROM {t} "
                f"WHERE node_kind='raw' AND channel=%s ORDER BY ch_ord",
                (channel,))
    out = []
    for row in cur.fetchall():
        out.append((record_from_row(row[:-1]), _parse_vec(row[-1])))
    con.close()
    return out


def iter_records(domain: str, variant: str,
                 dbname: Optional[str] = None,
                 user: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    """Every message record for a domain+variant, corpus order within channels.

    Rebuilt FROM THE COLUMNS, which is why there is no `raw_json`: every record
    key is a column, so a verbatim copy of the record would only be a second
    place for the same facts.
    """
    kw = {k: v for k, v in (("dbname", dbname), ("user", user)) if v}
    t = table(domain, variant)
    con = connect(**kw)
    cur = con.cursor()
    cur.execute(f"SELECT {', '.join(REC_COLS)} FROM {t} "
                f"WHERE node_kind='raw' ORDER BY channel, ch_ord")
    for row in cur.fetchall():
        yield record_from_row(row)
    con.close()


def _parse_vec(v) -> Optional[List[float]]:
    """pgvector returns its column as the text literal '[a,b,...]'."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    s = str(v).strip().lstrip("[").rstrip("]")
    if not s:
        return None
    return [float(x) for x in s.split(",")]


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #

def restore_channel(writer, domain: str, variant: str, channel: str,
                    dbname: Optional[str] = None,
                    user: Optional[str] = None) -> int:
    """Rebuild a writer's per-channel state from the rows. Returns messages done.

    EVERYTHING is recomputed; nothing about the writer's state is persisted
    beyond the rows themselves. `written` and `_cvec` come from the rows
    directly, the registry comes from replaying them through its own
    register/revise/restate, and the supersedes chains come from the `supersedes`
    column the same way the reader's graph does.

    The replay is also what keeps the registry's BM25 state correct. Assigning
    `writer.registry.slots` a list of reconstructed Slot objects would bypass
    `DecisionRegistry._reindex` -- only reachable from register/revise -- so a
    resumed channel would come back with empty doc-frequency counters, which
    `lexical.ranked_ids` drops entirely, leaving `retrieve` fused on cosine alone
    for every restored slot for the rest of the run. Going through the real call
    sites cannot have that class of bug, because there is no second path.

    Zero LLM calls and no prompt hashing: the run continues from ch_ord+1.
    """
    return replay_records(writer,
                          load_channel_records(domain, variant, channel,
                                               dbname, user),
                          channel)
