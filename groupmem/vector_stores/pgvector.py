"""Postgres backend. ONE TABLE, ``gm_<domain>__<variant>``, for the whole system.

Design
------
Vectors live in Postgres (``vector`` column, pgvector). Dense retrieval runs
SERVER-SIDE via the ``<=>`` cosine-distance operator, so the 30k x 1024 matrix is
never materialized in Python -- this is the whole point of the DB backend: skip
loading/materializing embeddings, query them on demand.

The lightweight graph (scalar fields + reply/supersedes edges) is pulled once at
startup into an ordinary in-memory ``MemoryStore`` WITHOUT embeddings (~10 MB,
builds in seconds). That keeps ``FusionRetriever`` and every read tool working
UNCHANGED -- they only touch scalar fields and edges, never the raw vector.

So the split is:
  - embeddings      -> Postgres (pgvector), searched with ``ORDER BY emb <=> q``
  - graph/metadata  -> in-memory MemoryStore (embedding=None), same interface

It is the writer's table directly: edges are the ``reply_to`` / ``supersedes`` /
``sources`` columns, the reader's noise filter is a ``keep`` WHERE clause, and the
derived fields (is_noise, decision_type, link_confidence, phase timeline ...) are
recomputed in ``load_light_store`` rather than persisted. The ``FUS_*`` rows are
SYNTHESIZED (``fusion.apply_fusion``) rather than written by a message;
``bench/finalize.py`` appends them into this same table after ingest.

A ``PgVectorIndex`` is attached to the store as ``store.vindex``; the retriever's
``_dense_rank`` delegates to it when present.
"""

from __future__ import annotations

import json as _json
import os
import threading
import atexit
import weakref
from typing import Any, Dict, List, Optional, Sequence

import psycopg2
from psycopg2.extras import execute_values

from ..schema import Memory, MemoryStore, Relation, _parse_ts
from .base import (GM_COLS, VectorIndex, build_store)  # noqa: F401
from .base import fusion_row as _neutral_fusion_row

PG_DBNAME = os.environ.get("PG_DBNAME", "groupmem")
PG_USER = os.environ.get("PG_USER", os.environ.get("USER", "postgres"))

# THE column list, in a fixed order, shared by both sides. Every row -- written
# by the ingest or synthesized by finalize -- is built as a dict and turned into a
# tuple by `row_tuple`, so the order lives in exactly one place.
#
# Field groups: identity, corpus structure, the writer's own fields, the decision
# pointer. `timestamp` is TEXT ON PURPOSE: the corpus string is kept verbatim and
# parsed only on the way into a Memory, so a row always says what the corpus said.
#
# `message_type` is NOT here. The corpus ships it ("post" / "reply"), the row
# stored it, and nothing ever read it: `ingest.format_passage` does not render it,
# no retrieval filter tests it, no QA tool asks for it. It was moved from the
# corpus into the record into the column and back out of the column into a Memory
# field, four hops of pure transport. Tables written before it was dropped keep
# the column; every SELECT here names its columns, so an extra one is inert.
# GM_COLS is the backend-neutral row contract and lives in base.py; it is
# re-exported above because this module's DDL and SQL are built from it.

_DDL_TYPES = {
    "memory_id": "TEXT PRIMARY KEY",   # == msg_node for a message, FUS_<head> else
    "node_kind": "TEXT NOT NULL",      # "raw" | "fusion"
    "channel": "TEXT NOT NULL",
    "ch_ord": "INT",                   # position within the channel; the resume
                                       # anchor. NULL on a synthesized row.
    "ord": "INT",                      # read order; the corpus position, so it is
                                       # known at write time and never recomputed.
                                       # FUS_ rows are appended after the last one.
    "author": "TEXT",
    "role": "TEXT",
    "timestamp": "TEXT",
    "reply_to": "TEXT",
    "content": "TEXT",                 # verbatim body
    "rewritten": "TEXT",               # what the indexes are built on
    "keep": "BOOLEAN",                 # false = noise; the reader's WHERE clause
    "verdict": "TEXT",                 # new | revision | restate | NULL. Also THE
                                       # decision flag: there is no is_decision
                                       # column, because it was `verdict IS NOT
                                       # NULL` stored a second time (0 rows
                                       # disagreed across 29,940). load_light_store
                                       # derives is_decision_point from it.
    "slot": "INT",                     # registry slot; replayed on resume
    "decision_quote": "TEXT",
    "topic": "TEXT",
    "phase": "TEXT",                   # the LLM's own label, as written. Equal to
                                       # phase_name under --phase-src=record (the
                                       # default setting) and DIFFERENT under
                                       # 'corpus'; kept either way because the
                                       # resume replay feeds it back to the
                                       # registry so a restored slot carries the
                                       # label it was written with.
    "phase_name": "TEXT",              # the label the reader groups by
    "supersedes": "TEXT",              # the decision pointer; one per row
    "sources": "JSONB",                # fusion rows: the whole chain
    "link_conf": "TEXT",               # high | low; graded by agent_write.link_tier
}

# One lock id for all DDL on this table. DDL runs once per process at startup and
# takes milliseconds, so a single global lock is simpler than per-table keys and
# costs nothing.
_DDL_LOCK = 0x77524543          # 'wREC'


def connect(dbname: str = PG_DBNAME, user: str = PG_USER):
    con = psycopg2.connect(dbname=dbname, user=user)
    _register_conn(con)
    return con


# EXIT-SAFETY: psycopg2 connections left open are finalized during interpreter
# teardown in a nondeterministic order, which segfaults at the C (libpq) level
# -- observed on multi-worker QA runs (workers>=3) that finish their work and
# write results, then crash on the way out. We track every live connection in a
# WeakSet and close them ALL at atexit, BEFORE the interpreter starts tearing
# down modules, so libpq shuts each connection down cleanly while the runtime is
# still intact.
_OPEN_CONNS: "weakref.WeakSet" = weakref.WeakSet()
_CONN_LOCK = threading.Lock()


def _register_conn(con) -> None:
    with _CONN_LOCK:
        _OPEN_CONNS.add(con)


@atexit.register
def _close_all_conns() -> None:
    # NOT DEAD CODE. The decorator IS the only caller, so a reference-counting
    # dead-symbol sweep reports zero callers and wants to delete it. Deleting it
    # brings back the interpreter-teardown crash the WeakSet above exists to
    # prevent.
    with _CONN_LOCK:
        conns = list(_OPEN_CONNS)
    for c in conns:
        try:
            if not getattr(c, "closed", 1):
                c.close()
        except Exception:  # noqa: BLE001
            pass


def table(domain: str, variant: str) -> str:
    """``gm_<domain>__<variant>``. The variant is REQUIRED: it is what keeps a
    baseline from being overwritten in place when write-side behaviour changes,
    and this table is never dropped, so there is no other protection."""
    safe = "".join(c for c in domain if c.isalnum() or c == "_").lower()
    vsafe = "".join(c for c in variant if c.isalnum() or c == "_").lower()
    if not vsafe:
        raise ValueError("a variant is required; it names the run")
    return f"gm_{safe}__{vsafe}"


def vec_literal(emb: Optional[Sequence[float]]) -> Optional[str]:
    """pgvector text literal for an embedding, or None.

    6 decimals is the stored precision -- changing it silently invalidates
    every stored vector's comparability against the tables already written.
    """
    if not emb:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in emb) + "]"


def row_tuple(d: Dict[str, Any], emb: Optional[Sequence[float]] = None) -> tuple:
    """Column-order tuple for one row, embedding last.

    Both producers go through this -- the ingest's per-message upsert and
    finalize's fusion insert -- so neither can drift from GM_COLS.
    """
    return tuple(d.get(c) for c in GM_COLS) + (vec_literal(emb),)


# --------------------------------------------------------------------------- #
# DDL
# --------------------------------------------------------------------------- #

def ensure_table(domain: str, variant: str, dim: int = 1024,
                 dbname: Optional[str] = None,
                 user: Optional[str] = None) -> None:
    """CREATE IF NOT EXISTS the table and every index it needs.

    NON-DESTRUCTIVE, and there is deliberately no drop helper. This one table is
    now the writer's resume state as well as the read table, so a DROP would
    throw away hours of paid calls; and this package cannot tell whether the rows
    it would delete were written by itself or by another copy pointed at the same
    database. Rebuilding from scratch is what ``--variant`` is for.

    Serialized behind an advisory lock because ``CREATE TABLE IF NOT EXISTS`` is
    NOT atomic in PostgreSQL: two processes running it at the same moment both
    pass the existence check and then collide inserting the row type, raising
    `duplicate key value violates unique constraint "pg_type_typname_nsp_index"`.
    Observed for real -- two sharded ingests of the same domain started together
    and both crashed on startup. Sharded runs share this table by design, so
    concurrent callers are the normal case, not an edge case.

    The read indexes are created HERE, up front, rather than after a bulk load:
    the load is incremental (one commit per message), so there is no "after".
    """
    kw = {k: v for k, v in (("dbname", dbname), ("user", user)) if v}
    con = connect(**kw)
    con.autocommit = False           # advisory_xact_lock needs a transaction
    cur = con.cursor()
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (_DDL_LOCK,))
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    t = table(domain, variant)
    cols = ",\n            ".join(f"{c} {_DDL_TYPES[c]}" for c in GM_COLS)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
            {cols},
            embedding vector({dim})
        )""")
    # (channel, ch_ord) serves the resume scan; the other three serve the
    # read-side scope lookups. No ANN index on `embedding` on purpose: at 30k an
    # exact scan is ~ms, and exact keeps the results identical to the numpy
    # baseline instead of approximate.
    cur.execute(f"CREATE INDEX IF NOT EXISTS {t}_ch_ord ON {t} (channel, ch_ord)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {t}_topic ON {t} (topic)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {t}_phase ON {t} (phase_name)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {t}_author ON {t} (author)")
    con.commit()
    con.close()


def table_exists(domain: str, variant: str, dbname: str = PG_DBNAME,
                 user: str = PG_USER) -> bool:
    con = connect(dbname, user)
    cur = con.cursor()
    cur.execute("SELECT to_regclass(%s)", (table(domain, variant),))
    ok = cur.fetchone()[0] is not None
    con.close()
    return ok


# --------------------------------------------------------------------------- #
# fusion rows (finalize)
# --------------------------------------------------------------------------- #

def fusion_row(m: Memory, ord_: int) -> Dict[str, Any]:
    """``base.fusion_row`` with ``sources`` serialized for the JSONB column.

    The row itself is defined in base.py -- see there for what a fused row claims
    and why it carries no ``link_conf``. The ONLY difference here is the encoding:
    this column is JSONB, so the chain arrives as text and psycopg2 stores it as
    JSON; every non-SQL backend keeps the list.
    """
    d = _neutral_fusion_row(m, ord_)
    d["sources"] = _json.dumps(d["sources"], ensure_ascii=False)
    return d


def fetch_vectors(ids: Sequence[str], domain: str, variant: str,
                  dbname: str = PG_DBNAME,
                  user: str = PG_USER) -> Dict[str, List[float]]:
    """Vectors the write step already committed, read back by id.

    A fused row's vector is the mean of its chain's vectors, not an embedding of
    its own text, so the entire write path calls the embedding model in exactly
    ONE place: the per-message commit. finalize needs no API key.

    The cast to ``real[]`` is not cosmetic: psycopg2 has no adapter for the
    pgvector type and hands back the literal string ``'[0.1,...]'`` otherwise.
    """
    if not ids:
        return {}
    con = connect(dbname, user)
    cur = con.cursor()
    cur.execute(f"SELECT memory_id, embedding::real[] "
                f"FROM {table(domain, variant)} "
                f"WHERE memory_id = ANY(%s) AND embedding IS NOT NULL",
                (list(ids),))
    out = {r[0]: list(r[1]) for r in cur.fetchall()}
    con.close()
    return out


def fetch_ords(ids: Sequence[str], domain: str, variant: str,
               dbname: str = PG_DBNAME,
               user: str = PG_USER) -> Dict[str, int]:
    """``ord`` by id. A fused row's ord is a function of its chain ROOT's ord
    (``fusion.fused_ord``), so finalize needs the root's position to land on the
    same value the per-message writer would have written for that chain."""
    if not ids:
        return {}
    con = connect(dbname, user)
    cur = con.cursor()
    cur.execute(f"SELECT memory_id, ord FROM {table(domain, variant)} "
                f"WHERE memory_id = ANY(%s) AND ord IS NOT NULL", (list(ids),))
    out = {r[0]: int(r[1]) for r in cur.fetchall()}
    con.close()
    return out


def insert_rows(rows: List[tuple], domain: str, variant: str,
                dbname: str = PG_DBNAME, user: str = PG_USER,
                batch: int = 500) -> None:
    """Bulk-UPSERT prebuilt ``row_tuple``s.

    ON CONFLICT DO UPDATE, not DO NOTHING, because a fused row is not append-only
    state: its id is its chain's ROOT, so a chain that has since grown arrives
    here as the SAME id with more hops, and skipping it would leave the table
    holding a chain that is one hop short for ever. Overwriting every column is
    safe precisely because none of them is history -- ``ord`` is a function of
    the root, and the vector is the mean of the hops now on the chain.
    """
    if not rows:
        return
    t = table(domain, variant)
    con = connect(dbname, user)
    con.autocommit = True
    cur = con.cursor()
    allcols = GM_COLS + ["embedding"]
    collist = ", ".join(allcols)
    upd = ", ".join(f"{c}=EXCLUDED.{c}" for c in allcols if c != "memory_id")
    execute_values(cur, f"INSERT INTO {t} ({collist}) VALUES %s "
                        f"ON CONFLICT (memory_id) DO UPDATE SET {upd}",
                   rows, page_size=batch)
    cur.execute(f"ANALYZE {t}")
    con.close()


# --------------------------------------------------------------------------- #
# server-side vector search
# --------------------------------------------------------------------------- #

class PgVectorIndex(VectorIndex):
    """Thin wrapper around a domain table for server-side cosine search.

    Concurrency model: ONE dedicated daemon thread owns ONE connection for its
    whole lifetime; callers submit closures through a queue. Shared-conn+lock,
    thread-local and per-query connections ALL produced intermittent C-level
    segfaults under the QA agent's ThreadPoolExecutor on macOS -- even
    lock-guarded cross-thread use of a libpq handle touches its C state from
    several OS threads. Vector queries are millisecond-fast, so a single consumer
    is not a throughput concern (LLM calls dominate and stay fully parallel).
    A caller that times out marks its request abandoned so the worker skips it
    rather than draining dead work behind a live backlog -- without that, a
    transient slowdown cascades into a permanent 100% stall (see _db_worker).
    """

    _LOCK = threading.Lock()

    def __init__(self, domain: str, dbname: str = PG_DBNAME, user: str = PG_USER,
                 variant: str = "", keep_noise: bool = False):
        self.table = table(domain, variant)
        # The index must answer over EXACTLY the rows the store loaded, so it
        # takes the loader's own `keep` decision rather than defaulting to the
        # table. `keep` is a WHERE clause, not a table (module header), and
        # FusionRetriever._dense_rank passes candidate_ids=None on the
        # whole-store path precisely so the server can skip the id array -- so
        # without this predicate that path ranked the writer's DISCARDED rows
        # and handed their ids back. Those ids are absent from store.memories
        # and reranker.reorder_head indexes it unguarded, so the whole
        # search_memory call raised KeyError and the agent got an error string
        # instead of passages. LoCoMo is what exposed it because the writer drops
        # a large fraction of dialogue turns there, against almost none on the
        # four synthetic domains, so the discarded-row ids were reachable far more
        # often.
        self._rows = "TRUE" if keep_noise else "keep"
        self._dbname = dbname
        self._user = user
        self._local = threading.local()
        self._shared_con = None
        self._worker = None          # dedicated DB thread (owns the conn)
        self._q = None               # request queue for the worker

    def _get_con(self):
        """Shared autocommit connection; created once, reused under _LOCK."""
        c = self._shared_con
        if c is None or getattr(c, "closed", 1):
            c = connect(self._dbname, self._user)
            c.autocommit = True
            self._shared_con = c
        return c

    def _db_worker(self):
        while True:
            fn, box, ev = self._q.get()
            if fn is None:          # shutdown sentinel
                break
            # A caller that already timed out marks its request abandoned.
            # Running it would spend this single worker's throughput on a result
            # nobody awaits -- the positive feedback that turns a passing
            # slowdown into a permanent stall: dead work keeps every later
            # request queued past its own 60s wait, so the store locks at 100%
            # timeout and never recovers (observed as 45/45 questions, every
            # search, all-or-nothing). Skipping abandoned items keeps the worker
            # always serving a live request that can still finish in time, so
            # load sheds gracefully and self-heals instead of cascading.
            if box.get("abandoned"):
                continue
            try:
                box["val"] = fn()
            except Exception as e:  # noqa: BLE001
                box["err"] = e
            finally:
                ev.set()

    def _run_on_worker(self, fn):
        with self._LOCK:            # lazy-start the worker exactly once
            if self._worker is None or not self._worker.is_alive():
                import queue as _queue
                self._q = _queue.Queue()
                self._worker = threading.Thread(
                    target=self._db_worker, daemon=True,
                    name=f"pgvec-{self.table}")
                self._worker.start()
        box: Dict[str, object] = {}
        ev = threading.Event()
        self._q.put((fn, box, ev))
        if not ev.wait(timeout=60):
            box["abandoned"] = True   # worker skips it if still queued
            raise TimeoutError(f"pg worker stall on {self.table}")
        if "err" in box:
            raise box["err"]  # type: ignore[misc]
        return box["val"]

    @staticmethod
    def _vec(qvec: Sequence[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"

    def search(self, qvec, k: int, candidate_ids: Optional[Sequence[str]] = None) -> List[str]:
        """Top-k memory_ids by cosine distance (ascending = most similar).
        Optionally restrict to candidate_ids (structure-channel scope)."""
        q = self._vec(qvec)

        def _do():
            cur = self._get_con().cursor()
            if candidate_ids is not None:
                ids = list(candidate_ids)
                if not ids:
                    return []
                cur.execute(
                    f"SELECT memory_id FROM {self.table} "
                    f"WHERE {self._rows} AND memory_id = ANY(%s) "
                    f"AND embedding IS NOT NULL "
                    f"ORDER BY embedding <=> %s::vector LIMIT %s",
                    (ids, q, k))
            else:
                cur.execute(
                    f"SELECT memory_id FROM {self.table} "
                    f"WHERE {self._rows} AND embedding IS NOT NULL "
                    f"ORDER BY embedding <=> %s::vector LIMIT %s",
                    (q, k))
            return [r[0] for r in cur.fetchall()]
        return self._run_on_worker(_do)

    def close(self):
        # shut the worker down first so the connection is closed by the same
        # thread that used it (the whole point of the worker model)
        w, q = self._worker, self._q
        if w is not None and w.is_alive() and q is not None:
            def _close():
                c = self._shared_con
                if c is not None:
                    try:
                        c.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._shared_con = None
            box, ev = {}, threading.Event()
            q.put((_close, box, ev))
            ev.wait(timeout=10)
            q.put((None, None, threading.Event()))   # sentinel
            self._worker = None
            return
        c = self._shared_con
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
            self._shared_con = None


# --------------------------------------------------------------------------- #
# load light graph (no embeddings) + attach vector index
# --------------------------------------------------------------------------- #

def load_light_store(domain: str, dbname: str = PG_DBNAME, user: str = PG_USER,
                     variant: str = "", keep_noise: bool = False):
    """The pgvector backend's loader: rows -> ``base.build_store`` -> a store
    with a ``PgVectorIndex`` attached. Returns (store, id_by_index).

    ALL IT OWNS IS THE SELECT. Which columns are the node's fields, which of them
    are derived, and how the edges are built are ``base.build_store``'s -- they
    are the method and do not change with the backend. What is Postgres's here is
    that the rows arrive ``ORDER BY ord`` (``build_store`` requires that order for
    the last-write-wins superseded_by pass), that ``sources`` comes back already
    decoded because the column is JSONB, and that the vectors never leave the
    server: the graph carries ``embedding=None`` on every node and ranking goes
    through the index below.

    The table's columns are the version contract: a store written by an older
    writer fails on the SELECT rather than loading half-populated. Noise rows stay
    in the table (the writer needs them to resume) and are dropped by
    ``build_store`` unless ``keep_noise``.
    """
    if not variant:
        raise ValueError("load_light_store needs a variant; it names the run")
    t = table(domain, variant)
    con = connect(dbname, user)
    cur = con.cursor()
    cur.execute(f"SELECT {', '.join(GM_COLS)} FROM {t} ORDER BY ord")
    rows = [dict(zip(GM_COLS, row)) for row in cur.fetchall()]
    con.close()
    store, ids = build_store(rows, keep_noise=keep_noise)
    store.vindex = PgVectorIndex(domain, dbname, user, variant,
                                 keep_noise=keep_noise)  # type: ignore[attr-defined]
    return store, ids
