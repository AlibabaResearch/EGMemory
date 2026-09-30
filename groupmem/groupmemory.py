"""``GroupMemory`` -- the whole system behind four calls.

    mem = GroupMemory.from_config({"name": "demo"})          # zero dependencies
    mem.add([{"author": "amy", "content": "we are moving to Postgres"}])
    print(mem.answer("what database are we using?"))
    print(mem.current("database choice"))
    for hit in mem.search("database"):
        print(hit["memory_id"], hit["text"][:60])

THE DEFAULT CONFIG NEEDS NOTHING INSTALLED and calls nothing over the network:
the hashing embedder, the in-memory index, a JSONL record log and the no-op
reranker. That default is a WORKING DEMONSTRATION, NOT THE REFERENCE CONFIGURATION
-- ``add`` and ``answer`` both need a real LLM, so they raise until one is
configured, and the reference benchmark results additionally require the DashScope
embedder, the cross-encoder and Postgres (``bench/`` produces them).

CONCURRENCY. ``add`` is serial within a channel because the decision registry
accumulates as it goes; the parallel write is ``bench/run_ingest.py``, which owns
the executor and one writer per channel. ``search`` / ``answer`` / ``current``
are safe concurrently once a read store exists (nothing on the read path mutates
it), but the first call builds it, so build it once with ``load()`` before
fanning out.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .configs.defaults import (AGENT_MODEL, EMBEDDING_MODEL, READ_MAX_PASSAGES,
                               WRITE_MODEL)
from .embeddings import get_embedder
from .memory.pipeline import ingest_channel
from .memory.reader import ReadTools, run_qa_agentloop
from .memory.retrieval import FusionRetriever
from .memory.versioning import assemble
from .memory.writer import MemoryWriter
from .prompts import QA_SYSTEM, WRITE_SYSTEM, load_prompt
from .reranker import get_reranker

# The label the reader groups by is ALWAYS the writer's own `phase` field. Copying a
# corpus's own label instead borrows the dataset builder's session structure, which
# no deployment has, so this is a constant here too rather than a facade option.
PHASE_SRC = "record"

DEFAULTS: Dict[str, Any] = {
    "name": "groupmem",
    # Providers, by the names their factories know. `llm` is None on purpose: there
    # is no model to guess, and a wrong one produces a complete, plausible,
    # differently-scoring result set with nothing in the logs to say why.
    "llm": None,                        # {"backend": ..., "model": ..., ...}
    "embedder": {"backend": "hashing"},
    "reranker": {"backend": "noop"},
    "record_store": {"backend": "jsonl"},
    "vector_store": {"backend": "inmemory"},
}

# The reference configuration: kept HERE, next to the default, so the difference
# between "it runs" and "it reproduces" is one dict rather than a paragraph someone
# has to find. Still needs DASHSCOPE_API_KEY, and Postgres for `record_store`.
RELEASE_CONFIG: Dict[str, Any] = {
    "llm": {"backend": "dashscope", "model": WRITE_MODEL},
    "embedder": {"backend": "dashscope_native", "model": EMBEDDING_MODEL},
    "reranker": {"backend": "dashscope"},
    "record_store": {"backend": "pg"},
    "vector_store": {"backend": "pgvector"},
}


class GroupMemory:
    def __init__(self, name: str = "groupmem", llm: Any = None,
                 embedder: Any = None, reranker: Any = None,
                 record_store: Any = None,
                 vector_store_backend: str = "inmemory",
                 write_prompt: Optional[str] = None,
                 qa_prompt: Optional[str] = None,
                 max_passages: int = READ_MAX_PASSAGES) -> None:
        """Built from ready-made providers. ``from_config`` is the friendlier door.

        ``llm`` may be None: a memory that can be read but not written is a real
        thing to want (an existing store, a different writer), and the error then
        arrives at ``add`` naming what is missing rather than at construction.
        """
        self.name = name
        self.llm = llm
        self.embedder = embedder or get_embedder("hashing")
        # ONE reranker instance for the whole facade. Its counters are the only
        # view into the cross-encoder failing open, and per-call instances would
        # each report zero.
        self.reranker = reranker or get_reranker("noop")
        self.records = record_store or _default_records(name)
        self.vector_store_backend = vector_store_backend
        self.write_prompt = (write_prompt if write_prompt is not None
                             else load_prompt(WRITE_SYSTEM))
        self.qa_prompt = (qa_prompt if qa_prompt is not None
                          else load_prompt(QA_SYSTEM))
        self.max_passages = max_passages

        # Write state. One writer per channel, kept because the registry
        # accumulates across calls: rebuilding it per `add` would either cost a
        # replay of the whole channel every time or, worse, start each batch with
        # an empty registry and re-register decisions that already have slots.
        self._writers: Dict[str, MemoryWriter] = {}
        self._lock = threading.Lock()
        # `ord` is the corpus position, i.e. read order. Seeded from what is
        # already committed so a restarted process continues the sequence instead
        # of interleaving new messages among the old ones.
        prog = self.records.progress()
        self._ord = itertools.count(sum(v + 1 for v in prog.values()))
        # Minted ids for messages that arrive without one. Per instance, so two
        # facades in one process cannot hand out the same id.
        self._minted = itertools.count(1)
        self._read: Optional[Dict[str, Any]] = None

    # ---- construction ------------------------------------------------------- #
    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None
                    ) -> "GroupMemory":
        """Build from a plain dict; anything omitted comes from ``DEFAULTS``.

        Provider sections are ``{"backend": name, ...}`` and the rest of the
        section is passed to that backend's constructor, so a provider's own
        options never need a mirror here.
        """
        cfg = dict(DEFAULTS)
        cfg.update(config or {})
        name = cfg["name"]

        llm = None
        if cfg.get("llm"):
            from .llms import get_llm
            spec = dict(cfg["llm"])
            backend = spec.pop("backend")
            model = spec.pop("model", None)
            if not model:
                raise ValueError(
                    "llm config needs a 'model'; there is no safe default -- "
                    f"the release used {WRITE_MODEL} for writing and "
                    f"{AGENT_MODEL} for answering")
            llm = get_llm(backend, model, **spec)

        espec = dict(cfg["embedder"])
        embedder = get_embedder(espec.pop("backend"), **espec)
        rspec = dict(cfg["reranker"])
        reranker = get_reranker(rspec.pop("backend"), **rspec)

        sspec = dict(cfg["record_store"])
        records = _make_record_store(sspec.pop("backend"), name,
                                     embedder=embedder, **sspec)

        vspec = dict(cfg["vector_store"])
        return cls(name=name, llm=llm, embedder=embedder, reranker=reranker,
                   record_store=records,
                   vector_store_backend=vspec.pop("backend"),
                   **{k: v for k, v in cfg.items()
                      if k in ("write_prompt", "qa_prompt", "max_passages")})

    # ---- write -------------------------------------------------------------- #
    def add(self, messages: Iterable[Mapping[str, Any]],
            channel: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ingest messages. Returns one committed record per message.

        Messages are dicts; only ``content`` is required::

            {"content": ..., "author": ..., "channel": ..., "timestamp": ...,
             "msg_id": ..., "reply_to": ...}

        ``channel`` overrides the per-message field for the whole batch. Messages
        are grouped by channel and each group is processed IN ORDER through that
        channel's writer, because a decision registry is channel-local and a
        supersedes pointer never crosses channels.

        The read store is invalidated, not patched: the reader's graph, its gate and
        its BM25 index are derived from the rows, and patching one node's edges by
        hand is how the two renderings drift apart.
        """
        if self.llm is None:
            raise RuntimeError(
                "add() needs an LLM: every write-side field (topic, rewritten "
                "text, the decision fields, the supersedes pointer) is produced "
                "by one tool call. Pass llm={'backend': ..., 'model': ...}")
        batch = [self._normalize(m, channel) for m in messages]
        out: List[Dict[str, Any]] = []
        by_ch: Dict[str, List[Dict[str, Any]]] = {}
        for m in batch:
            by_ch.setdefault(m["_channel"], []).append(m)
        with self._lock:
            for ch, msgs in by_ch.items():
                w = self._writer(ch)
                # The corpus position is assigned HERE, at arrival, not by a later
                # pass: it is a fact about the order the messages came in, and the
                # fused row's `ord` is derived from its chain root's.
                corpus_ord = {m["msg_node"]: next(self._ord) for m in msgs}
                with self.records.channel(ch) as handle:
                    out.extend(ingest_channel(msgs, w, handle, corpus_ord))
            self._read = None
        return out

    def _writer(self, channel: str) -> MemoryWriter:
        w = self._writers.get(channel)
        if w is not None:
            return w
        w = MemoryWriter(_chat_fn(self.llm), self.write_prompt,
                         self.embedder.embed,
                         chat_tools_fn=_write_tools_fn(self.llm),
                         phase_src=PHASE_SRC)
        # Replay this channel's committed rows through the registry's own
        # register/revise/restate. Zero LLM calls, and the only way the rebuilt
        # state is the state that was written.
        self.records.restore(w, channel)
        self._writers[channel] = w
        return w

    def _normalize(self, m: Mapping[str, Any],
                   channel: Optional[str]) -> Dict[str, Any]:
        """A user's message dict -> the writer's field names.

        ``msg_node`` is the id everything downstream keys on -- rows, supersedes
        pointers, ``reply_to`` -- so one is minted when the caller has none, from a
        counter rather than the content: two identical messages are two messages.
        """
        mid = m.get("msg_id") or m.get("msg_node")
        if not mid:
            mid = f"{self.name}-m{next(self._minted)}"
        return {"msg_node": str(mid),
                "_channel": str(channel or m.get("channel") or "main"),
                "reply_to": m.get("reply_to"),
                "author": m.get("author") or m.get("speaker") or "unknown",
                "role": m.get("role") or "user",
                # Text, not a datetime: every corpus converter passes text, the
                # column is text, and `resolve_date` parses it. A datetime here
                # would serialize differently per backend.
                "timestamp": str(m.get("timestamp")
                                 or time.strftime("%Y-%m-%d %H:%M:%S")),
                "content": m.get("content") or ""}

    # ---- read --------------------------------------------------------------- #
    def load(self, force: bool = False) -> Dict[str, Any]:
        """Build the read side from the committed rows. Cached; ``add`` clears it.

        This is the expensive call -- the gate walks every supersedes edge and BM25
        is rebuilt -- which is why it is one call per store rather than one per
        question, and why it is exposed: a server wants it warm before the first
        request.
        """
        if self._read is not None and not force:
            return self._read
        store, idx, lexical = assemble(*self._load_rows())
        retr = FusionRetriever(store, idx, self.embedder, lexical, self.reranker)
        tools = ReadTools(store, retr, max_passages=self.max_passages)
        self._read = {"store": store, "ids": idx, "lexical": lexical,
                      "retriever": retr, "tools": tools}
        return self._read

    def _load_rows(self):
        """(store, id_by_index) from whichever vector backend is configured."""
        if self.vector_store_backend == "inmemory":
            from .vector_stores.inmemory import load_store
            rows = self.records.export_rows()
            return load_store(rows, embedder=self.embedder)
        if self.vector_store_backend == "pgvector":
            from .vector_stores.pgvector import load_light_store
            return load_light_store(self.name,
                                    variant=getattr(self.records, "variant",
                                                    "release"))
        raise ValueError(
            f"unknown vector store {self.vector_store_backend!r}; "
            f"known: inmemory, pgvector")

    def search(self, query: str, k: int = 10, phase: str = "",
               topic: str = "", author: str = "",
               dense_query: str = "") -> List[Dict[str, Any]]:
        """Retrieved memories, best first, as dicts.

        The same two-channel fusion + cross-encoder selection the agent's own
        ``search_memory`` tool uses -- this returns the structure instead of the
        formatted passage, for callers that are not an LLM. ``text`` is the
        rewritten text, i.e. what both indexes were built over.
        """
        r = self.load()
        scope = {k2: v for k2, v in (("phase", phase), ("topic", topic),
                                     ("author", author)) if v}
        hits = r["retriever"].retrieve(query, k, scope or None,
                                       dense_query=dense_query or None)
        store, ids = r["store"], r["ids"]
        out = []
        for i in hits:
            mem = store.memories[ids[i]]
            out.append({
                "memory_id": mem.memory_id,
                # index_text("rewritten"), i.e. exactly the string the dense and
                # BM25 channels were built over -- returning the verbatim content
                # here would show text the retriever never matched against.
                "text": mem.index_text("rewritten"),
                "author": mem.author, "channel": mem.channel,
                "topic": mem.topic, "timestamp": mem.timestamp,
                "phase": mem.phase_name,
                "is_decision": mem.is_decision_point,
                "superseded_by": mem.superseded_by,
                "supersedes": list(mem.supersedes or []),
            })
        return out

    def answer(self, question: str, asking_user: str = "",
               n_max: int = 5, user_context: Optional[str] = None,
               with_trace: bool = False):
        """Run the read-side agent loop. Returns the answer text.

        ``with_trace=True`` returns ``(text, trace)`` instead; the trace is the
        tool calls the agent made, which is the only record of WHY an answer looks
        the way it does.

        ``n_max`` is the tool-loop bound. It is a cost and divergence bound, not a
        quality dial.
        """
        if self.llm is None:
            raise RuntimeError("answer() needs an LLM. Use search() / current() "
                               "for the retrieval side without one.")
        tools = self.load()["tools"]
        text, trace = run_qa_agentloop(_read_tools_fn(self.llm), self.qa_prompt,
                                       question, asking_user, tools,
                                       n_max=n_max, user_context=user_context)
        return (text, trace) if with_trace else text

    def current(self, topic: str) -> str:
        """The topic's current decision heads, newest first, as the agent sees them.

        A formatted string rather than a list because that IS the answer to "what is
        the current decision": which node is the chain head, and the annotation
        saying the order is reversed relative to ``search``. The structure is
        reachable via ``search(topic=...)``.
        """
        return self.load()["tools"].get_current_decision(topic)

    # ---- misc --------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        """Counts, plus the reranker's counters (its fail-open is invisible
        otherwise). Does NOT build the read side if it is not built."""
        out: Dict[str, Any] = {"name": self.name,
                               "channels": self.records.progress(),
                               "reranker": self.reranker.stats()}
        if self._read is not None:
            out["nodes"] = len(self._read["ids"])
            out["edges"] = len(self._read["store"].edges)
        return out

    def close(self) -> None:
        self.records.close()


# --------------------------------------------------------------------------- #
# provider glue
# --------------------------------------------------------------------------- #
def _default_records(name: str):
    from .storage.jsonl import JsonlRecordStore
    return JsonlRecordStore(name)


def _make_record_store(backend: str, name: str, embedder: Any = None, **kw: Any):
    """A record store by name. ``embedder`` is here for ONE reason: its width.

    The pgvector column is created as ``vector(dim)`` before anything is embedded,
    and Postgres rejects a wrong-length vector rather than truncating it, so the
    number cannot be deferred and cannot be guessed. It used to default to 1024 in
    every layer, which meant the default embedder (512-wide, hashing) plus the
    release record store created a table that failed on its first commit -- a
    combination nothing forbade and nothing checked.
    """
    if backend == "jsonl":
        from .storage.jsonl import JsonlRecordStore
        return JsonlRecordStore(name, **kw)
    if backend in ("pg", "postgres", "pgvector"):
        from .storage.pg import PgRecordStore
        if "dim" not in kw:
            dim = getattr(embedder, "dim", 0)
            if not dim:
                raise ValueError(
                    "record_store 'pg' needs the vector width up front: the "
                    "column is vector(dim). This embedder does not state a "
                    "`dim`, so pass one explicitly, e.g. "
                    "record_store={'backend': 'pg', 'dim': 1024}")
            kw["dim"] = dim
        return PgRecordStore(name, **kw)
    raise ValueError(f"unknown record store {backend!r}; known: jsonl, pg")


def _chat_fn(llm):
    """(system, user) -> text, at the WRITE stage's settings.

    The writer does NOT deliberate and gets the short timeout. That asymmetry is
    what the benchmark stores were written with, so it is stated at the call site
    rather than left to the provider's default -- which is the QA stage's, and would
    silently ingest under a writer that deliberates.
    """
    from .llms.base import HARD_TIMEOUT_WRITE, THINKING_WRITE

    def chat(sp: str, up: str) -> str:
        return llm.chat(sp, up, max_tokens=0, thinking=THINKING_WRITE,
                        hard_timeout=HARD_TIMEOUT_WRITE)
    return chat


def _write_tools_fn(llm):
    """(messages, schemas) -> message dict, at the WRITE stage's settings.

    Every write-side LLM turn is a tool call, so this -- not ``_chat_fn`` -- is the
    one the stage settings actually have to reach. It is a separate function from
    the read side's for exactly that reason: one shared helper would have to pick a
    stage, and either choice is wrong for the other caller.
    """
    from .llms.base import HARD_TIMEOUT_WRITE, THINKING_WRITE

    def call_tools(messages, schemas):
        return llm.call_tools(messages, schemas, max_tokens=0,
                             thinking=THINKING_WRITE,
                             hard_timeout=HARD_TIMEOUT_WRITE)
    return call_tools


def _read_tools_fn(llm):
    """(messages, schemas) -> message dict, at the QA stage's settings.

    Left to the provider's defaults, which ARE the QA stage's: deliberation on, the
    long timeout.
    """
    def call_tools(messages, schemas):
        return llm.call_tools(messages, schemas)
    return call_tools
