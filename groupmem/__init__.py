"""Group-conversation memory -- pluggable layout.

ONE NAME IS EXPORTED, LAZILY::

    from groupmem import GroupMemory
    mem = GroupMemory.from_config({"name": "demo"})

Everything else is named by its module::

    from groupmem.memory.retrieval import FusionRetriever
    from groupmem.vector_stores import pgvector

Layout (write path -> read path). The directory names follow the provider/core
split so a backend can be swapped without touching the method:

  schema                  Memory / MemoryStore / Relation -- the data model
  configs/defaults        the ONLY configurable surface: model ids + passage counts

  memory/                 THE METHOD. Nothing here talks to a vendor API directly.
    passage               how a memory renders into a prompt passage
    registry              the writer's state: decision slots, quote grounding
    writer                THE writer: one agent per message, then the code-only commit
    versioning            the deterministic post-pass: hard-edge regate, QA loader
    fusion                FUS_* synthesis, applied by bench/finalize.py
    retrieval             cascade retrieval + graph expansion
    reader                the QA tool loop (four-tool surface)

  llms/                   chat + tool-calling clients (retries, timeouts)
  embeddings/             embedders + on-disk cache
  reranker/               cross-encoder reordering of the head
  vector_stores/          the store's schema and its loader
  storage/                the per-message record table + the resume replay
  utils/                  lexical (tokenizer, BM25, rank fusion), relative dates

  groupmemory             GroupMemory: the four calls, assembled correctly
  prompts/                the two prompts, bundled with the code

``bench/`` is the measurement harness, NOT part of the method. Its four entry
points run in order: ``run_ingest.py`` -> ``finalize.py`` -> ``run_qa.py`` ->
``summarize.py``. See ``README.md`` for the commands.

Environment:
  DASHSCOPE_API_KEY   required (chat, embedding and rerank endpoints)
  DASHSCOPE_BASE_URL  OpenAI-compatible endpoint override
  GMB_ROOT            GroupMemBench checkout (default: ./GroupMemBench)
  PG_DBNAME, PG_USER  Postgres connection
  configs/defaults.py names model ids and passage counts
"""

from typing import TYPE_CHECKING

__all__ = ["GroupMemory"]

if TYPE_CHECKING:                       # for type checkers and IDEs only
    from .groupmemory import GroupMemory


def __getattr__(name: str):
    """PEP 562 lazy attribute: import ``groupmemory`` on first use, not on import.

    ``del`` is NOT used to cache the result here because module ``__getattr__`` is
    only consulted when normal lookup fails, and the ``from ... import`` below
    binds the name into this module's globals -- so the second access never
    reaches this function.
    """
    if name == "GroupMemory":
        from .groupmemory import GroupMemory
        globals()["GroupMemory"] = GroupMemory
        return GroupMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
