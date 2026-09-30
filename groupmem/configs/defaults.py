"""The only configurable surface of this release: model ids and top-k counts.

Every other number in the package is FROZEN at the value the default benchmark
runs were produced with. These are not, because they are the two things a reader
legitimately has to change: point the pipeline at a different model, or ask for
more/fewer passages per call.

Each still DEFAULTS to the reference value, and any override is ANNOUNCED on
stderr, so a wrong setting cannot go unnoticed. Two overrides carry a cost
beyond changing the resulting numbers:

  EMBEDDING_MODEL   vectors from two models are not comparable, so changing it
                    is only meaningful together with a full re-embed of BOTH
                    sides: the read table's pgvector column and the writer's own
                    rewritten vectors.
  RERANK_TOPN and   these two SPLIT one budget. The agent sees RERANK_TOPN
  READ_AUTO_HOP_K   similarity hits plus READ_AUTO_HOP_K graph-neighbourhood
                    hits (5 + 10 = 15 per search_memory call). Moving one at the
                    other's expense is a real trade, not a free widening.

There is deliberately NO output-token cap here: a clipped write-side reply parses
to `{}` and commits an empty plan, and a clipped QA turn ships half a sentence to
the judge. `llms/base.note_truncation` counts every `finish_reason=length` instead.
"""

from __future__ import annotations

import os
import sys

# The default benchmark results were produced with these.
_FINAL_MODELS = {
    "WRITE_MODEL": "qwen3.7-max",      # the one write-side stage
    "AGENT_MODEL": "qwen3.7-max",      # QA tool loop
    "JUDGE_MODEL": "qwen3.7-max",      # scorer; also pinned to temperature 0
    "EMBEDDING_MODEL": "text-embedding-v4",
    "RERANK_MODEL": "qwen3-rerank",    # cross-encoder in the selection stage
}

_FINAL_COUNTS = {
    "RERANK_TOPN": 5,          # passages the cross-encoder keeps per search
    "READ_AUTO_HOP_K": 10,     # passages the anchored hop appends to those
    "READ_MAX_PASSAGES": 12,   # cap on every OTHER read tool's return
}


def _announce(name: str, value, frozen, extra: str = "") -> None:
    print(f"[config] {name}={value} (default: {frozen}) -- results are no "
          f"longer comparable to the default benchmark numbers.{extra}",
          file=sys.stderr, flush=True)


def _model(name: str) -> str:
    frozen = _FINAL_MODELS[name]
    got = (os.environ.get(name) or "").strip()
    if not got or got == frozen:
        return frozen
    extra = ""
    if name == "EMBEDDING_MODEL":
        extra = (" Every stored vector has to be rebuilt with it (read table "
                 "AND the writer's rewritten vectors), or retrieval degrades "
                 "silently instead of failing.")
    _announce(name, got, frozen, extra)
    return got


def _count(name: str) -> int:
    frozen = _FINAL_COUNTS[name]
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return frozen
    try:
        got = int(raw)
    except ValueError:
        print(f"[config] {name}={raw!r} is not an integer; using {frozen}",
              file=sys.stderr, flush=True)
        return frozen
    if got <= 0:
        print(f"[config] {name}={got} must be positive; using {frozen}",
              file=sys.stderr, flush=True)
        return frozen
    if got != frozen:
        _announce(name, got, frozen)
    return got


WRITE_MODEL = _model("WRITE_MODEL")
AGENT_MODEL = _model("AGENT_MODEL")
JUDGE_MODEL = _model("JUDGE_MODEL")
EMBEDDING_MODEL = _model("EMBEDDING_MODEL")
RERANK_MODEL = _model("RERANK_MODEL")

RERANK_TOPN = _count("RERANK_TOPN")
READ_AUTO_HOP_K = _count("READ_AUTO_HOP_K")
READ_MAX_PASSAGES = _count("READ_MAX_PASSAGES")
