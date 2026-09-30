"""The four calls, against a real model, with no database.

    export DASHSCOPE_API_KEY=sk-...
    pip install -e '.[dashscope]'
    python3 examples/quickstart.py

The method, the real writer agent, the real embedder and the real read-side tool
loop -- with the rows in a JSONL log and the vectors in this process instead of
Postgres. ``vector_stores/base.build_store`` is shared, so a row means the same
graph here as in the table.

NOT the full benchmark configuration: the benchmark numbers additionally use
pgvector and the cross-encoder reranker and are scored by ``bench/``. The reranker
is left OFF here rather than quietly on -- see ``RELEASE_CONFIG`` in
``groupmem/groupmemory.py``.

For the same walk-through with no key and no network, see ``offline_smoke.py``.
"""

import os
import sys

from groupmem import GroupMemory
from groupmem.configs.defaults import AGENT_MODEL, EMBEDDING_MODEL

CHAT = [
    ("amy", "2026-01-05 09:00:00",
     "for the event log I say we go with Postgres, it's what ops already runs"),
    ("bob", "2026-01-05 09:02:00",
     "fine by me. standup stays at 10 by the way"),
    ("amy", "2026-01-05 09:04:00",
     "also let's cap the retention at 30 days to keep the table small"),
    ("carl", "2026-01-19 14:10:00",
     "heads up: we're switching the event log to ClickHouse instead of Postgres, "
     "the ingest rate outgrew it"),
    ("amy", "2026-01-19 14:12:00",
     "agreed, and with ClickHouse we can push retention out to 90 days"),
]


def main() -> int:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("DASHSCOPE_API_KEY is not set. This example calls a real model;\n"
              "run examples/offline_smoke.py for the no-key walk-through.",
              file=sys.stderr)
        return 2

    mem = GroupMemory.from_config({
        "name": "quickstart",
        # AGENT_MODEL is the read side's; the writer uses the same id by default.
        # Naming it from configs/defaults means this example cannot drift from the
        # default model.
        "llm": {"backend": "dashscope", "model": AGENT_MODEL},
        "embedder": {"backend": "dashscope_native", "model": EMBEDDING_MODEL},
        "reranker": {"backend": "noop"},        # see the module docstring
        "record_store": {"backend": "jsonl"},
        "vector_store": {"backend": "inmemory"},
    })

    print(f"ingesting {len(CHAT)} messages (one agent turn each) ...")
    recs = mem.add([{"author": a, "timestamp": t, "content": c}
                    for a, t, c in CHAT], channel="platform")
    for r in recs:
        print(f"  {r['msg_node']:>14}  keep={str(r.get('keep')):<5} "
              f"verdict={str(r.get('verdict')):<9} topic={r.get('topic')!r}")

    print("\n-- current decision on the event log --")
    print(mem.current("event log database"))

    print("-- search('retention') --")
    for h in mem.search("retention", k=3):
        flag = "SUPERSEDED" if h["superseded_by"] else "current   "
        print(f"  [{flag}] {h['memory_id']:>14}  {h['text'][:70]}")

    print("\n-- answer() --")
    for q in ("what database is the event log on?",
              "how long do we keep event log data?"):
        print(f"  Q: {q}\n  A: {mem.answer(q, asking_user='dana')}\n")

    print("stats:", mem.stats())
    print(f"rows are in {mem.records.dir}/ -- "
          f"{len(mem.records.export_rows())} of them; "
          f"a second run of this file resumes from them.")
    mem.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
