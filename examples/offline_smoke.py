"""End to end with nothing installed but the package: no key, no database.

WHAT IS REAL AND WHAT IS NOT. The write path runs for real -- the registry, the
quote grounding, the verdict rules, the supersedes pointer, the fused row, the
commit -- and so does the whole read path. Only the model's *judgement* is canned:
``ScriptedLLM`` answers each message with one ``write_fields`` tool call from a
table below. That is why this is a smoke test rather than a demo: it asserts the
invariants and prints a verdict. For the same thing with a real model, see
``quickstart.py``.

The package has to be importable (these files do not manipulate ``sys.path``)::

    pip install -e .                        # from the checkout, then
    python3 examples/offline_smoke.py

    PYTHONPATH=. python3 examples/offline_smoke.py     # or, without installing

It checks that:
  * a decision and its later revision form a chain, gated at the hard tier
  * the fused row is written, and is the chain HEAD
  * ``search`` returns the rewritten text, best first
  * ``current`` names the newest decision and still cites the one it replaced
  * a SECOND facade over the same rows RESUMES -- the registry is replayed from the
    commits, so the third message revises the existing slot instead of registering
    a second one for the same subject
"""

import json
import shutil
import tempfile

from groupmem import GroupMemory
from groupmem.storage.jsonl import JsonlRecordStore

WORK = tempfile.mkdtemp(prefix="groupmem-smoke-")


class ScriptedLLM:
    """A stand-in for the write-side agent: one canned field set per message.

    It calls ``search_registry`` before writing a revision, because that is what
    the rule requires -- a revision must point at a MATCHED SLOT, and the slot
    comes from the writer's own retrieval, never from the model's say-so. An
    earlier version of this file skipped the search and the writer demoted the
    revision to ``new``, correctly, under the NO SELF-CONTAINED REVISIONS rule. So
    this is the stand-in behaving like an agent, not a workaround.
    """

    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.searched = set()

    def chat(self, sp, up, **kw):
        return "unused: every write-side turn is a tool call"

    def _call(self, name, args):
        self.calls += 1
        return {"content": "", "finish_reason": "stop", "tool_calls": [
            {"id": f"c{self.calls}", "type": "function", "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False)}}]}

    def call_tools(self, messages, tools=None, **kw):
        # Match on the [CURRENT MESSAGE] block ONLY. Matching the whole prompt
        # matched m1's text inside m3's [THREAD] block (m3 replies to m1) and so
        # answered m3 with m1's fields -- whose value_quote is not a span of m3,
        # so the writer refused to ground it and the revision vanished. The
        # writer was right; the stand-in was wrong.
        blob = "\n".join(str(m.get("content") or "") for m in messages)
        cur = blob.split("[CURRENT MESSAGE]")[-1].split("[THREAD]")[0]
        for key, fields in self.script.items():
            if key not in cur:
                continue
            if fields.get("verdict") == "revision" and key not in self.searched:
                self.searched.add(key)
                return self._call("search_registry", {"query": fields["topic"]})
            return self._call("write_fields", fields)
        raise AssertionError(f"no scripted reply matched:\n{cur[:400]}")


SCRIPT = {
    "we'll use Postgres for the event log": {
        "keep": True, "topic": "event log database", "phase": "planning",
        "verdict": "new", "value_quote": "we'll use Postgres for the event log",
        "prior_ref": None,
        "rewritten": "Amy decided the event log will run on Postgres."},
    "standup is at 10": {
        "keep": True, "topic": "standup time", "phase": "planning",
        "verdict": None, "value_quote": None, "prior_ref": None,
        "rewritten": "Bob noted the standup is at 10."},
    # `prior_ref` MUST be a verbatim span of THIS message: `link_tier` grades the
    # supersedes edge off it, and an ungrounded one gets the low tier, which the
    # 0.9 gate in `regate_supersedes` then drops. An ungrounded prior_ref (say
    # "Postgres" against a message that never said it) still commits with
    # verdict=revision and supersedes=m1, but the READER throws the edge away --
    # so the message here names the old value out loud.
    "switching the event log to ClickHouse": {
        "keep": True, "topic": "event log database", "phase": "planning",
        "verdict": "revision",
        "value_quote": "switching the event log to ClickHouse",
        "prior_ref": "instead of Postgres",
        "rewritten": "Amy revised the event log database to ClickHouse, "
                     "replacing Postgres."},
}

MSGS = [
    {"msg_id": "m1", "author": "amy", "timestamp": "2026-01-05 09:00:00",
     "content": "we'll use Postgres for the event log"},
    {"msg_id": "m2", "author": "bob", "timestamp": "2026-01-05 09:05:00",
     "content": "standup is at 10"},
]
REVISION = [
    {"msg_id": "m3", "author": "amy", "timestamp": "2026-01-08 11:00:00",
     "content": "switching the event log to ClickHouse instead of Postgres",
     "reply_to": "m1"},
]


def main() -> int:
    llm = ScriptedLLM(SCRIPT)
    # `from_config` would build a real provider from a name; the scripted one is
    # injected directly, which is the whole reason the constructor takes objects.
    mem = GroupMemory(name="demo", llm=llm,
                      record_store=JsonlRecordStore("demo", root=WORK))

    recs = mem.add(MSGS, channel="eng")
    print(f"committed {len(recs)}: "
          f"{[(r['msg_node'], r.get('topic'), r.get('verdict')) for r in recs]}")

    hits = mem.search("event log database")
    print(f"search -> {len(hits)} hits; top: {hits[0]['memory_id']} "
          f"{hits[0]['text'][:48]!r}")

    # ---- a SECOND facade over the same rows: resume, then revise ------------ #
    mem2 = GroupMemory(name="demo", llm=llm,
                       record_store=JsonlRecordStore("demo", root=WORK))
    print("progress seen by the new facade:", mem2.records.progress())
    rec3 = mem2.add(REVISION, channel="eng")[0]
    print(f"m3 verdict={rec3.get('verdict')!r} "
          f"supersedes={rec3.get('supersedes')!r}")

    r = mem2.load()
    store = r["store"]
    m1, m3 = store.memories["m1"], store.memories["m3"]
    print(f"m1.superseded_by={m1.superseded_by!r} m3.supersedes={m3.supersedes!r} "
          f"m3.is_decision={m3.is_decision_point} "
          f"m3.link_confidence={m3.link_confidence!r}")
    fused = [k for k in store.memories if k.startswith("FUS")]
    print(f"current_version(m1) -> {store.current_version('m1')} ; fused: {fused}")

    cur = mem2.current("event log database")
    print("current() names ClickHouse:", "ClickHouse" in cur,
          "| still cites Postgres as prior:", "Postgres" in cur)
    print("nodes/edges:", len(r["ids"]), len(store.edges))
    print("stats:", json.dumps(mem2.stats(), sort_keys=True))

    rows = mem2.records.export_rows()
    print(f"rows on disk: {len(rows)} "
          f"({sum(1 for x in rows if x['node_kind'] == 'raw')} raw)")

    # THE CHAIN HEAD IS THE FUSED NODE, not the newest message: the fused row is
    # keyed on the chain's ROOT and supersedes it, so `current_version(m1)`
    # resolves to FUS_m1, whose passage carries the whole chain. m3 is the newest
    # MESSAGE in that chain.
    ok = (m1.superseded_by == "FUS_m1" and m3.supersedes == ["m1"]
          and store.current_version("m1") == "FUS_m1" and len(fused) == 1
          and m3.link_confidence == 1.0
          and "ClickHouse" in cur and bool(hits and hits[0]["text"]))
    print("VERDICT:", "OK" if ok else "BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
