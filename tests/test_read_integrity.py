"""Offline read-side regression: no model calls, no database."""
from datetime import datetime
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from groupmem.memory import reader
from groupmem.memory.passage import format_passage
from groupmem.memory.versioning import regate_supersedes
from groupmem.schema import Memory, MemoryStore, Relation
from groupmem.utils.text import extract_final, split_reasoning_and_final
from groupmem.vector_stores import base, pgvector as pg
from bench import run_qa as qa, summarize
from bench.judge import parse_judgment


class FakeTools:
    def __init__(self, result="evidence"):
        self.result = result

    def new_session(self):
        pass

    def tool_schemas(self):
        return []

    def dispatch(self, name, args):
        return self.result


def store(confidence="high"):
    cols = base.GM_COLS
    rows = []
    for mid, prev, kind, sources in (
        ("old", None, "raw", []), ("middle", "old", "raw", []),
        ("latest", "middle", "raw", []),
        ("FUS_old", None, "fusion", ["old", "middle", "latest"]),
    ):
        row = dict.fromkeys(cols)
        row.update(memory_id=mid, content=mid, rewritten=mid + "\nfull body",
                   keep=True, node_kind=kind, sources=sources,
                   verdict=("revision" if prev else "new") if kind == "raw" else None,
                   decision_quote=mid, supersedes=prev, link_conf=confidence)
        rows.append(row)
    return base.build_store(rows)[0]


class ParserTests(unittest.TestCase):
    def test_unmarked_multiline(self):
        text = "item one\n\nitem two\n  - subitem\nlast item"
        self.assertEqual(split_reasoning_and_final(text), ("", text))

    def test_marked_multiline(self):
        self.assertEqual(split_reasoning_and_final("thinking\nFinal: item one\nitem two"),
                         ("thinking", "item one\nitem two"))

    def test_empty_marker_keeps_list(self):
        self.assertEqual(split_reasoning_and_final("thinking\nFINAL ANSWER:\n- a\n- b\n"),
                         ("thinking", "- a\n- b"))

    def test_last_marker_and_crlf(self):
        self.assertEqual(split_reasoning_and_final("Final: draft\r\nrevised\r\nFinal: a\r\nb\r\n"),
                         ("Final: draft\nrevised", "a\nb"))

    def test_empty_and_single_line(self):
        for text in ("", " \n ", None):
            self.assertEqual(split_reasoning_and_final(text), ("", ""))
        self.assertEqual(split_reasoning_and_final("Final: OK"), ("", "OK"))

    def test_no_character_cap(self):
        text = "head\n" + "body" * 10000 + "\ntail"
        self.assertEqual(split_reasoning_and_final(text)[1], text)
        self.assertEqual(split_reasoning_and_final("Final:\n" + text)[1], text)

    def test_judge_unchanged(self):
        for text, verdict in (("analysis\nCorrect", True), ("analysis\nIncorrect", False),
                              ("analysis\nFinal: Correct\nnote", True),
                              ("Final answer: Not correct", False), ("Unknown", None)):
            final = split_reasoning_and_final(text, multiline=False)[1]
            self.assertEqual(final, extract_final(text))
            self.assertIs(parse_judgment(final), verdict)


class LoopTests(unittest.TestCase):
    def test_separate_reasoning_stays_separate(self):
        for think in ("separate reasoning", "preview\nFinal: draft that should not be scored"):
            body = "1. full item one\n2. full item two"
            chat = Mock(return_value={"content": body, "reasoning": think, "finish_reason": "stop"})
            raw, trace = reader.run_qa_agentloop(chat, "s", "q", "", FakeTools())
            self.assertEqual(split_reasoning_and_final(raw), (think, body))
            self.assertEqual(trace[-1]["finish_reason"], "stop")

    def test_empty_body_and_marked_body(self):
        for body, think, expected in (("", "Final: a\nb", ("", "a\nb")),
                                      ("", "a\nb", ("", "a\nb")),
                                      ("Final: a\nb", "reasoning", ("reasoning", "a\nb"))):
            raw, _ = reader.run_qa_agentloop(Mock(return_value={"content": body, "reasoning": think}),
                                            "s", "q", "", FakeTools())
            self.assertEqual(split_reasoning_and_final(raw), expected)

    def test_full_trace_and_forced_final(self):
        evidence = "evidence" * 5000 + "evidence-end"
        deliberation = "reasoning" * 3000 + "reasoning-end"
        answer = "item one\n" + "answer" * 5000 + "\ntail"
        tc = {"id": "call_1", "function": {"name": "search_memory", "arguments": "{}"}}
        for n_max in (1, 2):
            responses = iter([
                {"content": "", "reasoning": deliberation, "tool_calls": [tc]},
                {"content": answer, "reasoning": "final reasoning", "finish_reason": "stop"},
            ])
            seen = []

            def chat(messages, schemas):
                seen.append(list(messages))
                return next(responses)

            raw, trace = reader.run_qa_agentloop(chat, "s", "q", "", FakeTools(evidence), n_max=n_max)
            self.assertEqual(trace[0]["ret"], evidence)
            self.assertEqual(trace[0]["think"], deliberation)
            self.assertEqual([m["content"] for m in seen[-1] if m["role"] == "tool"], [evidence])
            self.assertEqual(split_reasoning_and_final(raw), ("final reasoning", answer))
            self.assertEqual(bool(trace[-1].get("forced")), n_max == 1)


class HarnessTests(unittest.TestCase):
    def test_judge_receives_full_answer(self):
        for content in ("a\nb", "Final: a\nb", "Final:\na\nb"):
            with patch.object(qa, "ds_call_tools", return_value={
                "content": content, "reasoning": "separate reasoning", "finish_reason": "stop"
            }), patch.object(qa, "ds_chat", return_value="judge reasoning\nCorrect") as judge:
                rec, ok = qa.answer_one({"id": "q", "question": "question", "answer": "answer"},
                                        FakeTools(), "agent", "judge", 2)
            self.assertTrue(ok)
            self.assertEqual(rec["agent_answer"], "a\nb")
            self.assertEqual(rec["agent_reasoning"], "separate reasoning")
            self.assertEqual(rec["answer_parser"], "full_text_v1")
            self.assertEqual(rec["version_status_renderer"], "raw_revision_v1")
            self.assertEqual(rec["decision_renderer"], "chain_scope_v1")
            self.assertEqual(rec["judge_raw"], "judge reasoning\nCorrect")
            self.assertEqual(split_reasoning_and_final(rec["agent_raw"])[1], rec["agent_answer"])
            self.assertTrue(judge.call_args.args[3].endswith("Agent Answer:\na\nb\n"))
            self.assertNotIn("separate reasoning", judge.call_args.args[3])

    def test_http_answer_and_chat(self):
        from groupmem.server import Handler
        for raw in ("item one\nitem two", "reasoning\nFinal: item one\nitem two"):
            handler = object.__new__(Handler)
            handler.state = SimpleNamespace(mem=SimpleNamespace(answer=Mock(return_value=raw)))
            handler._send = Mock()
            handler._stream_one = Mock()
            handler._answer({"question": "q"})
            self.assertEqual(handler._send.call_args.args[1]["answer"], "item one\nitem two")
            handler._chat_completion({"messages": [{"role": "user", "content": "q"}]})
            self.assertEqual(handler._send.call_args.args[1]["choices"][0]["message"]["content"],
                             "item one\nitem two")
            handler._chat_completion({"messages": [{"role": "user", "content": "q"}], "stream": True})
            self.assertEqual(handler._stream_one.call_args.args[0], "item one\nitem two")

    def test_reference_counts(self):
        self.assertEqual(summarize.REFERENCE_TOTAL, (502, 745))
        self.assertEqual(tuple(map(sum, zip(*summarize.REFERENCE.values()))), (502, 745))


class VersionTests(unittest.TestCase):
    def header(self, s, mid):
        return format_passage(s.memories[mid]).split("\n", 1)[0]

    def test_fusion_does_not_retire_latest(self):
        s = store()
        regate_supersedes(s)
        self.assertEqual(s.memories["latest"].superseded_by, "FUS_old")
        self.assertIsNone(s.memories["latest"].decision_superseded_by)
        self.assertIn("status=CURRENT_DECISION", self.header(s, "latest"))
        self.assertIn("status=CURRENT_DECISION", self.header(s, "FUS_old"))

    def test_old_versions_retired(self):
        s = store()
        regate_supersedes(s)
        for mid, newer in (("old", "middle"), ("middle", "latest")):
            self.assertEqual(s.memories[mid].decision_superseded_by, newer)
            self.assertIn("status=SUPERSEDED_BY", self.header(s, mid))

    def test_demoted_edges_clear_business_status(self):
        s = store("low")
        regate_supersedes(s)
        for mid in s.order:
            self.assertIsNone(s.memories[mid].decision_superseded_by)
            self.assertNotIn("status=", self.header(s, mid))
            self.assertIsNone(s.memories[mid].decision_prior_value)

    def test_fusion_alone_not_replacement(self):
        s = MemoryStore()
        s.add(Memory("source", "still valid"))
        s.add(Memory("fused", "copy", node_kind="fusion", sources=["source"]))
        s.link("fused", "source", Relation.SUPERSEDES)
        self.assertEqual(s.current_version("source"), "fused")
        self.assertNotIn("status=", self.header(s, "source"))

    def test_graph_and_index_text_unchanged(self):
        s = store()
        edges = list(s.edges)
        texts = {mid: m.index_text("rewritten") for mid, m in s.memories.items()}
        regate_supersedes(s)
        self.assertEqual(s.edges, edges)
        for mid, m in s.memories.items():
            self.assertEqual(s.current_version(mid), "FUS_old")
            self.assertEqual(m.index_text("rewritten"), texts[mid])
            self.assertEqual(format_passage(m).split("\n", 1)[1], texts[mid])

    def test_repeated_gate(self):
        s = store()
        regate_supersedes(s)
        before = {mid: self.header(s, mid) for mid in s.order}
        regate_supersedes(s)
        self.assertEqual(before, {mid: self.header(s, mid) for mid in s.order})


class DecisionScopeTests(unittest.TestCase):
    def tools(self, confidence="high", exact=True, cap=12):
        s = store(confidence)
        regate_supersedes(s)
        s.by_topic["t"] = list(s.order)
        return reader.ReadTools(s, SimpleNamespace(_heads_by_topic={"t": ["FUS_old"]} if exact else {}),
                                max_passages=cap)

    def test_head_and_history_separated(self):
        tools = self.tools()
        out = tools.get_current_decision("t")
        self.assertIn("stored chain head source=latest", out)
        self.assertIn("[HEAD VALUE -- only this chain's decision field]\nlatest", out)
        self.assertIn("historical and current", out)
        self.assertIn(tools._passage(tools.store.memories["FUS_old"]), out)
        self.assertIn("NOT semantically verified", out)

    def test_independent_chains(self):
        tools = self.tools()
        s = tools.store
        s.add(Memory("other_old", "other item old value"))
        s.add(Memory("other", "other item new value", decision_new_value="other item new value"))
        s.link("other", "other_old", Relation.SUPERSEDES)
        tools.retr._heads_by_topic["t"].append("other")
        edges = list(s.edges)
        out = tools.get_current_decision("t")
        for text in ("NOT one evolving decision", "Different fields can coexist",
                     "decision candidate=FUS_old", "decision candidate=other;"):
            self.assertIn(text, out)
        self.assertEqual(edges, s.edges)
        self.assertIsNone(s.memories["other"].decision_superseded_by)

    def test_single_fallback_warned(self):
        out = self.tools(exact=False).get_current_decision("t")
        for text in ("showing 1 of 1", "NO VERIFIED CHAIN HEADS", "UNVERIFIED CANDIDATE"):
            self.assertIn(text, out)
        self.assertNotIn("[HEAD VALUE", out)

    def test_weak_chain_not_trusted(self):
        out = self.tools(confidence="low").get_current_decision("t")
        self.assertIn("UNVERIFIED CANDIDATE", out)
        self.assertNotIn("[HEAD VALUE", out)
        self.assertIn("FUS_old\nfull body", out)

    def test_broken_sources_not_trusted(self):
        for broken in ("missing", "disconnected", "duplicate"):
            tools = self.tools()
            if broken == "missing":
                tools.store.memories.pop("middle")
            elif broken == "disconnected":
                tools.store.memories["latest"].supersedes = []
            else:
                tools.store.memories["FUS_old"].sources.append("latest")
            out = tools.get_current_decision("t")
            self.assertIn("UNVERIFIED CANDIDATE", out)
            self.assertNotIn("[HEAD VALUE", out)

    def test_stale_head_not_current(self):
        tools = self.tools()
        tools.store.add(Memory("newer", "new revision"))
        tools.store.link("newer", "latest", Relation.SUPERSEDES)
        self.assertIn("UNVERIFIED CANDIDATE", tools.get_current_decision("t"))

    def test_quote_and_body_not_capped(self):
        tools = self.tools()
        quote = "head\n" + "current value" * 5000 + "\ntail"
        body = "old value\n" + "historical evidence" * 5000 + "\nbody-end"
        tools.store.memories["latest"].decision_new_value = quote
        tools.store.memories["FUS_old"].content_rewritten = body
        out = tools.get_current_decision("t")
        self.assertIn(quote, out)
        self.assertIn(body, out)

    def test_missing_quote_no_substitute(self):
        tools = self.tools()
        tools.store.memories["latest"].decision_new_value = None
        tools.store.memories["FUS_old"].decision_new_value = "not a substitute"
        out = tools.get_current_decision("t")
        self.assertIn("No stored head-value quote", out)
        self.assertNotIn("[HEAD VALUE", out)

    def test_timestamp_cap_and_other_tools(self):
        tools = self.tools(cap=1)
        tools.store.add(Memory("newest", "new candidate", timestamp=datetime(2026, 1, 1)))
        tools.retr._heads_by_topic["t"].append("newest")
        before = tools._fmt(["FUS_old", "newest"])
        out = tools.get_current_decision("t")
        self.assertIn("showing 1 of 2", out)
        self.assertIn("PARTIAL TOPIC INDEX", out)
        self.assertIn(tools._passage(tools.store.memories["newest"]), out)
        self.assertNotIn("decision candidate=FUS_old", out)
        self.assertEqual(before, tools._fmt(["FUS_old", "newest"]))
        self.assertEqual(tools.get_current_decision("missing"),
                         "No current decision found for topic 'missing'.")


class WorkerTests(unittest.TestCase):
    def test_abandoned_request_is_skipped(self):
        index = object.__new__(pg.PgVectorIndex)
        index._q = queue.Queue()
        abandoned = Mock()
        live = Mock(return_value="ok")
        event = threading.Event()
        result = {}
        index._q.put((abandoned, {"abandoned": True}, threading.Event()))
        index._q.put((live, result, event))
        index._q.put((None, {}, threading.Event()))
        index._db_worker()
        abandoned.assert_not_called()
        live.assert_called_once()
        self.assertTrue(event.is_set())
        self.assertEqual(result["val"], "ok")

    def test_timeout_marks_abandoned(self):
        index = object.__new__(pg.PgVectorIndex)
        index._worker = SimpleNamespace(is_alive=lambda: True)
        index._q = queue.Queue()
        index.table = "test"
        event = Mock()
        event.wait.return_value = False
        with patch.object(pg.threading, "Event", return_value=event):
            with self.assertRaises(TimeoutError):
                index._run_on_worker(lambda: None)
        self.assertTrue(index._q.get()[1]["abandoned"])


if __name__ == "__main__":
    unittest.main()
