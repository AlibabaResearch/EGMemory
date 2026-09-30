#!/usr/bin/env python3
"""Stage 3 of 3 -- read side: answer a domain's questions and score them.

    python3 bench/run_qa.py --domain Technology
    python3 bench/run_qa.py --domain Technology --qtypes knowledge_update,temporal

Per question: an agent loop over four tools (search_memory, lookup_phrase,
get_current_decision, resolve_date) against the one `gm_<domain>__<variant>`
table stage 1 wrote and stage 2 finalized, then an independent judge call
comparing the answer to gold. Results go to
`<results>/<arm>/<domain>__<qtype>.jsonl`, one JSON object per question, with a
`.meta.json` manifest beside it.

Flags:
  --n-max            the loop's round budget (10 here). It bounds cost and
                     divergence, not behaviour.
  --skip-existing    resume a batch: a result file counts as done when it has one
                     line per question AND no line carrying an `error` key.
  --workers          questions in flight. Questions are independent; ReadTools is
                     safe for concurrent use (it never mutates the store).

The judge prompt is a BENCHMARK asset, read from the GroupMemBench checkout
(`$GMB_ROOT/prompts/hipporag_judge_system.txt`) rather than vendored here, so
scoring cannot silently diverge from the benchmark's own definition of correct.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dashscope                                                  # noqa: E402

from groupmem.vector_stores import pgvector as pg_store
from groupmem.reranker import get_reranker
from groupmem.memory.reader import ReadTools, run_qa_agentloop, DECISION_RENDERER   # noqa: E402
from groupmem.memory.versioning import load_qa_store                   # noqa: E402
from groupmem.configs.defaults import (AGENT_MODEL, EMBEDDING_MODEL,     # noqa: E402
                                JUDGE_MODEL, RERANK_MODEL)
from groupmem.llms.dashscope import ds_call_tools, ds_chat        # noqa: E402
from groupmem.utils.io import load_env_file, read_text           # noqa: E402
from groupmem.prompts import load_prompt                          # noqa: E402
from groupmem.utils.text import split_reasoning_and_final        # noqa: E402
from bench.gmb_corpus import (GMB, data_path, load_questions,    # noqa: E402
                              questions_path)
from bench.judge import JUDGE_TEMPERATURE, parse_judgment        # noqa: E402
from groupmem.embeddings import get_embedder                     # noqa: E402
from bench.manifest import (build_manifest, prompt_sha,    # noqa: E402
                                   write_manifest)
from groupmem.memory.retrieval import FusionRetriever                # noqa: E402
from bench.stats import wilson_ci                          # noqa: E402

from groupmem.memory.passage import VERSION_STATUS_RENDERER

QA_PROMPT = "qa_agentloop_system.txt"
ANSWER_PARSER = "full_text_v1"
JUDGE_PROMPT = "prompts/hipporag_judge_system.txt"   # relative to $GMB_ROOT

# The six question types the four synthetic domains ship. Other corpora name
# their own, so --qtypes takes them verbatim; this is only the default.
DEFAULT_QTYPES = ("knowledge_update", "multi_hop", "temporal", "term_ambiguity",
                  "user_implicit", "abstention")


def result_path(results_dir: str, arm: str, domain: str, qtype: str) -> str:
    return os.path.join(results_dir, arm, f"{domain}__{qtype}.jsonl")


def _complete(path: str, n: int) -> bool:
    """True when `path` holds a finished run of `n` questions.

    A line count alone is not enough: `score_one` records a failed question as a
    row with an `error` key and counts it wrong, so that a crash cannot quietly
    shorten a result set. Those rows have to re-run, which is why they disqualify
    the file here.
    """
    if not os.path.exists(path):
        return False
    rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            if '"error":' in line:
                return False
    return rows >= n


def answer_one(q, tools, agent_system, judge_system, n_max: int) -> tuple:
    """Answer + judge one question. Returns (record, correct)."""
    qtext = q["question"]
    gold = q.get("answer", "")
    asking = q.get("asking_user_id") or ""

    def chat_fn(msgs, schemas):
        return ds_call_tools(dashscope, AGENT_MODEL, msgs, schemas)

    t0 = time.perf_counter()
    final_raw, trace = run_qa_agentloop(chat_fn, agent_system, qtext, asking,
                                        tools, n_max=n_max)
    t_loop = time.perf_counter() - t0
    agent_reasoning, agent_answer = split_reasoning_and_final(final_raw)

    # The judge is a separate call with temperature 0 and no view of the store,
    # so it can only compare two strings -- it cannot rescue an answer or invent
    # evidence for one.
    t1 = time.perf_counter()
    judge_out = ds_chat(dashscope, JUDGE_MODEL, judge_system,
                        f"Question:\n{qtext}\n\nGold Answer:\n{gold}\n\n"
                        f"Agent Answer:\n{agent_answer}\n",
                        temperature=JUDGE_TEMPERATURE)
    t_judge = time.perf_counter() - t1
    _, judge_answer = split_reasoning_and_final(judge_out, multiline=False)

    final_mark = next((t for t in reversed(trace)
                       if t.get("tool") == "__final__"), None)
    rec = {"id": q.get("id"), "query": qtext, "asking_user_id": asking,
           "tool_trace": trace,
           "n_steps": sum(1 for t in trace if t.get("tool") != "__final__"),
           "finish_reason": (final_mark or {}).get("finish_reason", ""),
           "agent_reasoning": agent_reasoning, "agent_answer": agent_answer,
           "agent_raw": final_raw, "answer_parser": ANSWER_PARSER,
           "version_status_renderer": VERSION_STATUS_RENDERER,
           "decision_renderer": DECISION_RENDERER,
           "judge_answer": judge_answer, "judge_raw": judge_out,
           # answer_s is the user-facing latency; the judge is evaluation
           # overhead and is timed separately rather than folded in.
           "timing": {"answer_s": round(t_loop, 3),
                      "judge_s": round(t_judge, 3)}}
    return rec, (parse_judgment(judge_answer) is True)


def score_one(q, tools, agent_system, judge_system, n_max: int) -> tuple:
    """answer_one, but a failure becomes a recorded wrong answer.

    Never let one question kill a batch of several hundred paid calls -- and
    never let it silently disappear either, which is what `--skip-existing`
    reads the `error` key for.
    """
    try:
        return answer_one(q, tools, agent_system, judge_system, n_max)
    except Exception as e:      # noqa: BLE001
        return ({"id": q.get("id"), "query": q.get("question"),
                 "error": f"{type(e).__name__}: {e}"}, False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read side: QA + judge over gm_<domain>__<variant>.")
    ap.add_argument("--domain", required=True,
                    help="one domain, or a comma-separated list")
    ap.add_argument("--variant", default="release", help="table variant")
    ap.add_argument("--qtypes", default=",".join(DEFAULT_QTYPES),
                    help="comma-separated question types; each is a "
                         "questions/<domain>/<qtype>.jsonl file")
    ap.add_argument("--arm", default="release",
                    help="name of the results subdirectory. Change it when "
                         "anything about the run changes, so two configurations "
                         "can never land in one file.")
    ap.add_argument("--results-dir", default=os.path.join(_ROOT, "results"))
    ap.add_argument("--workers", type=int, default=4,
                    help="questions in flight")
    ap.add_argument("--n-max", type=int, default=10,
                    help="tool-loop rounds before the forced final answer")
    ap.add_argument("--limit-questions", type=int, default=0,
                    help="0 = all; N = the first N of each qtype (pre-flight)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a (domain, qtype) whose result file is already "
                         "complete and error-free")
    ap.add_argument("--keep-noise", action="store_true",
                    help="load the rows the writer marked keep=false as well. "
                         "For corpora where that judgement does not transfer: on "
                         "dyadic personal conversation the write prompt marks "
                         "most messages noise, which would drop ~95%% of the "
                         "corpus out of retrieval. The rows are in the table "
                         "either way, so this is a WHERE clause and needs no "
                         "re-ingest.")
    # Provider selection. The DEFAULTS ARE THE SHIPPED ARM, which is the opposite
    # of the library factories (groupmem.reranker / groupmem.embeddings take no
    # default backend on purpose). A library must not guess; a reproduction script
    # must reproduce, and `--rerank noop` is then the ablation rather than an edit.
    ap.add_argument("--rerank", default="dashscope",
                    choices=["dashscope", "noop"],
                    help="selection backend. 'noop' keeps the fused order and is "
                         "the ABLATION arm; it is not a "
                         "way to run without a key and still report a number.")
    ap.add_argument("--embedder", default="dashscope_native",
                    choices=["dashscope_native", "dashscope", "openai",
                             "hashing"],
                    help="dense channel backend. MUST MATCH WHAT THE TABLE WAS "
                         "EMBEDDED WITH: a mismatch does not error, cosine still "
                         "returns a number and retrieval silently degrades.")
    args = ap.parse_args()

    load_env_file(os.path.join(GMB, ".env"))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set (nor in "
                         f"{os.path.join(GMB, '.env')})")
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

    agent_system = load_prompt(QA_PROMPT)
    judge_path = os.path.join(GMB, JUDGE_PROMPT)
    if not os.path.exists(judge_path):
        raise SystemExit(f"judge prompt not found at {judge_path}. Point "
                         f"GMB_ROOT at the GroupMemBench checkout.")
    judge_system = read_text(judge_path)
    # Both prompts are fingerprinted into the manifest (see _run_domain), not
    # pinned. The judge IS the metric, so recording its hash is what makes two runs
    # scored by different judge texts visibly incomparable rather than silently so.
    # Editing either prompt is allowed; the recorded hash simply follows the edit.

    qtypes = [q.strip() for q in args.qtypes.split(",") if q.strip()]
    domains = [d.strip() for d in args.domain.split(",") if d.strip()]

    # The cross-encoder decides every passage slot the agent sees, and it fails
    # OPEN -- so a dead rerank endpoint does not stop the run, it just quietly
    # scores it like the pre-rerank pipeline. That is found out here rather than
    # after paying for the whole batch.
    #
    # ONE instance for the whole run, threaded through _run_domain: the counters
    # printed at the end have to be the counters the retriever bumped, or the
    # degradation check below reads zero on a run that degraded.
    reranker = get_reranker(args.rerank)
    if not reranker.probe():
        raise SystemExit(f"rerank probe failed ({RERANK_MODEL}). Rerank fails "
                         f"open, so the run WOULD complete and score low for a "
                         f"reason nothing in the results would name. Check the "
                         f"key's entitlement for this model, then rerun.")

    print(f"arm {args.arm}: {len(domains)} domain(s) x {len(qtypes)} qtype(s), "
          f"agent {AGENT_MODEL}, judge {JUDGE_MODEL} @T{JUDGE_TEMPERATURE}, "
          f"embedding {EMBEDDING_MODEL}, rerank {RERANK_MODEL}, "
          f"n_max {args.n_max}, "
          f"qa prompt {prompt_sha(agent_system)[:8]}, "
          f"judge prompt {prompt_sha(judge_system)[:8]}", flush=True)

    rc = 0
    for domain in domains:
        if not pg_store.table_exists(domain, args.variant):
            print(f"[{domain}] {pg_store.table(domain, args.variant)} is missing "
                  f"-- run bench/run_ingest.py then bench/finalize.py first",
                  flush=True)
            rc = 1
            continue
        rc |= _run_domain(domain, qtypes, args, agent_system, judge_system,
                          reranker)

    # Printed once at the end, because a fail-open reranker is invisible in the
    # accuracy itself: `passthrough` counts searches that silently fell back to
    # the fused order, and any of those means this number is a blend of two
    # configurations rather than a measurement of one.
    rs = reranker.stats()
    degraded = sum(v for k, v in rs.items()
                   if k in ("passthrough", "fail", "no_key")
                   or k.startswith("http_"))
    print(f"[rerank] {rs}", flush=True)
    if degraded:
        print(f"[rerank] WARNING: {degraded} search call(s) fell back to the "
              f"un-reranked order. Do not report this run as a rerank result.",
              flush=True)
    return rc


def _run_domain(domain, qtypes, args, agent_system, judge_system,
                reranker) -> int:
    # The store is loaded ONCE per domain and shared by every qtype: assembly
    # regates the supersedes edges and rebuilds the BM25 index, which is minutes
    # of work that does not depend on the questions.
    t0 = time.time()
    store, idx, lexical = load_qa_store(domain, variant=args.variant,
                                        keep_noise=args.keep_noise)
    embedder = get_embedder(args.embedder, cache_dir=None)
    retr = FusionRetriever(store, idx, embedder, lexical, reranker)
    tools = ReadTools(store, retr)
    print(f"[{domain}] store ready: {len(idx)} nodes, {len(store.edges)} edges "
          f"({time.time()-t0:.0f}s)", flush=True)

    for qtype in qtypes:
        qpath = questions_path(domain, qtype)
        if not os.path.exists(qpath):
            print(f"[{domain}/{qtype}] no question file at {qpath}; skipped",
                  flush=True)
            continue
        questions = load_questions(qpath)
        if args.limit_questions:
            questions = questions[: args.limit_questions]
        out_path = result_path(args.results_dir, args.arm, domain, qtype)
        if args.skip_existing and _complete(out_path, len(questions)):
            print(f"[{domain}/{qtype}] already complete ({len(questions)}); "
                  f"skipped", flush=True)
            continue

        t1 = time.time()
        # Submitted in order and collected by index, so the file's line order is
        # the question file's order regardless of which finished first -- a
        # result set has to be diffable against another arm's line by line.
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(score_one, q, tools, agent_system, judge_system,
                              args.n_max) for q in questions]
            rows = [fu.result() for fu in futs]
        correct = sum(1 for _, ok in rows if ok)
        n = len(rows)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec, ok in rows:
                rec["correct"] = ok
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_manifest(out_path, build_manifest(
            domain=domain, qtype=qtype, config=args.arm,
            corpus_path=data_path(domain), repo_dir=_ROOT,
            agent_model=AGENT_MODEL, judge_model=JUDGE_MODEL,
            emb_model=EMBEDDING_MODEL, emb_backend=args.embedder,
            top_k=None, limit_questions=args.limit_questions or None,
            n_messages=len(idx), graph_stats=store.stats(),
            accuracy=(correct / n if n else None), correct=correct, total=n,
            extra={"variant": args.variant, "n_max": args.n_max,
                   "rerank_backend": args.rerank,
                   "qa_prompt_sha": prompt_sha(agent_system)[:8],
                   "judge_prompt_sha": prompt_sha(judge_system)[:8],
                   "answer_parser": ANSWER_PARSER,
                   "version_status_renderer": VERSION_STATUS_RENDERER,
                   "decision_renderer": DECISION_RENDERER,
                   "tools": [t["function"]["name"]
                             for t in tools.tool_schemas()]}))
        lo, hi = wilson_ci(correct, n)
        print(f"[{domain}/{qtype}] acc={correct}/{n}="
              f"{(correct/n if n else 0):.3f} 95%CI=[{lo:.3f},{hi:.3f}] "
              f"-> {out_path} ({time.time()-t1:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
