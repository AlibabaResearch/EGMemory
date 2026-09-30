#!/usr/bin/env python3
"""The write path -- GroupMemBench corpus -> gm_<domain>__<variant>.

    python3 bench/run_ingest.py --domain Technology --workers 3

One writer per channel, messages serial inside a channel (the decision registry
accumulates as it goes), channels in parallel. Per message the writer commits
the message's own row AND the fused row of the supersedes chain it extended, in
the same transaction, so nothing has to run after this script for the table to
be complete. `finalize.py` reconciles those fused rows over a whole domain and
is run at the end of a completed run, where it should find nothing to do.

Operational behaviour this script owns:
  durability   one message == one transaction, one row; an interruption costs
               only the message in flight.
  resume       `channel_progress` says how far each channel got; `restore_channel`
               replays that channel's rows through the registry with zero LLM calls.
  drift guard  a resume must not cross prompt or code versions -- see the
               `promptfp` block below.
  moderation   after two content-400 attempts a message is committed as a
               keep=False stub so the row count stays complete and resume never loops.
  export       the .jsonl product is regenerated FROM the table at the end, never
               assembled from what this process happened to handle.

Postgres is the authority; `stores/<variant>_records_<domain>.jsonl` is a
readable export of it. `groupmem/configs/defaults.py` is the only configurable
surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dashscope                                                  # noqa: E402

from groupmem.vector_stores import pgvector as pg_store
from groupmem.storage.pg import PgRecordStore                     # noqa: E402
from groupmem.memory.writer import MemoryWriter             # noqa: E402
from groupmem.configs.defaults import EMBEDDING_MODEL, WRITE_MODEL       # noqa: E402
from groupmem.llms.base import HARD_TIMEOUT_WRITE, THINKING_WRITE  # noqa: E402
from groupmem.llms.dashscope import ds_call_tools, ds_chat        # noqa: E402
from groupmem.utils.io import load_env_file, read_text           # noqa: E402
from groupmem.prompts import load_prompt                          # noqa: E402
from groupmem.embeddings import get_embedder                     # noqa: E402
from groupmem.embeddings.dashscope import NATIVE_DIM             # noqa: E402
from groupmem.memory.pipeline import ingest_channel               # noqa: E402
from bench.gmb_corpus import GMB, data_path, load_messages       # noqa: E402
from bench.manifest import prompt_sha                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finalize import finalize_domain                              # noqa: E402

STORES = os.path.join(_ROOT, "stores")

# THE ONE WRITE-SIDE PROMPT. Its fingerprint is recorded (printed at startup and
# stamped into stores/promptfp_*.json), not asserted: an edit is allowed and simply
# shows up in the stamp. There is one prompt per role, and no selection mechanism.
MAIN_PROMPT = "write_system.txt"        # stage-1 fields + VERDICT RULES

# WHERE `phase_name` COMES FROM, and the reason this is a constant rather than a
# flag: the label the reader groups by is ALWAYS the writer's own `phase` field.
# Copying the corpus's label instead borrows the dataset builder's own session
# structure, which is not a signal a deployment has. On the synthetic domains the
# corpus label is the synthesizer's topic segmentation outright, i.e. handing back
# the partition the writer exists to infer. So there is no corpus for which it is
# legitimate, and a knob whose every setting but one is forbidden is a knob that
# gets turned by accident.
PHASE_SRC = "record"


def _writer_code_sha() -> str:
    """sha of the writer's own modules, for the resume guard.

    Prompts are not the only thing that can change under a resume: a run can get
    thousands of messages in before memory/writer.py gains, say, a salvage path
    for fields returned as bare JSON, with every prompt hash identical. Not just
    the loop: `memory/registry` decides which slot a message matches, which is as
    load-bearing as the loop being edited, and `fusion` renders the fused row the
    loop now commits per message -- edit its template mid-run and a table ends up
    holding two renderings. (That one is recoverable, unlike the others:
    `bench/finalize.py` re-renders every chain. It is in the fingerprint so the
    run tells you, rather than so you cannot fix it.)

    THE FILE NAMES ARE PART OF THE HASH, so renaming a hashed module moves this
    value even when the writer's behaviour does not change: a store stamped before
    such a rename will trip the drift guard below on RESUME. That is the guard
    working as specified -- it refuses to continue one store under two versions of
    the code and cannot tell a rename from a rewrite. Finished stores are
    unaffected (the read side never reads this stamp); to extend one, either
    re-stamp its stores/promptfp_<domain>_<variant>.json by hand or use a new
    --variant.
    """
    h = hashlib.sha256()
    for name in ("writer.py", "registry.py", "fusion.py"):
        p = os.path.join(_ROOT, "groupmem", "memory", name)
        with open(p, "rb") as f:
            h.update(name.encode())
            h.update(f.read())
    return h.hexdigest()[:8]


def _check_prompt_fingerprint(domain: str, variant: str, fp: dict) -> None:
    """Stamp the fingerprint on first write; refuse to resume under a different
    one.

    Resume otherwise only checks HOW MANY messages are committed, so editing a
    prompt and restarting produces one store written under two sets of rules:
    part of it judged revisions by one prompt and the rest by another, with
    nothing on screen to say so.
    """
    path = os.path.join(STORES, f"promptfp_{domain}_{variant}.json")
    if not os.path.exists(path):
        # Atomic, because a --shard'ed domain has several processes racing for
        # this one file. A torn write parses back as None, which reads as
        # "nothing recorded" and silently disarms the guard.
        os.makedirs(STORES, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(fp, f, indent=1)
        os.replace(tmp, path)
        print(f"[promptfp] stamped {os.path.basename(path)}: {fp}", flush=True)
        return
    try:
        old = json.load(open(path, encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        old = None
    # Compare only keys the stamp ACTUALLY RECORDED, so adding a key later never
    # blocks the resume of a run already in flight.
    drift = [k for k in fp if k in (old or {}) and old[k] != fp[k]]
    if drift:
        raise SystemExit(
            f"PROMPT DRIFT on resume of {domain}/{variant}: "
            + ", ".join(f"{k} {old.get(k)} -> {fp[k]}" for k in drift)
            + f"\nThe rows already in "
              f"{pg_store.table(domain, variant)} were written under "
              f"the first set; continuing would mix two rulesets in one store. "
              f"Use a new --variant, or restore what changed.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="write side: corpus -> gm_<domain>__<variant>.")
    ap.add_argument("--domain", required=True,
                    help="corpus name; also names the table and the product")
    ap.add_argument("--variant", default="release",
                    help="table suffix. The table is the whole store -- the "
                         "reader loads these rows directly -- so use a NEW "
                         "variant whenever write-side behaviour changes, and a "
                         "baseline is never overwritten in place.")
    ap.add_argument("--workers", type=int, default=3,
                    help="channels in flight. Messages inside a channel stay "
                         "serial -- the registry is built one at a time -- so "
                         "this is the only axis of parallelism.")
    ap.add_argument("--corpus", default="",
                    help="corpus JSON; default = the GroupMemBench path for "
                         "--domain. json.load on a full domain costs ~553 MB "
                         "resident, so a pilot is better served by a pre-cut "
                         "prefix file than by --limit on the whole thing.")
    ap.add_argument("--channels-limit", type=int, default=0,
                    help="0 = all channels; N = the first N (cheap pre-flight)")
    ap.add_argument("--channels-file", default="",
                    help="path to a channel allowlist, one name per line; "
                         "process EXACTLY those. For re-running the subset that "
                         "finished before a run was killed.")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = whole channel; N = the first N messages of each. A "
                         "PREFIX, never a sample: the registry is built in "
                         "channel order, so the first N see exactly the history "
                         "they would in a full run and can be diffed message by "
                         "message against another variant. A random sample is "
                         "comparable to nothing, because every message's "
                         "[REGISTRY] view would be missing.")
    ap.add_argument("--shard", default="",
                    help="i/n: process only the i-th of n channel shards, so one "
                         "domain can be split across API keys. Channels share no "
                         "state, so a shard is self-contained and no link is "
                         "lost. Shards are filled round-robin over "
                         "size-descending channels because wall clock is set by "
                         "the largest channel.")
    ap.add_argument("--no-finalize", action="store_true",
                    help="skip the whole-domain reconciliation of the fused "
                         "rows at the end of a completed corpus. The rows "
                         "themselves are written per message either way, so this "
                         "only skips the check that finalize.py's rendering "
                         "agrees with the writer's -- and that check is "
                         "re-runnable at any time.")
    args = ap.parse_args()

    load_env_file(os.path.join(GMB, ".env"))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set (nor in "
                         f"{os.path.join(GMB, '.env')})")
    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

    sys_prompt = load_prompt(MAIN_PROMPT)
    # The write prompt's fingerprint, RECORDED not enforced: it is printed below and
    # stamped into stores/promptfp_*.json (see _check_prompt_fingerprint) so a run
    # is attributable to the exact prompt bytes. Editing the prompt is allowed --
    # the recorded hash simply follows the edit.
    sha = {"main": prompt_sha(sys_prompt)[:8]}

    msgs = load_messages(args.corpus or data_path(args.domain))
    # THE `ord` COLUMN, decided here and nowhere else: a message's position in the
    # corpus file. It is the order the reader walks the store in, so it must not
    # depend on which shard or --limit produced the row -- hence the whole msgs
    # list, before any filtering. Two consequences worth knowing: rows dropped as
    # noise leave holes (only the relative order is a contract), and the fused
    # rows finalize.py appends sit after every message by construction.
    gord = {m["msg_node"]: i for i, m in enumerate(msgs)}
    by_ch = defaultdict(list)
    for m in msgs:
        by_ch[m["_channel"]].append(m)
    chans = sorted(by_ch)
    if args.channels_file:
        with open(args.channels_file, encoding="utf-8") as f:
            want = {ln.strip() for ln in f if ln.strip()}
        missing = want - set(chans)
        if missing:
            print(f"WARN channels not in corpus: {sorted(missing)}", flush=True)
        chans = [c for c in chans if c in want]
    if args.channels_limit:
        chans = chans[: args.channels_limit]
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        by_size = sorted(chans, key=lambda c: -len(by_ch[c]))
        chans = [c for j, c in enumerate(by_size) if j % sn == si - 1]
    if args.limit:
        for c in chans:
            by_ch[c] = by_ch[c][: args.limit]
    n_msgs = sum(len(by_ch[c]) for c in chans)
    print(f"{args.domain}/{args.variant}: {n_msgs} msgs, "
          f"{len(chans)}/{len(by_ch)} channels, "
          f"model {WRITE_MODEL}, embedding {EMBEDDING_MODEL}, "
          f"thinking={'ON' if THINKING_WRITE else 'OFF'}, "
          f"phase_src {PHASE_SRC}, prompt "
          + " ".join(f"{k} {v}" for k, v in sha.items()), flush=True)

    # The column width is the embedder's, from the one place that states it --
    # the table is created here, before line ~350 builds the embedder, so the
    # number cannot be read off a vector.
    pg_store.ensure_table(args.domain, args.variant, dim=NATIVE_DIM)
    print(f"table ready: {pg_store.table(args.domain, args.variant)}",
          flush=True)
    # The configurable surface goes into the fingerprint for the same reason the
    # code does: overriding WRITE_MODEL or EMBEDDING_MODEL halfway through builds
    # a store whose halves are not the same system, and nothing else would catch
    # it. phase_src is in there because it decides the label the reader groups
    # by, and half a store partitioned one way is not one store.
    _check_prompt_fingerprint(args.domain, args.variant, {
        **sha, "code": _writer_code_sha(),
        "write_model": WRITE_MODEL, "emb_model": EMBEDDING_MODEL,
        "phase_src": PHASE_SRC,
        "thinking": "1" if THINKING_WRITE else "0"})

    # ---- no LLM reply store ---------------------------------------------- #
    # Recording every raw reply would be by far the largest thing this pipeline
    # stores -- most of the whole footprint -- for a table the production path
    # never reads; its only consumer would be a resume-time check that the rebuilt
    # prompt matched the recorded one. A resumed run simply makes live calls for
    # the messages it has not committed. The writer's cache_get/cache_put hook is
    # gone as well: this was the only place that could have supplied it and it
    # never did, so every call site was a branch on a constant None.
    #
    # It could not have been a cost saver either: keying a reply cache by
    # mid + sha(user_prompt) fails because the user prompt embeds the mutable
    # [REGISTRY] block, so one flipped decision re-keys every later message in the
    # channel and the tail of every channel misses.

    # No output cap on either call, and thinking OFF -- both stated once here
    # rather than at each writer call site. A truncated reply is not a shorter
    # record: parse_plan_json returns {} and _commit then defaults keep=True with
    # topic/rewritten empty, so the node drops out of topic grouping and falls
    # back to verbatim in the index. Cost is unaffected: billing follows tokens
    # actually generated and the prompts end by naming the tool to call.
    def chat(sp, up):
        return ds_chat(dashscope, WRITE_MODEL, sp, up, max_tokens=0,
                       thinking=THINKING_WRITE,
                       hard_timeout=HARD_TIMEOUT_WRITE)

    def chat_tools(messages, schemas):
        return ds_call_tools(dashscope, WRITE_MODEL, messages, schemas,
                             max_tokens=0, thinking=THINKING_WRITE,
                             hard_timeout=HARD_TIMEOUT_WRITE)

    # No local embedding cache. Ingestion is DB-backed; the old JSONL cache
    # loaded ENTIRELY into a dict at startup (~8 GB resident per process) and
    # swapped a 16 GB machine as soon as two domains ran at once. Only
    # message-content vectors could hit anyway -- registry slot and pool-query
    # texts are LLM output and change with the prompt -- and a miss costs ~0.3s
    # against a ~4.8s main call.
    emb = get_embedder("dashscope_native", cache_dir=None)

    t0 = time.time()
    # The store is a RecordStore, not this script's own SQL: the facade writes
    # through the same interface, so there is one commit path and one resume path.
    # Swap PgRecordStore for storage.jsonl.JsonlRecordStore to run with no DB.
    rec_store = PgRecordStore(args.domain, args.variant)
    progress = rec_store.progress()
    if progress:
        print(f"resume: {sum(v + 1 for v in progress.values())} messages already "
              f"committed across {len(progress)} channels", flush=True)

    def run_channel(ch):
        # One writer AND one handle per channel: channels share no state, and a
        # single serialized DB thread would become the bottleneck. The loop itself
        # is groupmem.memory.pipeline -- library code, so the facade and this
        # script drive the writer through exactly one implementation.
        w = MemoryWriter(chat, sys_prompt, emb.embed, chat_tools_fn=chat_tools,
                         phase_src=PHASE_SRC)
        sl = by_ch[ch]
        start = 0
        done = rec_store.restore(w, ch)
        if done:
            start = progress.get(ch, done - 1) + 1
            print(f"  [{ch}] resumed at {start}/{len(sl)} "
                  f"({len(w.registry.slots)} slots replayed)", flush=True)

        def _tick(n, total):
            if n % 500 == 0:
                print(f"  [{ch}] {n}/{total} ({time.time()-t0:.0f}s)", flush=True)

        with rec_store.channel(ch) as handle:
            out = ingest_channel(sl, w, handle, gord, start=start,
                                 on_progress=_tick)
        # stage1_stats is the ONLY view into the stage-1 loop hitting its caps:
        # budget_hit counts messages that burned all MAX_LOOPS turns without
        # reaching write_fields, no_fields the ones that therefore commit with no
        # decision fields at all -- a quality loss the record count cannot show,
        # since the row is still written.
        print(f"  [{ch}] DONE {len(sl)} agent={w.agent_stats} "
              f"stage1={getattr(w, 'stage1_stats', None)} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return ch, out

    suffix = f".c{args.channels_limit}" if args.channels_limit else ""
    if args.limit:
        suffix += f".n{args.limit}"
    if args.shard:
        suffix += ".s" + args.shard.split("/")[0]
    if args.channels_file:
        # The allowlist has to name the product, or several per-channel runs
        # sharing a --variant silently overwrite each other's records.
        suffix += "." + os.path.splitext(
            os.path.basename(args.channels_file))[0]
    os.makedirs(STORES, exist_ok=True)
    out_path = os.path.join(
        STORES, f"{args.variant}_records_{args.domain}{suffix}.jsonl")

    # Flush per channel as it finishes. Accumulating everything and writing once
    # at the end meant a run killed before the last channel produced NOTHING: a
    # Technology run with 6/7 channels done and the 7th at 5000/5021 was stopped
    # 21 messages short and left a zero-byte file behind 8.5h of paid calls.
    # as_completed rather than ex.map (which yields in submit order and would
    # hold every finished channel hostage to the slowest), so the file on disk is
    # always a valid prefix.
    recs = []
    with open(out_path, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_channel, c): c for c in chans}
            for fu in as_completed(futs):
                ch, out = fu.result()
                for r in out:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                recs.extend(out)
                print(f"  [{ch}] FLUSHED {len(out)} -> {out_path} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"WROTE {len(recs)} records -> {out_path} "
          f"({time.time()-t0:.0f}s total)", flush=True)

    # Re-export from the table so the product holds EVERY record for this
    # variant. The loop above opens with "w", which truncates -- correct for a
    # fresh run and silently destructive for a resumed one.
    full = []
    try:
        full = list(rec_store.iter_records())
        if len(full) > len(recs):
            with open(out_path, "w", encoding="utf-8") as f:
                for r in full:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"EXPORTED {len(full)} records from the table -> {out_path} "
                  f"(this run handled {len(recs)})", flush=True)
    except Exception as e:      # noqa: BLE001
        # Never fail the run over the export; the data is already committed.
        print(f"WARN: export failed ({type(e).__name__}: {e}); "
              f"{out_path} holds only this run's {len(recs)} records", flush=True)

    _link_scorecard(args, msgs, by_ch, chans, full or recs, bool(full))

    # ---- the whole-domain reconciliation, once this domain is whole -------- #
    # The fused rows are already in the table -- one per message that extended a
    # chain, committed with that message. What runs here is finalize's pass over
    # every chain, which on a healthy run finds nothing to do and says so; that
    # is the independent check that both renderings agree.
    #
    # The gate is a COST decision, not a safety one: the pass loads the whole
    # table, and running it before the corpus is complete would only be repeated.
    # It is safe at any point -- a fused row is keyed on its chain's root, so a
    # later pass updates it rather than leaving a stale twin.
    if args.no_finalize:
        print(f"skipping the fused-row reconciliation (--no-finalize); the rows "
              f"are already committed. bench/finalize.py --domain "
              f"{args.domain} --variant {args.variant} checks them", flush=True)
        return 0
    if not full:
        # The export failed, so what the table holds is unknown here.
        print("cannot tell whether the corpus is complete (table read failed "
              "above) -- skipping the reconciliation; run bench/finalize.py "
              "once it is", flush=True)
        return 0
    missing = {m["msg_node"] for m in msgs} - {r["msg_node"] for r in full}
    if missing:
        print(f"{len(missing)} of {len(msgs)} corpus messages are not committed "
              f"yet (e.g. {sorted(missing)[0]}) -- leaving the reconciliation to "
              f"the run that commits the last one", flush=True)
        return 0
    print(f"corpus complete ({len(msgs)} messages committed) -- reconciling the "
          f"fused rows, no API call", flush=True)
    rc = finalize_domain(args.domain, args.variant)
    if rc:
        # The messages are committed and exported either way; say which half
        # failed rather than let a non-zero exit read as "the ingest broke".
        print("INGEST OK, FINALIZE FAILED -- the raw rows are complete and "
              f"exported; re-run bench/finalize.py --domain {args.domain} "
              f"--variant {args.variant}", flush=True)
    return rc


def _link_scorecard(args, msgs, by_ch, chans, scored, from_table: bool) -> None:
    """Offline check of the supersedes edges against the corpus's gold links.

    Scored over the WHOLE variant read back from the table, not over what this
    process handled: on a resumed run the two differ by everything already
    committed, so scoring the latter under-reports edges in proportion to how
    little was left to do. `gold_links` is unaffected -- it comes from the corpus,
    not the run.
    """
    in_scope = {m["msg_node"] for c in chans for m in by_ch[c]}
    gold_links = {}
    for m in msgs:
        if m["msg_node"] not in in_scope:
            continue
        dl = m.get("decision_label")
        if isinstance(dl, dict) and dl.get("linked_to_msg"):
            gold_links[m["msg_node"]] = dl["linked_to_msg"]
    sup = {r["msg_node"]: r["supersedes"] for r in scored if r.get("supersedes")}
    on_gold = [(m, t) for m, t in sup.items() if m in gold_links]
    exact = sum(1 for m, t in on_gold if gold_links[m] == t)
    n_high = sum(1 for r in scored
                 if r.get("supersedes") and r.get("link_conf") == "high")
    n_low = sum(1 for r in scored
                if r.get("supersedes") and r.get("link_conf") == "low")
    print(f"\n===== LINK SCORECARD ({args.domain}/{args.variant}) =====")
    print(f"scored over {len(scored)} records "
          f"({'whole variant from the table' if from_table else 'THIS RUN ONLY'})")
    print(f"gold changed: {len(gold_links)} | edges produced: {len(sup)} "
          f"(high={n_high} low={n_low}) | on-gold: {len(on_gold)} | "
          f"exact: {exact} ({exact/max(len(on_gold),1):.2f} of on-gold, "
          f"{exact/max(len(gold_links),1):.2f} of all gold)")


if __name__ == "__main__":
    sys.exit(main())
