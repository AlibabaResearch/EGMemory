#!/usr/bin/env python3
"""Score a results directory, and compare two arms on the same questions.

    python3 bench/summarize.py                          # the default arm
    python3 bench/summarize.py --arm my_ablation
    python3 bench/summarize.py --arm my_ablation --vs release

No API calls, no database. Reads `<results>/<arm>/<domain>__<qtype>.jsonl` --
whatever `bench/run_qa.py` wrote -- and prints per-question-type and per-domain
accuracy with Wilson intervals. The headline number is the SUM across domains,
not the mean.

`--vs` matches rows by question id and runs an exact McNemar test on the
disagreements only; read the pooled McNemar p rather than per-qtype. A row
carrying an `error` key counts WRONG rather than being dropped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.judge import parse_judgment                     # noqa: E402
from bench.manifest import meta_path_for                  # noqa: E402
from bench.stats import (mcnemar_counts, mcnemar_exact,    # noqa: E402
                                wilson_ci)

QTYPES = ("knowledge_update", "multi_hop", "temporal", "term_ambiguity",
          "user_implicit", "abstention")

# Published read-side reference on the release store; not a score produced here
# by the open-source write side.
REFERENCE = {"knowledge_update": (44, 107), "multi_hop": (137, 182),
             "temporal": (130, 162), "term_ambiguity": (48, 106),
             "user_implicit": (33, 49), "abstention": (110, 139)}
REFERENCE_TOTAL = (502, 745)
REFERENCE_NAME = "published reference"


def verdict(row: dict):
    """True / False for one result row.

    `correct` when present; otherwise the judge's reply is re-parsed, which is
    how result files from before that field existed still score. `None` from
    `parse_judgment` means the judge answered neither -- counted wrong, since an
    unparseable verdict is not evidence of a right answer.
    """
    if "error" in row:
        return False
    if "correct" in row:
        return bool(row["correct"])
    return parse_judgment(row.get("judge_answer") or "") is True


def load_arm(results_dir: str, arm: str):
    """-> (rows_by_key, fingerprints) where key = (domain, qtype).

    rows_by_key values are [(question_id, verdict), ...] in file order; the id is
    what `--vs` pairs on, and file order is what makes two arms diffable when a
    question has no id.
    """
    rows = {}
    fps = {}
    pat = os.path.join(results_dir, arm, "*__*.jsonl")
    for path in sorted(glob.glob(pat)):
        stem = os.path.basename(path)[: -len(".jsonl")]
        if "__" not in stem:
            continue
        domain, qtype = stem.split("__", 1)
        out = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out.append((r.get("id") or f"#{i}", verdict(r)))
        rows[(domain, qtype)] = out
        fps[(domain, qtype)] = _fingerprint(path)
    return rows, fps


def _fingerprint(jsonl_path: str) -> dict:
    """The few manifest fields that decide whether results are comparable."""
    mp = meta_path_for(jsonl_path)
    if not os.path.exists(mp):
        return {}
    try:
        m = json.load(open(mp, encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        return {}
    e = m.get("extra") or {}
    mo = m.get("models") or {}
    return {"qa_prompt": e.get("qa_prompt_sha"),
            "judge_prompt": e.get("judge_prompt_sha"),
            "variant": e.get("variant") or e.get("arm"),
            "n_max": e.get("n_max") or (m.get("retrieval") or {}).get("top_k"),
            "agent": mo.get("agent_model"), "judge": mo.get("judge_model"),
            "emb": mo.get("emb_model"),
            "answer_parser": e.get("answer_parser", "unrecorded"),
            "version_status_renderer": e.get("version_status_renderer", "unrecorded"),
            "decision_renderer": e.get("decision_renderer", "unrecorded")}


def _warn_mixed(arm: str, fps: dict) -> None:
    """One arm must be one configuration.

    A results directory is append-only in practice -- `--skip-existing` fills
    gaps, and a later rerun overwrites single files -- so an arm can end up
    holding two configurations with nothing on screen to say so. Mixing two
    classify prompts across one build is exactly how that happens. The same
    failure on the read side would show up here and only here.
    """
    seen = defaultdict(set)
    for fp in fps.values():
        for k, v in fp.items():
            if v is not None:
                seen[k].add(json.dumps(v, sort_keys=True))
    mixed = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    if mixed:
        print(f"  [{arm}] WARNING: mixes configurations -- "
              + "; ".join(f"{k}: {', '.join(v)}" for k, v in mixed.items()))
    else:
        one = {k: sorted(v)[0].strip('"') for k, v in seen.items()}
        if one:
            print(f"  [{arm}] config: " + "  ".join(f"{k}={v}" for k, v in
                                                   sorted(one.items())))


def _tally(rows, key_fn):
    agg = defaultdict(lambda: [0, 0])
    for key, verdicts in rows.items():
        bucket = key_fn(key)
        agg[bucket][0] += sum(1 for _, ok in verdicts if ok)
        agg[bucket][1] += len(verdicts)
    return agg


def _print_table(title: str, agg, order=None, reference=None) -> tuple:
    keys = list(order) if order else sorted(agg)
    head = f"  {title:<18} {'n':>5} {'correct':>8} {'acc':>7}   {'95% CI':<16}"
    if reference:
        head += f" {'reference':>13} {'Δ':>6}"
    print(head)
    tc = tt = 0
    for k in keys:
        c, t = agg.get(k, [0, 0])
        if not t:
            continue
        tc += c
        tt += t
        lo, hi = wilson_ci(c, t)
        line = (f"  {k:<18} {t:5d} {c:8d} {100.0*c/t:6.1f}%   "
                f"[{100*lo:4.1f}, {100*hi:4.1f}]     ")
        if reference:
            rc, rt = reference.get(k, (0, 0))
            line += (f" {rc:4d}/{rt:<4d} {100.0*rc/rt:5.1f}%" if rt
                     else f" {'-':>13}")
            line += f" {c - rc:+6d}" if rt == t else f" {'n/a':>6}"
        print(line)
    return tc, tt


def _print_total(tc: int, tt: int, reference=None) -> None:
    if not tt:
        print("  (no rows)")
        return
    lo, hi = wilson_ci(tc, tt)
    print(f"  {'TOTAL':<18} {tt:5d} {tc:8d} {100.0*tc/tt:6.1f}%   "
          f"[{100*lo:4.1f}, {100*hi:4.1f}]")
    if not reference:
        return
    rc, rt = reference
    if tt != rt:
        print(f"  (reference is {rc}/{rt}; this run covers {tt} questions, so "
              f"the two totals are not comparable)")
        return
    d = tc - rc
    print(f"  reference          {rt:5d} {rc:8d} {100.0*rc/rt:6.1f}%   "
          f"Δ = {d:+d} questions")
    print(f"  reference surface: {REFERENCE_NAME}; a total-only delta cannot tell "
          "a gain from a regression.")
    print("  a significance test needs per-question results on the SAME questions, "
          "checked against the data tables and the config fingerprint.")


def _paired(a_rows, b_rows, a_name: str, b_name: str) -> None:
    """Exact McNemar over every question both arms answered."""
    print(f"\npaired: {a_name} vs {b_name}")
    on, off, per_q = [], [], defaultdict(lambda: [[], []])
    only_a = only_b = 0
    for key in sorted(set(a_rows) | set(b_rows)):
        av, bv = a_rows.get(key), b_rows.get(key)
        if av is None:
            only_b += 1
            continue
        if bv is None:
            only_a += 1
            continue
        bmap = dict(bv)
        for qid, ok in av:
            if qid not in bmap:
                continue
            on.append(ok)
            off.append(bmap[qid])
            per_q[key[1]][0].append(ok)
            per_q[key[1]][1].append(bmap[qid])
    if only_a or only_b:
        print(f"  {only_a} file(s) only in {a_name}, {only_b} only in "
              f"{b_name}; those are excluded -- a paired test needs both sides.")
    if not on:
        print("  no questions in common; nothing to pair.")
        return
    print(f"  {'qtype':<18} {'paired':>6} {a_name[:9]:>10} {b_name[:9]:>10} "
          f"{'helped':>7} {'hurt':>5} {'p':>8}")
    for q in QTYPES:
        if q not in per_q:
            continue
        o, f = per_q[q]
        b, c = mcnemar_counts(o, f)
        print(f"  {q:<18} {len(o):6d} {sum(o):10d} {sum(f):10d} "
              f"{c:7d} {b:5d} {mcnemar_exact(b, c):8.3f}")
    b, c = mcnemar_counts(on, off)
    p = mcnemar_exact(b, c)
    print(f"  {'POOLED':<18} {len(on):6d} {sum(on):10d} {sum(off):10d} "
          f"{c:7d} {b:5d} {p:8.4f}")
    print(f"  {a_name} helped on {c} questions and hurt on {b}; the "
          f"{len(on)-b-c} questions both arms agree on carry no information "
          f"about the difference, which is the point of pairing.")
    if p >= 0.05:
        print(f"  p={p:.4f}: no significant difference on the current paired sample; "
              "this does not prove the configs are equivalent, nor attribute the "
              "difference to chance.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score a results arm; optionally pair it against another.")
    ap.add_argument("--arm", default="release",
                    help="subdirectory of --results-dir to score")
    ap.add_argument("--vs", default="",
                    help="second arm; adds an exact McNemar test on the "
                         "questions both arms answered. USE THIS rather than "
                         "comparing two accuracies by eye.")
    ap.add_argument("--results-dir", default=os.path.join(_ROOT, "results"))
    ap.add_argument("--no-reference", action="store_true",
                    help="hide the published-reference column (it is meaningless "
                         "for a corpus other than the four synthetic domains)")
    args = ap.parse_args()

    rows, fps = load_arm(args.results_dir, args.arm)
    if not rows:
        print(f"no result files under "
              f"{os.path.join(args.results_dir, args.arm)}")
        return 1
    domains = sorted({d for d, _ in rows})
    print(f"arm {args.arm}: {len(rows)} file(s), {len(domains)} domain(s): "
          f"{', '.join(domains)}")
    _warn_mixed(args.arm, fps)

    ref = None if args.no_reference else REFERENCE
    order = [q for q in QTYPES if any(q == qt for _, qt in rows)]
    # Question types outside the six the synthetic domains ship (another corpus,
    # or a renamed split) are appended rather than dropped.
    order += sorted({qt for _, qt in rows} - set(QTYPES))
    print()
    tc, tt = _print_table("qtype", _tally(rows, lambda k: k[1]), order=order,
                          reference=ref)
    _print_total(tc, tt, None if args.no_reference else REFERENCE_TOTAL)

    if len(domains) > 1:
        print("\n  per domain -- a diagnostic only. The domains differ in "
              "revision density and question count, so these accuracies are "
              "not comparable to each other and their mean is not a result.")
        _print_table("domain", _tally(rows, lambda k: k[0]), order=domains)

    if args.vs:
        b_rows, b_fps = load_arm(args.results_dir, args.vs)
        if not b_rows:
            print(f"\nno result files under "
                  f"{os.path.join(args.results_dir, args.vs)}; skipping the "
                  f"paired test")
            return 1
        print()
        _warn_mixed(args.vs, b_fps)
        _paired(rows, b_rows, args.arm, args.vs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
