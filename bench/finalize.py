#!/usr/bin/env python3
"""Reconcile the fused decision rows of gm_<domain>__<variant>.

    python3 bench/finalize.py --domain Technology

The ingest already writes these rows per message; this script is the
whole-domain repair / re-render pass, run by hand or at the end of an ingest
that completed a corpus. ZERO LLM CALLS AND ZERO API CALLS OF ANY KIND: it is
rules plus arithmetic, needs no key, and cannot fail on a network.

Idempotent: a fused row's id is its chain's root, so a chain that grew is
updated in place rather than duplicated; running the pass twice writes nothing
the second time, and running it on a finished domain reports "no new chains to
fuse". Fused-row vectors are the mean of the hop vectors the writer already
committed, so no embedding call is made.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from groupmem.vector_stores import pgvector as pg_store
from groupmem.memory.fusion import (apply_fusion, fused_ord,         # noqa: E402
                                 mean_unit)


def finalize_domain(domain: str, variant: str) -> int:
    """Reconcile the fused rows of one domain. 0 on success, 1 on refusal.

    Callable rather than inlined in `main` so `ingest.py` can run the step at the
    end of a complete run: same code, same convergence, no subprocess and no
    second place that knows how a fused row is built.
    """
    t = pg_store.table(domain, variant)
    if not pg_store.table_exists(domain, variant):
        print(f"{t} does not exist -- run bench/run_ingest.py --domain "
              f"{domain} --variant {variant} first")
        return 1

    t0 = time.time()
    store, idx = pg_store.load_light_store(domain, variant=variant)
    before = set(store.order)
    print(f"[{domain}] loaded {len(store.order)} nodes, "
          f"{len(store.edges)} edges from {t} ({time.time()-t0:.0f}s)")

    store, idx, touched = apply_fusion(store, idx)
    if not touched:
        print(f"[{domain}] every chain's fused row is up to date; {t} is final")
        return 0
    fresh = [m for m in touched if m not in before]

    # The hops' vectors are already in the table; a fused row's vector is their
    # mean. Missing one is a hard stop rather than a silently weaker vector: a
    # chain whose hop has no embedding means the ingest committed a row it should
    # not have, and averaging what is left would hide that.
    hops = sorted({s for m in touched for s in store.memories[m].sources})
    hopvecs = pg_store.fetch_vectors(hops, domain, variant)
    short = [m for m in touched
             if any(s not in hopvecs for s in store.memories[m].sources)]
    if short:
        print(f"[{domain}] {len(short)} chains have a hop with no vector "
              f"in {t} (first: {short[0]}) -- refusing to average around it",
              file=sys.stderr)
        return 1
    vecs = [mean_unit([hopvecs[s] for s in store.memories[m].sources])
            for m in touched]
    print(f"[{domain}] pooled {len(touched)} fused vectors from "
          f"{len(hopvecs)} hop vectors, no API call ({time.time()-t0:.0f}s)")

    # `ord` is a function of the chain ROOT's corpus position, so this pass lands
    # on the value the per-message writer computed for the same chain and a
    # rewritten row does not move. A root with no ord is a hand-edited row.
    roots = [store.memories[m].sources[0] for m in touched]
    ords = pg_store.fetch_ords(roots, domain, variant)
    lost = [r for r in roots if r not in ords]
    if lost:
        print(f"[{domain}] {len(lost)} chain roots have no ord in {t} "
              f"(first: {lost[0]}) -- refusing to place their fused rows",
              file=sys.stderr)
        return 1
    rows = [pg_store.row_tuple(
                pg_store.fusion_row(store.memories[m], fused_ord(ords[root])),
                v)
            for m, root, v in zip(touched, roots, vecs)]
    pg_store.insert_rows(rows, domain, variant)
    print(f"[{domain}] WROTE {len(rows)} fused rows -> {t} "
          f"({len(fresh)} new, {len(rows)-len(fresh)} re-rendered) "
          f"{time.time()-t0:.0f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="finalize: reconcile the fused rows of gm_<domain>__<variant>.")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--variant", default="release")
    args = ap.parse_args()
    return finalize_domain(args.domain, args.variant)


if __name__ == "__main__":
    sys.exit(main())
