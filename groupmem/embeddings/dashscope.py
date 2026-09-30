"""The DashScope SDK embedding provider, with an on-disk cache.

``DashScopeNativeEmbedder`` -- the SDK's TextEmbedding endpoint. THIS IS WHAT EVERY
SHIPPED NUMBER WAS PRODUCED WITH, which is why it is not folded into the
OpenAI-compatible provider: that one is a protocol, this one is the exact client the
stores came from, down to its own timeout and retry behaviour.

The OpenAI-compatible embedder used to live here as ``DashScopeEmbedder``; it is now
``openai_compat.OpenAICompatEmbedder`` and takes ``api_key``/``base_url``, so the
backend name ``dashscope`` and the backend name ``openai`` both resolve to it. The
offline hashing backend is ``hashing.py``, the selection is ``__init__.get_embedder``,
and ``base.py`` states what an embedder must guarantee.

The cache hashes by content, so a re-run over the same corpus is free.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ..configs.defaults import EMBEDDING_MODEL
from .base import Embedder, l2_normalize

# DashScope's TextEmbedding.call has no server-side timeout: a request can
# establish a socket and then hang forever with the server never replying
# (observed mid-ingest, 0% CPU, established TCP, no progress). Unlike ds_chat,
# _embed_batch had no wall-clock guard, so one hung socket froze the whole run.
# Run each attempt in a worker and abandon it after EMB_HARD_TIMEOUT, then retry
# -- same pattern ds_chat uses for the generation endpoint.
_EMB_POOL = ThreadPoolExecutor(max_workers=32)
_EMB_HARD_TIMEOUT = 60.0        # seconds per attempt before the socket is abandoned

# In-flight embedding batches per embed() call. Kept well under the pool's 32
# because the same account's chat endpoint returns 429 under bursts; the serial
# path (==1) is still reachable and is what the comment in embed() describes.
EMB_CONCURRENCY = 8

# The width of the default model's vectors, and therefore of every pgvector column
# in every stored table. It is stated here rather than read from the first response
# because the pgvector table is CREATED before anything is embedded. Both
# text-embedding-v3 and -v4 return 1024; a model with another width must be given a
# matching `dim` (and a fresh table -- the column cannot be re-widened in place).
NATIVE_DIM = 1024

# THE EMBEDDING MODEL DEFAULTS TO text-embedding-v4 AND SAYS SO WHEN IT DOES
# NOT. Every stored vector -- the pgvector column of a derived read table, the
# write-side `_cvec` map, and the disk cache -- was produced by that model, and a
# vector from a different one is not comparable: cosine still returns a number,
# so a mismatch does not fail, it silently degrades retrieval. A caller that
# points EMBEDDING_MODEL at a different model, or forgets to set it, can get a
# store that looks fine and retrieves worse -- silently, which is the part that
# costs time.
#
# So the env var stays honoured (EMBEDDING_MODEL, imported from config at the
# top of this file) but the DEFAULT is the reference model and an override prints
# a line naming it. Overriding it is a re-embed of everything, deliberately.


class DashScopeNativeEmbedder(Embedder):
    """Qwen embedding via the native dashscope SDK, with disk cache.

    Env:
      DASHSCOPE_API_KEY   (required)

    The model defaults to text-embedding-v4 (see EMBEDDING_MODEL above).
    """

    def __init__(self, model: Optional[str] = None, cache_dir: Optional[str] = None,
                 batch_size: int = 10, dim: int = NATIVE_DIM) -> None:
        import dashscope  # lazy
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        dashscope.api_key = api_key
        self._ds = dashscope
        self.model = model or EMBEDDING_MODEL
        self.dim = dim
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        # `_cache` is a text-hash -> vector memo, backed by an append-only jsonl
        # loaded whole at startup. Cost is O(file) RAM (a 5.2 GB cache is ~8 GB
        # resident per process, which OOMs parallel ingests), and the alternative
        # -- a sqlite table of float32 BLOBs, O(1) startup and shareable across
        # processes -- was implemented and then never used: every cache produced
        # for these runs is emb_text-embedding-v4.jsonl. It is gone rather than
        # left as a second code path, because the two are NOT interchangeable
        # within one store: the float32 round-trip differs from the JSON float64
        # by ~1e-7, so a corpus half-filled from each has vectors that disagree
        # below the noise floor of cosine, which is exactly the kind of defect
        # that never announces itself.
        self._cache = {}
        self._cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_path = os.path.join(cache_dir, f"emb_{self.model}.jsonl")
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                            self._cache[row["h"]] = row["v"]
                        except Exception:
                            pass

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256((text or "").encode()).hexdigest()

    def _get(self, h):
        """Cached vector for hash `h`, or None (the whole file is preloaded)."""
        return self._cache.get(h)

    def _flush(self, new_rows):
        if not new_rows:
            return
        if not self._cache_path:
            return
        with open(self._cache_path, "a", encoding="utf-8") as f:
            for h, v in new_rows:
                f.write(json.dumps({"h": h, "v": v}) + "\n")

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        last = None
        for attempt in range(5):
            try:
                fut = _EMB_POOL.submit(
                    self._ds.TextEmbedding.call, model=self.model, input=batch)
                resp = fut.result(timeout=_EMB_HARD_TIMEOUT)
                if getattr(resp, "status_code", 200) != 200:
                    raise RuntimeError(f"status={resp.status_code} {getattr(resp,'message','')}")
                embs = resp.output["embeddings"]
                embs = sorted(embs, key=lambda e: e.get("text_index", 0))
                return [l2_normalize(list(e["embedding"])) for e in embs]
            except Exception as e:  # noqa: BLE001 (incl. TimeoutError)
                last = e
                # A retry used to be completely silent, so a read-side query that
                # failed four times before succeeding showed up only as a ~30 s
                # tool call (3+6+9+12 s of backoff) with no trace of why. The
                # retry itself is right; hiding it is not.
                print(f"    [emb retry {attempt + 1}/5, sleeping "
                      f"{3 * (attempt + 1)}s] {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"dashscope embedding failed: {last}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed with cache, issuing the uncached batches CONCURRENTLY.

        A strictly serial batch loop costs one round trip per 10 texts, so a
        whole-domain index build is thousands of sequential network waits with the
        process otherwise idle. Concurrency here is safe in a way that chat
        concurrency is not: embedding requests are short, uniform, and stateless.

        EMB_CONCURRENCY in-flight batches, deliberately well under the pool's
        32: the chat side of this same account can return 429 "Request rate
        increased too quickly", so this stays modest rather than bursting.

        Cache writes stay on THIS thread (after the gather), because the cache is
        an append to one shared file -- letting workers flush concurrently would
        interleave rows.
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        pending_idx, pending_txt = [], []
        for i, t in enumerate(texts):
            h = self._key(t)
            cached = self._get(h)
            if cached is not None:
                results[i] = cached
            else:
                pending_idx.append(i)
                pending_txt.append(t)

        chunks = []
        for start in range(0, len(pending_txt), self.batch_size):
            chunks.append((pending_idx[start:start + self.batch_size],
                           pending_txt[start:start + self.batch_size]))
        if not chunks:
            return [r if r is not None else [] for r in results]

        if EMB_CONCURRENCY == 1 or len(chunks) == 1:
            done = [(idx, self._embed_batch([t if t.strip() else " " for t in txt]))
                    for idx, txt in chunks]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=EMB_CONCURRENCY) as ex:
                futs = [(idx, ex.submit(
                            self._embed_batch,
                            [t if t.strip() else " " for t in txt]))
                        for idx, txt in chunks]
                done = [(idx, f.result()) for idx, f in futs]

        new_rows = []
        for chunk_idx, vecs in done:
            for j, v in zip(chunk_idx, vecs):
                results[j] = v
                h = self._key(texts[j])
                self._cache[h] = v
                new_rows.append((h, v))
        self._flush(new_rows)
        return [r if r is not None else [] for r in results]
