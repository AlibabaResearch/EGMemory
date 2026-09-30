"""Embeddings over ``POST {base_url}/embeddings`` -- the OpenAI protocol.

One class serving every provider that speaks this protocol (OpenAI, vLLM, Ollama,
TEI, DashScope's compatible mode); DashScope is one caller of it.

WHAT PRODUCED THE SHIPPED NUMBERS IS NOT THIS -- ``DashScopeNativeEmbedder`` (the
SDK, ``dashscope.py``) is. A vector from a different model is not comparable to a
stored one, so switching embedder means re-embedding the store, not pointing at a
new endpoint. The ``openai`` package is imported inside ``__init__`` so installing
without the extra costs nothing until you ask for this backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import List, Optional

from ..configs.defaults import EMBEDDING_MODEL
from .base import Embedder, l2_normalize

# DashScope's OpenAI-compatible route. Kept as the fallback default because the
# `dashscope` backend name resolves to this class, and a caller who names that
# backend has said which host they mean.
DASHSCOPE_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class OpenAICompatEmbedder(Embedder):
    """Any endpoint that speaks ``POST {base_url}/embeddings``, with a disk cache.

    Resolution order for both credentials is EXPLICIT ARGUMENT, then the vendor's
    own variable, then the OpenAI one::

        api_key  : api_key=...  | DASHSCOPE_API_KEY  | OPENAI_API_KEY
        base_url : base_url=... | DASHSCOPE_BASE_URL | OPENAI_BASE_URL | default

    DashScope comes first in that chain for one reason: every store in this repo was
    ingested with DASHSCOPE_API_KEY exported, and a machine that has both set should
    keep reaching the endpoint its stores came from rather than silently switch.

    ``dim`` DEFAULTS TO 0, i.e. "not stated", because this class serves every
    provider and their widths differ (1024 for text-embedding-v3/v4, 1536 for
    OpenAI's text-embedding-3-small, whatever your vLLM is serving). 0 is honest and
    ``base.py``'s contract handles it: an inmemory index never needs the number, and
    the pgvector record store refuses at construction with a message naming what to
    pass, instead of creating a ``vector(1024)`` column your model cannot fill. The
    ``dashscope`` backend name fills in 1024 because for that host it IS known.

    A wrong width fails at the first commit; a *plausible* wrong model (right width,
    different vectors) does not fail at all, it retrieves worse.
    """

    def __init__(self, model: Optional[str] = None, cache_dir: Optional[str] = None,
                 batch_size: int = 10, dim: int = 0,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None) -> None:
        from openai import OpenAI  # imported lazily; see the module docstring
        api_key = (api_key or os.environ.get("DASHSCOPE_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            raise RuntimeError(
                "no embedding API key: pass api_key=..., or export "
                "DASHSCOPE_API_KEY / OPENAI_API_KEY. A local server that wants no "
                "key still needs a placeholder here -- it ignores the header.")
        base_url = (base_url or os.environ.get("DASHSCOPE_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL") or DASHSCOPE_COMPAT_URL)
        # max_retries=0: the retry policy is _embed_batch's, which PRINTS. The SDK's
        # own silent retries were how a five-attempt embed looked like one slow call.
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.base_url = base_url
        self.model = model or EMBEDDING_MODEL
        self.dim = dim
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._cache = {}
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
        else:
            self._cache_path = None

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256((text or "").encode()).hexdigest()

    def _flush(self, new_rows):
        if not self._cache_path or not new_rows:
            return
        with open(self._cache_path, "a", encoding="utf-8") as f:
            for h, v in new_rows:
                f.write(json.dumps({"h": h, "v": v}) + "\n")

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        last = None
        for attempt in range(5):
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                return [l2_normalize(list(d.embedding)) for d in resp.data]
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"    [emb retry {attempt + 1}/5, sleeping "
                      f"{3 * (attempt + 1)}s] {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"embedding failed: {last}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        pending_idx, pending_txt = [], []
        for i, t in enumerate(texts):
            h = self._key(t)
            if h in self._cache:
                results[i] = self._cache[h]
            else:
                pending_idx.append(i)
                pending_txt.append(t)

        new_rows = []
        for start in range(0, len(pending_txt), self.batch_size):
            chunk_txt = pending_txt[start:start + self.batch_size]
            chunk_idx = pending_idx[start:start + self.batch_size]
            vecs = self._embed_batch([t if t.strip() else " " for t in chunk_txt])
            for j, v in zip(chunk_idx, vecs):
                results[j] = v
                h = self._key(texts[j])
                self._cache[h] = v
                new_rows.append((h, v))
            self._flush(new_rows)
            new_rows = []
        return [r if r is not None else [] for r in results]
