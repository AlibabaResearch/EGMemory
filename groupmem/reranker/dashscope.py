"""DashScope text-rerank: the SELECTION backend the default benchmark uses.

The method lives in ``base.py`` -- why a cross-encoder earned this slot, where it
sits in the read pipeline, what it is scored against, and why ``stats()`` must be
checked before a number is reported. THIS FILE IS ONE HTTP CALL. If you are
swapping in another cross-encoder, subclass ``Reranker`` and implement ``rank``;
nothing else here is method.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import requests

from ..configs.defaults import RERANK_MODEL
from .base import Reranker

# The model id stays configurable, with the default, in
# configs/defaults.py, because which cross-encoder does the selecting is exactly
# the kind of substitution a reader may need to make. DEPTH and DOC_CHARS are
# method, not transport, and live in base.py.
MODEL = RERANK_MODEL

_URL = ("https://dashscope.aliyuncs.com/api/v1/services/rerank/"
        "text-rerank/text-rerank")
_TIMEOUT = 60.0
_RETRIES = 3


class DashScopeReranker(Reranker):
    """Env: DASHSCOPE_API_KEY. Fails open (returns None) on every error path."""

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__()
        self.model = model or MODEL

    def rank(self, query: str, docs: List[str],
             top_n: int) -> Optional[List[int]]:
        key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not key:
            self._bump("no_key")
            return None
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": docs},
            # return_documents=False keeps the response small: only the permutation
            # is needed, the texts are already in hand.
            "parameters": {"top_n": top_n, "return_documents": False},
        }
        headers = {"Authorization": f"Bearer {key}",
                   "Content-Type": "application/json"}
        for attempt in range(_RETRIES):
            try:
                r = requests.post(_URL, json=payload, headers=headers,
                                  timeout=_TIMEOUT)
                if r.status_code == 200:
                    body = r.json()
                    results = (body.get("output") or {}).get("results") or []
                    self._bump("calls")
                    self._bump("tokens", int((body.get("usage") or {}).get(
                        "total_tokens", 0) or 0))
                    return [int(x["index"]) for x in results
                            if 0 <= int(x.get("index", -1)) < len(docs)]
                # 429/5xx are worth another try; a 4xx config error is not.
                if r.status_code not in (429, 500, 502, 503, 504):
                    self._bump(f"http_{r.status_code}")
                    return None
            except Exception:  # noqa: BLE001 -- fail open, see module docstring
                pass
            time.sleep(1.5 * (attempt + 1))
        self._bump("fail")
        return None
