"""The ONE lexical layer: what a token is, how BM25 scores, how ranks fuse.

Both halves of the system share it, so a slot is indexed and searched with the
same vocabulary.

WHAT IS HERE, AND WHO USES IT

  tokenize                 both sides. Plain alphanumeric, NOT stemming.
  bm25 / ranked_ids / fuse the WRITE side's channel combination, over registry
                           slots (`DecisionRegistry.retrieve`).
  LexicalIndex             the READ side's index, over the whole store, on
                           ``rank_bm25.BM25Okapi``.

Two BM25 implementations in one file is deliberate: the write side scores one
document at a time against a doc-frequency table it maintains incrementally, while
the read side scores a query against a corpus fixed at load. They are the same
formula (Okapi, identical IDF) over different access patterns, and keeping them
adjacent is what makes "the same formula" checkable.

`fuse` is weighted RRF at w_lex=2.0, k0=60, which is the read side's fusion too;
RRF consumes only RANKS, so it is immune to channel scale and needs no
normalization. `ranked_ids` drops scores of exactly 0.0, so a slot with no shared
token and no vector contributes to neither channel.

The read-side index is cheap to build and derives purely from the store, so it is
rebuilt on load rather than pickled.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from ..schema import MemoryStore

# Weight on the LEXICAL channel, mirroring the read side's BM25_RRF_WEIGHT. One
# value serves the write side rather than a per-layer parameter: the layers'
# optima are close enough that splitting them buys nothing.
RRF_W_LEX = 2.0

# k0 = 60, the read side's value (FusionRetriever._rrf). Identical so the layers
# differ in what they retrieve, not in how they fuse ranks.
RRF_K0 = 60

BM25_K1 = 1.5
BM25_B = 0.75

_TOK_PLAIN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric+underscore tokens. THE tokenizer, both sides.

    The registry retriever and its doc-frequency table have to tokenize through
    one function or BM25 silently scores against a different vocabulary than it
    indexed; so does the read-side index against the queries it is asked.
    """
    return _TOK_PLAIN.findall((text or "").lower())


def bm25(q_tokens: Sequence[str], d_tokens: Sequence[str],
         df: Dict[str, int], ndocs: int, avgdl: float) -> float:
    """Okapi BM25 of a token query against ONE tokenized document -- the write
    side's whole lexical channel, over registry slots.

    The IDF is the standard Okapi form, log(1 + (N - df + 0.5)/(df + 0.5)). That
    is the LUCENE variant, and it is deliberately not the one
    `rank_bm25.BM25Okapi` uses on the read side: BM25Okapi computes
    log(N - df + 0.5) - log(df + 0.5), which goes NEGATIVE for a term in more
    than half the documents and is then replaced by `epsilon * average_idf`. The
    two corpora are different (registry slots here, the whole store there) and no
    score ever crosses between them, so this is two appropriate choices rather
    than a drift -- but the `1 +` is why a stopword-ish term cannot subtract from
    a slot's score here.

    It ranks ahead of both an IDF-weighted coverage share and plain token
    overlap: term saturation and document-length normalization are what those two
    lack.
    """
    if not q_tokens or not d_tokens:
        return 0.0
    tf: Dict[str, int] = {}
    for t in d_tokens:
        tf[t] = tf.get(t, 0) + 1
    norm = BM25_K1 * (1 - BM25_B + BM25_B * len(d_tokens) / max(avgdl, 1e-9))
    score = 0.0
    for t in set(q_tokens):
        f = tf.get(t, 0)
        if not f:
            continue
        n_t = df.get(t, 0)
        idf = math.log(1 + (ndocs - n_t + 0.5) / (n_t + 0.5))
        score += idf * (f * (BM25_K1 + 1)) / (f + norm)
    return score


def ranked_ids(scored: Sequence[Tuple[float, Any]]) -> List[Any]:
    """Ids best-first, dropping zero scores.

    Tie-break by id, not by insertion order: the write side must be able to
    rebuild a prompt byte-for-byte on REPLAY, and a dict/set iteration order
    that follows the interpreter's hash seed silently breaks that (it already
    cost a REPLAY debugging round).
    """
    return [m for s, m in sorted(scored, key=lambda t: (-t[0], str(t[1])))
            if s > 0.0]


def fuse(lex_scored: Sequence[Tuple[float, Any]],
         cos_scored: Sequence[Tuple[float, Any]],
         k: int, seeds: Sequence[Any] = ()) -> List[Any]:
    """The ONE place the write side combines its two channels: weighted RRF.

    `seeds` are ids that must appear first and are never re-ranked. Returns at
    most k ids, seeds included.
    """
    out: List[Any] = list(dict.fromkeys(seeds))
    room = max(k - len(out), 0)
    if not room:
        return out[:k]
    keep = set(out)
    lex_scored = [(s, m) for s, m in lex_scored if m not in keep]
    cos_scored = [(s, m) for s, m in cos_scored if m not in keep]

    lr = {m: i for i, m in enumerate(ranked_ids(lex_scored))}
    cr = {m: i for i, m in enumerate(ranked_ids(cos_scored))}
    fused: Dict[Any, float] = {}
    for m, r in cr.items():
        fused[m] = fused.get(m, 0.0) + 1.0 / (RRF_K0 + r + 1)
    for m, r in lr.items():
        fused[m] = fused.get(m, 0.0) + RRF_W_LEX / (RRF_K0 + r + 1)
    return out + [m for m, _s in sorted(fused.items(),
                                        key=lambda t: (-t[1], str(t[0])))][:room]


class LexicalIndex:
    """BM25Okapi over an id-aligned corpus, for the READ side.

    ``ids[i]`` is the memory_id of the i-th document. ``scores(query)`` returns
    a list aligned to ``ids``; ``score_map`` keys it by memory_id; ``rank``
    returns memory_ids best-first.
    """

    def __init__(self, ids: List[str], texts: List[str]) -> None:
        self.ids = list(ids)
        self._corpus = [tokenize(t) for t in texts]
        # BM25Okapi requires a non-empty corpus; guard tiny/empty stores.
        self.bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def scores(self, query: str) -> List[float]:
        if self.bm25 is None:
            return [0.0] * len(self.ids)
        return list(self.bm25.get_scores(tokenize(query)))

    def score_map(self, query: str) -> Dict[str, float]:
        return {mid: s for mid, s in zip(self.ids, self.scores(query))}

    def rank(self, query: str, n: Optional[int] = None) -> List[str]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
        if n is not None:
            order = order[:n]
        return [self.ids[i] for i in order]


def build_lexical_from_store(store: MemoryStore,
                             text_source: str = "verbatim") -> LexicalIndex:
    """BM25 index over the store in ingestion order.

    ``text_source="rewritten"`` builds on the LLM-rewritten text (with per-node
    verbatim fallback), which is what the default read side uses so BM25 and
    dense sit on one surface. Ingestion order == the message list order, so the
    returned index aligns 1:1 with the ``id_by_index`` space.
    """
    ids = list(store.order)
    texts = [store.memories[mid].index_text(text_source) for mid in ids]
    return LexicalIndex(ids, texts)
