"""What an embedder has to be, plus the one piece of maths they all share.

``embed(texts) -> List[List[float]]``, one vector per input, SAME ORDER, and unit
length: every consumer compares with a bare dot product or pgvector's ``<=>``, so
an unnormalized vector does not error, it just ranks wrong.

An embedder may return ``[]`` for an input it could not embed; callers must treat
an empty vector as "no dense signal for this node", never as the origin.

Every embedder also STATES ITS WIDTH as ``dim``: pgvector's column is
``vector(dim)``, created when the table is, and Postgres rejects a vector of the
wrong length instead of truncating it.
"""

from __future__ import annotations

import math
from typing import List


class Embedder:
    """The interface. Two lines, because that is all the method needs of it."""

    #: Vector length. 0 means the provider did not state one, which a caller that
    #: needs it up front must refuse rather than guess -- see the module docstring.
    dim: int = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


def l2_normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        return v
    return [x / n for x in v]
