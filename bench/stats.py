"""Small, dependency-free statistics for eval reporting.

- wilson_ci    : Wilson score interval for a binomial proportion (good for the
                 small-n accuracies we report).
- mcnemar_exact: exact (binomial) McNemar test for paired same-question
                 on/off comparisons — the right test when the same questions
                 are scored under two configs.
"""

from __future__ import annotations

import math
from math import comb
from typing import List, Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for k successes out of n (default 95%)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_counts(on: List[bool], off: List[bool]) -> Tuple[int, int]:
    """Discordant pair counts for paired verdicts (must be same length/order).

    Returns (b, c):
      b = off correct & on wrong   (on hurt)
      c = off wrong  & on correct  (on helped)
    """
    b = sum(1 for o, f in zip(on, off) if f and not o)
    c = sum(1 for o, f in zip(on, off) if (not f) and o)
    return b, c


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)
