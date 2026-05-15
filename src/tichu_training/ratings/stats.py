"""Stdlib-only Spearman rank correlation."""

from typing import Sequence


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return Spearman rank correlation between `xs` and `ys`.

    Uses average ranks for tied values. Returns `None` if there are fewer
    than 2 points or if either side has zero variance after ranking.
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    return _pearson(rx, ry)


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        # Tied positions i..j (0-based) → 1-based ranks (i+1)..(j+1).
        avg = ((i + 1) + (j + 1)) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = sum((x - mx) ** 2 for x in xs)
    denom_y = sum((y - my) ** 2 for y in ys)
    if denom_x == 0 or denom_y == 0:
        return None
    return num / (denom_x * denom_y) ** 0.5
