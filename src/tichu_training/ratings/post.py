"""Post-sweep filtering and bucketing for the ratings table.

`apply_min_games` drops players who have not played enough.
`assign_skill_deciles` buckets the surviving population into deciles 0..9
ranked by `mu`, with ties broken by `sigma` (lower → higher) then handle.
"""

from typing import Mapping

from tichu_training.ratings.sweep import PlayerRating


def apply_min_games(
    ratings: Mapping[str, PlayerRating], min_games: int
) -> dict[str, PlayerRating]:
    return {h: r for h, r in ratings.items() if r.n_games >= min_games}


def assign_skill_deciles(ratings: Mapping[str, PlayerRating]) -> dict[str, int]:
    """Return handle → decile (0..9), partitioning by quantile on mu.

    Tie-break: lower sigma is stronger (more certain); handle lex is final.
    """
    if not ratings:
        return {}
    # Sort weakest → strongest.
    ordered = sorted(
        ratings.values(),
        key=lambda p: (p.mu, -p.sigma, p.handle),
    )
    n = len(ordered)
    out: dict[str, int] = {}
    for i, p in enumerate(ordered):
        # Map rank i ∈ [0, n) → decile 0..9 via quantile slicing.
        decile = min(9, (i * 10) // n)
        out[p.handle] = decile
    return out
