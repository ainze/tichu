"""Post-sweep filtering and bucketing: min-games + skill decile."""

import trueskill

from tichu_training.ratings.post import (
    apply_min_games,
    assign_skill_deciles,
)
from tichu_training.ratings.sweep import PlayerRating


def _r(handle: str, mu: float, sigma: float = 5.0, n_games: int = 10) -> PlayerRating:
    return PlayerRating(handle=handle, rating=trueskill.Rating(mu=mu, sigma=sigma), n_games=n_games)


def test_apply_min_games_drops_below_threshold():
    ratings = {
        "Alice": _r("Alice", mu=30.0, n_games=25),
        "Bob": _r("Bob", mu=22.0, n_games=10),
        "Carol": _r("Carol", mu=28.0, n_games=20),
    }
    surviving = apply_min_games(ratings, min_games=20)
    assert set(surviving.keys()) == {"Alice", "Carol"}


def test_apply_min_games_boundary_is_inclusive():
    ratings = {"Alice": _r("Alice", mu=30.0, n_games=20)}
    assert "Alice" in apply_min_games(ratings, min_games=20)


def test_assign_skill_deciles_covers_zero_through_nine():
    # 10 players, mus = 1..10. With deciles by quantile, each player → one decile.
    ratings = {f"p{i}": _r(f"p{i}", mu=float(i)) for i in range(1, 11)}
    deciles = assign_skill_deciles(ratings)
    assert set(deciles.values()) == set(range(10))
    # Highest mu → highest decile.
    assert deciles["p10"] == 9
    assert deciles["p1"] == 0


def test_assign_skill_deciles_ties_broken_by_sigma_then_handle():
    # Two players with the same mu: lower sigma should rank higher.
    ratings = {
        "Aaa": _r("Aaa", mu=25.0, sigma=8.0),
        "Bbb": _r("Bbb", mu=25.0, sigma=3.0),
    }
    deciles = assign_skill_deciles(ratings)
    assert deciles["Bbb"] > deciles["Aaa"]


def test_assign_skill_deciles_uniformity_on_100_players():
    ratings = {f"p{i:03d}": _r(f"p{i:03d}", mu=float(i)) for i in range(100)}
    deciles = assign_skill_deciles(ratings)
    counts = [0] * 10
    for d in deciles.values():
        counts[d] += 1
    total = sum(counts)
    # Spec sanity: no decile holds > 15% of rated players.
    assert max(counts) / total <= 0.15
    assert min(counts) >= 1
