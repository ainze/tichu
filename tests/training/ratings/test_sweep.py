"""Per-game TrueSkill sweep over parsed BSW games."""

import trueskill

from tichu_training.bsw.records import ParsedGame, ParsedRound
from tichu_training.ratings.sweep import compute_ratings


_DEFAULT_MU = trueskill.TrueSkill().mu


def _game(game_id: str, handles, ergebnis_rounds):
    """Build a stub ParsedGame with the given per-round ergebnis tuples."""
    rounds = tuple(
        ParsedRound(
            round_index=i,
            pre_deal_hands=((), (), (), ()),
            start_hands=((), (), (), ()),
            grand_tichu_callers=frozenset(),
            tichu_callers=frozenset(),
            schupfen=(),
            plays=(),
            ergebnis=ergebnis,
        )
        for i, ergebnis in enumerate(ergebnis_rounds)
    )
    return ParsedGame(game_id=game_id, handles=handles, rounds=rounds)


def test_winners_mu_rises_losers_mu_falls():
    # Team A (seats 0+2) beats Team B (seats 1+3): 200 vs 0 across one round.
    game = _game("1", ("Alice", "Bob", "Carol", "Dave"), [(200, 0)])
    ratings = compute_ratings([game])

    assert ratings["Alice"].mu > _DEFAULT_MU
    assert ratings["Carol"].mu > _DEFAULT_MU
    assert ratings["Bob"].mu < _DEFAULT_MU
    assert ratings["Dave"].mu < _DEFAULT_MU


def test_n_games_counts_appearances():
    g1 = _game("1", ("Alice", "Bob", "Carol", "Dave"), [(200, 0)])
    g2 = _game("2", ("Alice", "Bob", "Carol", "Dave"), [(0, 200)])
    ratings = compute_ratings([g1, g2])
    for handle in ("Alice", "Bob", "Carol", "Dave"):
        assert ratings[handle].n_games == 2


def test_draws_are_skipped():
    # Equal sums across the game's rounds → no update.
    game = _game("1", ("Alice", "Bob", "Carol", "Dave"), [(100, 100)])
    ratings = compute_ratings([game])
    for handle in ("Alice", "Bob", "Carol", "Dave"):
        # Player is still recorded with default rating but n_games stays 0.
        assert handle not in ratings or ratings[handle].n_games == 0


def test_winner_team_decided_by_sum_across_rounds():
    # Team A loses round 0 but wins overall on totals.
    game = _game("1", ("Alice", "Bob", "Carol", "Dave"), [(40, 60), (200, 0)])
    ratings = compute_ratings([game])
    assert ratings["Alice"].mu > _DEFAULT_MU
    assert ratings["Bob"].mu < _DEFAULT_MU


