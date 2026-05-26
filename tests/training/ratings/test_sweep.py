"""Per-game TrueSkill sweep over parsed BSW games."""

import trueskill

from tichu_training.bsw.records import ParsedGame, ParsedRound
from tichu_training.ratings.sweep import compute_ratings


_DEFAULT_MU = trueskill.TrueSkill().mu


def _game(game_id: str, handles, ergebnis_rounds, *, per_round_handles=None):
    """Build a stub ParsedGame with the given per-round ergebnis tuples.

    `per_round_handles`, if provided, overrides the per-round handles
    seat-tuple for each round (used to simulate substitutions and anonymous
    seats). Defaults to repeating the game-level `handles` per round.
    """
    if per_round_handles is None:
        per_round_handles = [handles] * len(ergebnis_rounds)
    assert len(per_round_handles) == len(ergebnis_rounds)
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
            handles=round_handles,
        )
        for i, (ergebnis, round_handles) in enumerate(zip(ergebnis_rounds, per_round_handles))
    )
    return ParsedGame(game_id=game_id, rounds=rounds)


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


def test_game_with_any_anonymous_seat_is_skipped():
    """A game where any round contains an Anonymous Seat (empty handle) is
    excluded from the TrueSkill sweep entirely. The anonymous handle ``""``
    never receives a rating, and the named players in that game are not
    updated either — the asymmetric tolerance design of ADR-0010: BC
    pipeline accepts anonymous seats (→ Neutral Skill Decile), TrueSkill
    rejects them (preserves per-identified-stable-game rating semantics)."""
    game = _game(
        "g",
        ("Alice", "Bob", "Carol", "Dave"),
        [(200, 0), (100, 100)],
        per_round_handles=[
            ("Alice", "", "Carol", "Dave"),  # round 0: seat 1 anonymous
            ("Alice", "Bob", "Carol", "Dave"),
        ],
    )
    ratings = compute_ratings([game])

    # Anonymous handle never enters the rating table.
    assert "" not in ratings
    # Named players in the skipped game are not updated.
    for handle in ("Alice", "Bob", "Carol", "Dave"):
        assert handle not in ratings or ratings[handle].n_games == 0


def test_game_with_seat_handle_substitution_is_skipped():
    """A game where a seat's handle changes between rounds (mid-game player
    substitution) is excluded from the TrueSkill sweep — neither the
    departing nor the arriving handle receives a rating update from this
    game. Preserves the per-identified-stable-game invariant. See ADR-0010."""
    game = _game(
        "g",
        ("Alice", "Bob", "Carol", "Dave"),
        [(200, 0), (200, 0)],
        per_round_handles=[
            ("Alice", "Bob", "Carol", "Dave"),
            ("Alice", "BobReplacement", "Carol", "Dave"),  # seat 1 swapped
        ],
    )
    ratings = compute_ratings([game])

    for handle in ("Alice", "Bob", "BobReplacement", "Carol", "Dave"):
        assert handle not in ratings or ratings[handle].n_games == 0


def test_winner_team_decided_by_sum_across_rounds():
    # Team A loses round 0 but wins overall on totals.
    game = _game("1", ("Alice", "Bob", "Carol", "Dave"), [(40, 60), (200, 0)])
    ratings = compute_ratings([game])
    assert ratings["Alice"].mu > _DEFAULT_MU
    assert ratings["Bob"].mu < _DEFAULT_MU


