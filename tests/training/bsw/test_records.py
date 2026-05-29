"""Unit tests for pure derivations on `ParsedGame` / `ParsedRound`."""

from tichu_training.bsw.records import (
    ParsedGame,
    ParsedRound,
    game_team_totals,
)


def _round(ergebnis: tuple[int, int]) -> ParsedRound:
    """Minimal ParsedRound — only `ergebnis` matters for these tests."""
    return ParsedRound(
        round_index=0,
        pre_deal_hands=(frozenset(), frozenset(), frozenset(), frozenset()),
        start_hands=(frozenset(), frozenset(), frozenset(), frozenset()),
        grand_tichu_callers=frozenset(),
        tichu_callers=frozenset(),
        schupfen=(),
        plays=(),
        ergebnis=ergebnis,
    )


def test_game_team_totals_at_999_is_incomplete_session():
    """Below the 1000-point threshold for both teams: Incomplete Session."""
    game = ParsedGame(game_id="x", rounds=(_round((999, -10)),))
    assert game_team_totals(game) is None


def test_game_team_totals_at_exactly_1000_is_complete_game():
    """The threshold is inclusive: Tichu's first-to-1000 rule means exactly
    1000 is game-over. Pinned because it's easy to off-by-one to >1000."""
    game = ParsedGame(game_id="x", rounds=(_round((1000, -10)),))
    assert game_team_totals(game) == (1000, -10)


def test_game_team_totals_sums_across_rounds():
    """Game with no single round reaching 1000 but cumulative crossing it."""
    game = ParsedGame(game_id="x", rounds=(
        _round((400, -100)),
        _round((300, 200)),
        _round((350, 400)),  # cumulative: (1050, 500) — team 0 wins
    ))
    assert game_team_totals(game) == (1050, 500)


def test_game_team_totals_either_team_reaching_threshold_completes_the_game():
    """Either team crossing 1000 makes it a Complete Game — doesn't have
    to be team 0. Mirrors the 'first-to-1000' rule symmetrically."""
    game = ParsedGame(game_id="x", rounds=(_round((-50, 1200)),))
    assert game_team_totals(game) == (-50, 1200)
