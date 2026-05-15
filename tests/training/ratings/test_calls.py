"""Tichu / Grand Tichu success-rate counters."""

from tichu_training.bsw.records import ParsedGame, ParsedRound
from tichu_training.ratings.calls import compute_call_stats


def _round(idx, ergebnis, tichu=(), grand=()):
    return ParsedRound(
        round_index=idx,
        pre_deal_hands=((), (), (), ()),
        start_hands=((), (), (), ()),
        grand_tichu_callers=frozenset(grand),
        tichu_callers=frozenset(tichu),
        schupfen=(),
        plays=(),
        ergebnis=ergebnis,
    )


def _game(handles, rounds):
    return ParsedGame(game_id="g", handles=handles, rounds=rounds)


def test_tichu_success_when_margin_at_least_100():
    # Seat 0 (Alice) is on team 0. Margin team0-team1 = 150 → success.
    game = _game(
        ("Alice", "Bob", "Carol", "Dave"),
        [_round(0, ergebnis=(150, 0), tichu={0})],
    )
    stats = compute_call_stats([game])
    assert stats["Alice"].tichu_calls == 1
    assert stats["Alice"].tichu_wins == 1


def test_tichu_failure_when_margin_below_100():
    # Margin = 50 — too small to credit the caller with a successful Tichu.
    game = _game(
        ("Alice", "Bob", "Carol", "Dave"),
        [_round(0, ergebnis=(80, 30), tichu={0})],
    )
    stats = compute_call_stats([game])
    assert stats["Alice"].tichu_calls == 1
    assert stats["Alice"].tichu_wins == 0


def test_grand_tichu_tracked_separately():
    game = _game(
        ("Alice", "Bob", "Carol", "Dave"),
        [_round(0, ergebnis=(250, 0), tichu=set(), grand={1})],
    )
    stats = compute_call_stats([game])
    assert "Alice" not in stats
    # Bob is on team 1. Margin team1-team0 = 0 - 250 = -250 → loss.
    assert stats["Bob"].grand_tichu_calls == 1
    assert stats["Bob"].grand_tichu_wins == 0
    assert stats["Bob"].tichu_calls == 0


def test_non_callers_never_appear():
    game = _game(
        ("Alice", "Bob", "Carol", "Dave"),
        [_round(0, ergebnis=(100, 100))],
    )
    stats = compute_call_stats([game])
    assert stats == {}


def test_counts_accumulate_across_games():
    g1 = _game(("Alice", "Bob", "Carol", "Dave"), [_round(0, (200, 0), tichu={0})])
    g2 = _game(("Alice", "Bob", "Carol", "Dave"), [_round(0, (0, 200), tichu={0})])
    stats = compute_call_stats([g1, g2])
    assert stats["Alice"].tichu_calls == 2
    assert stats["Alice"].tichu_wins == 1
