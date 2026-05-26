"""Tichu / Grand Tichu success-rate counters."""

from tichu_training.bsw.records import ParsedGame, ParsedRound
from tichu_training.ratings.calls import compute_call_stats


def _round(idx, ergebnis, tichu=(), grand=(), handles=("Alice", "Bob", "Carol", "Dave")):
    return ParsedRound(
        round_index=idx,
        pre_deal_hands=((), (), (), ()),
        start_hands=((), (), (), ()),
        grand_tichu_callers=frozenset(grand),
        tichu_callers=frozenset(tichu),
        schupfen=(),
        plays=(),
        ergebnis=ergebnis,
        handles=handles,
    )


def _game(handles, rounds):
    # `handles` retained as a parameter for back-compat with old tests; the
    # per-round handles on each ParsedRound are the canonical source. We
    # ignore the game-level arg — kept only so existing call sites still read.
    del handles
    return ParsedGame(game_id="g", rounds=rounds)


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


def test_call_credited_to_per_round_handle_not_game_snapshot():
    """When a seat's handle changes mid-game (BSW player substitution), a
    Tichu call in a post-substitution round must be credited to the seat's
    *round-level* handle, not the round-0 snapshot. See ADR-0010."""
    rounds = [
        _round(0, (200, 0), tichu={1}, handles=("Alice", "Bob", "Carol", "Dave")),
        _round(1, (200, 0), tichu={1}, handles=("Alice", "BobReplacement", "Carol", "Dave")),
    ]
    game = _game(("Alice", "Bob", "Carol", "Dave"), rounds)
    stats = compute_call_stats([game])

    assert stats["Bob"].tichu_calls == 1
    assert stats["BobReplacement"].tichu_calls == 1, (
        "post-substitution call mis-credited to round-0 handle"
    )


def test_call_from_anonymous_seat_is_skipped():
    """A call recorded for a seat whose handle is anonymous (``""``) is
    excluded from call-success counters — the empty-string handle never
    aggregates synthetic stats. See ADR-0010."""
    game = _game(
        ("", "Bob", "Carol", "Dave"),
        [_round(0, (200, 0), tichu={0}, handles=("", "Bob", "Carol", "Dave"))],
    )
    stats = compute_call_stats([game])
    assert "" not in stats
