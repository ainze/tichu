"""Validate that the replay engine reproduces BSW `Ergebnis` on real samples.

The engine's end-of-round scoring is not yet a perfect reproduction of BSW
(small off-by-N gaps remain on rare combinations). This test pins the current
match-rate baseline so future engine work can be measured: we expect a
substantial majority of rounds to match.
"""

from pathlib import Path

import pytest

from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.validate import validate_corpus, validate_game


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def _load_samples():
    games = []
    for fname in ("2417500.tch", "2417501.tch"):
        text = (_SAMPLES / fname).read_text(encoding="utf-8")
        games.append(parse_tch(text, game_id=fname[:-4]))
    return games


def test_validate_game_produces_one_result_per_round():
    games = _load_samples()
    result = validate_game(games[0])
    assert len(result.rounds) == len(games[0].rounds)
    assert result.game_id == "2417500"


def test_no_round_replay_produces_an_illegal_action():
    """Slice B guarantees this — pinned here so a regression on legality
    surfaces through the validation pipeline too."""
    for g in _load_samples():
        result = validate_game(g)
        assert not result.any_illegal, (
            f"{g.game_id}: illegal action in rounds "
            f"{[r.round_index for r in result.rounds if r.illegal_action]}"
        )


def test_majority_of_sample_rounds_match_bsw_ergebnis():
    """Baseline match-rate test: at least 12 of the 20 sample rounds must score
    exactly. Treat this as a regression floor — the engine should improve, not
    drift backwards."""
    games = _load_samples()
    total = 0
    matched = 0
    for g in games:
        v = validate_game(g)
        for r in v.rounds:
            total += 1
            if r.matches:
                matched += 1
    assert matched >= 12, f"only {matched}/{total} rounds match"


def test_validate_corpus_aggregates_per_game_results():
    games = _load_samples()
    stats = validate_corpus(games)
    assert stats.games_total == 2
    assert stats.games_with_illegal_actions == 0
    # Failed = whole-game mismatch.
    assert len(stats.failed_game_ids) <= 2


def test_game_1_replays_cleanly_with_in_progress_trick_fix():
    """Regression guard for the in-progress-trick fix. Before the fix, game 1
    had four rounds (5, 6, 7, 9) failing with |delta|=10 score mismatches
    because the 3rd-out's final-play trick was being dropped at _finalise_round.
    The fix credits in-progress tricks at round-end. All ten rounds of game 1
    must now match BSW's Ergebnis exactly."""
    fixture = Path(__file__).resolve().parent / "data" / "game_1.tch"
    game = parse_tch(fixture.read_text(encoding="utf-8"), game_id="1")
    result = validate_game(game)
    failures = [(r.round_index, r.actual, r.expected) for r in result.rounds if not r.matches]
    assert failures == [], f"unexpected failing rounds in game 1: {failures}"
    # Pin the specific BSW scores for the formerly-failing rounds so a future
    # engine change that only "looks correct" can't silently revert these.
    by_index = {r.round_index: r for r in result.rounds}
    assert by_index[5].actual == (85, 15) == by_index[5].expected
    assert by_index[6].actual == (55, 145) == by_index[6].expected
    assert by_index[7].actual == (90, 10) == by_index[7].expected
    assert by_index[9].actual == (110, 90) == by_index[9].expected


def test_game_10078_bomb_as_lead_via_interrupt_replays_cleanly():
    """Regression guard for the empty-trick bomb-interrupt fix. In round 10
    of game 10078, the previous trick winner (p2) had just resolved and not
    yet led when p3 preempted with a 5-card straight-flush bomb (B9–BK). The
    pre-fix engine rejected this because legal_bomb_interrupts returned an
    empty set when top is None. The fix allows preempt-bombs on empty
    tricks; the round must now match BSW (195, 105)."""
    fixture = Path(__file__).resolve().parent / "data" / "game_10078.tch"
    game = parse_tch(fixture.read_text(encoding="utf-8"), game_id="10078")
    result = validate_game(game)
    round_10 = next(r for r in result.rounds if r.round_index == 10)
    assert not round_10.illegal_action, f"round 10 still illegal: {round_10}"
    assert round_10.actual == (195, 105) == round_10.expected


def test_game_10022_replays_cleanly_when_bsw_omits_drache_an():
    """Regression guard for the omitted-Drache-an fix. Game 10022 round 11
    contains a sequence where the Dragon-winner plays again immediately
    without an explicit "Drache an:" line in the BSW log. The replay layer
    must synthesise the missing DragonGive (defaulting to the left opponent,
    which is the BSW majority preference) so the rest of the round can be
    executed and the final Ergebnis (265, 35) matches."""
    fixture = Path(__file__).resolve().parent / "data" / "game_10022.tch"
    game = parse_tch(fixture.read_text(encoding="utf-8"), game_id="10022")
    result = validate_game(game)
    round_11 = next(r for r in result.rounds if r.round_index == 11)
    assert not round_11.illegal_action, f"round 11 still illegal: {round_11}"
    assert round_11.actual == (265, 35) == round_11.expected
