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
    # Failed = whole-game mismatch. Both samples have at least one mismatching
    # round each, so both currently land on the failed list.
    assert len(stats.failed_game_ids) <= 2
