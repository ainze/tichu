"""Replay-legality test.

Every parsed action must be a legal engine action at the corresponding state.
This is the legality-only half of replay validation — score correctness comes
later (see test_replay_ergebnis.py once engine scoring is finished).
"""

from pathlib import Path

import pytest

from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.replay import replay_round


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


@pytest.mark.parametrize("filename", ["2417500.tch", "2417501.tch"])
def test_every_parsed_action_is_legal_in_the_engine(filename):
    game = parse_tch((_SAMPLES / filename).read_text(encoding="utf-8"), game_id=filename[:-4])
    for r in game.rounds:
        result = replay_round(r)
        assert result.illegal_action is None, (
            f"{filename} round {r.round_index}: illegal action {result.illegal_action!r} "
            f"at step {result.steps_taken}"
        )
