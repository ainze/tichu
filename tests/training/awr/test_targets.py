"""Unit tests for AWR value-baseline target selection.

Pure-Python: imports nothing from torch or BCModel, so this suite runs
without a working torch install.
"""

import pytest

from tichu_training.awr.targets import VALUE_TARGETS, target_value


def test_round_target_returns_round_outcome():
    assert target_value(round_outcome=42.0, game_won=None, kind="round") == 42.0


def test_round_target_ignores_game_won():
    """`round` target reads round_outcome regardless of game_won state —
    including Incomplete Sessions (game_won=None). Round-level data is
    always defined."""
    for gw in (None, True, False):
        assert target_value(round_outcome=-7.5, game_won=gw, kind="round") == -7.5


def test_game_target_returns_1_for_true():
    assert target_value(round_outcome=0.0, game_won=True, kind="game") == 1.0


def test_game_target_returns_0_for_false():
    assert target_value(round_outcome=0.0, game_won=False, kind="game") == 0.0


def test_game_target_returns_none_for_incomplete_session():
    """Incomplete Session: caller filters these out of the baseline fit."""
    assert target_value(round_outcome=0.0, game_won=None, kind="game") is None


def test_game_target_ignores_round_outcome():
    """A team that won a Complete Game stays game_won=True even if the
    last round was a loss (positive game_outcome, negative round_outcome)."""
    assert target_value(round_outcome=-200.0, game_won=True, kind="game") == 1.0


def test_unknown_kind_raises_value_error():
    with pytest.raises(ValueError, match="unknown value_target"):
        target_value(round_outcome=0.0, game_won=None, kind="margin")


def test_value_targets_constant_exposes_valid_kinds():
    """Sanity: any kind enumerated in VALUE_TARGETS must be accepted."""
    for kind in VALUE_TARGETS:
        target_value(round_outcome=0.0, game_won=True, kind=kind)
