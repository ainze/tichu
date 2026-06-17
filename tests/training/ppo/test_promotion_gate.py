"""PromotionGate — promote only on a CI-validated candidate-vs-champion margin."""

import numpy as np
import pytest

from tichu_training.ppo.promotion_gate import PromotionGate


def _normal(mean, sd, n, seed=0):
    return list(np.random.default_rng(seed).normal(mean, sd, n))


def test_not_ready_before_window_full():
    gate = PromotionGate(window_games=1000)
    gate.record(_normal(50, 100, 400))
    assert not gate.ready()
    assert gate.verdict()["promote"] is False


def test_strongly_positive_margin_promotes():
    gate = PromotionGate(window_games=2000)
    gate.record(_normal(40, 100, 2000))  # +40/game, SE ~2.2 over 2000 => CI well > 0
    v = gate.verdict()
    assert v["n"] == 2000
    assert v["ci_lo"] > 0.0
    assert v["promote"] is True


def test_margin_straddling_zero_does_not_promote():
    gate = PromotionGate(window_games=2000)
    gate.record(_normal(0.0, 100, 2000))  # mean ~0 => CI spans 0
    v = gate.verdict()
    assert v["ci_lo"] < 0.0 < v["ci_hi"]
    assert v["promote"] is False


def test_threshold_must_be_cleared_not_just_zero():
    # A clearly-significant +15/game edge (sd 60, n 4000 => CI ~[13, 17]): promotes
    # against a 0 threshold but NOT against a +20 threshold.
    gate = PromotionGate(window_games=4000, threshold=20.0)
    gate.record(_normal(15.0, 60, 4000))
    assert gate.verdict()["promote"] is False
    gate0 = PromotionGate(window_games=4000, threshold=0.0)
    gate0.record(_normal(15.0, 60, 4000))
    assert gate0.verdict()["promote"] is True


def test_record_accumulates_across_iters_and_reset_clears():
    gate = PromotionGate(window_games=1000)
    for _ in range(5):
        gate.record(_normal(30, 100, 256))  # 5 iters x 256 games = 1280
    assert gate.ready() and gate.n == 1280
    gate.reset()
    assert gate.n == 0 and not gate.ready()


def test_window_games_must_be_positive():
    with pytest.raises(ValueError):
        PromotionGate(window_games=0)
