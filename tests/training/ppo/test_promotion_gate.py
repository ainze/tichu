"""PromotionGate — promote only on a CI-validated margin against ALL opponents."""

import numpy as np
import pytest

from tichu_training.ppo.promotion_gate import PromotionGate


def _normal(mean, sd, n, seed=0):
    return list(np.random.default_rng(seed).normal(mean, sd, n))


def _champ(v):
    return v["opponents"]["champion"]


def test_not_ready_before_window_full():
    gate = PromotionGate(window_games=1000)
    gate.record("champion", _normal(50, 100, 400))
    assert not gate.ready()
    assert gate.verdict()["promote"] is False


def test_strongly_positive_margin_promotes():
    gate = PromotionGate(window_games=2000)
    gate.record("champion", _normal(40, 100, 2000))  # +40/game => CI well > 0
    v = gate.verdict()
    assert _champ(v)["n"] == 2000
    assert _champ(v)["ci_lo"] > 0.0
    assert v["promote"] is True


def test_margin_straddling_zero_does_not_promote():
    gate = PromotionGate(window_games=2000)
    gate.record("champion", _normal(0.0, 100, 2000))
    v = gate.verdict()
    assert _champ(v)["ci_lo"] < 0.0 < _champ(v)["ci_hi"]
    assert v["promote"] is False


def test_threshold_must_be_cleared_not_just_zero():
    gate = PromotionGate(window_games=4000, threshold=20.0)
    gate.record("champion", _normal(15.0, 60, 4000))  # CI ~[13,17]
    assert gate.verdict()["promote"] is False
    gate0 = PromotionGate(window_games=4000, threshold=0.0)
    gate0.record("champion", _normal(15.0, 60, 4000))
    assert gate0.verdict()["promote"] is True


def test_record_accumulates_across_iters_and_reset_clears():
    gate = PromotionGate(window_games=1000)
    for _ in range(5):
        gate.record("champion", _normal(30, 100, 256))  # 5 x 256 = 1280
    assert gate.ready() and gate.n("champion") == 1280
    gate.reset()
    assert gate.n("champion") == 0 and not gate.ready()


def test_window_games_must_be_positive():
    with pytest.raises(ValueError):
        PromotionGate(window_games=0)


def test_unknown_opponent_rejected():
    gate = PromotionGate(window_games=10)
    with pytest.raises(KeyError):
        gate.record("bc", [1.0])


# --- multi-opponent: must beat EVERY opponent (the anti-cycling / beat-both gate) ---

def test_multi_opponent_requires_all_to_clear():
    gate = PromotionGate(opponents=("champion", "bc"), window_games=2000)
    gate.record("champion", _normal(40, 100, 2000))  # clears
    gate.record("bc", _normal(0.0, 100, 2000))        # straddles 0
    v = gate.verdict()
    assert _champ(v)["ci_lo"] > 0.0
    assert v["opponents"]["bc"]["ci_lo"] < 0.0
    assert v["promote"] is False  # bc fails => no promotion despite beating champion


def test_multi_opponent_not_ready_until_all_windows_full():
    gate = PromotionGate(opponents=("champion", "bc"), window_games=2000)
    gate.record("champion", _normal(40, 100, 2000))   # champion full
    # bc not recorded yet
    assert not gate.ready()
    assert gate.verdict()["promote"] is False
    gate.record("bc", _normal(40, 100, 2000))         # now both full + positive
    assert gate.ready()
    assert gate.verdict()["promote"] is True
