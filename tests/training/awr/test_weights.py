"""AWR weight function: clipped exp(advantage / beta)."""

import numpy as np
import pytest

from tichu_training.awr.weights import awr_weights


def test_returns_float32_array_of_same_shape():
    adv = np.array([0.5, -0.5, 1.5], dtype=np.float64)
    w = awr_weights(adv, beta=1.0)
    assert w.shape == adv.shape
    assert w.dtype == np.float32


def test_weights_are_non_negative_and_finite():
    adv = np.array([-10.0, 0.0, 10.0, 100.0, -100.0], dtype=np.float32)
    w = awr_weights(adv, beta=1.0, max_weight=20.0)
    assert (w >= 0).all()
    assert np.isfinite(w).all()


def test_max_weight_clip_enforced():
    adv = np.array([100.0, 200.0, 1000.0], dtype=np.float32)
    w = awr_weights(adv, beta=1.0, max_weight=20.0)
    assert (w <= 20.0).all()


def test_monotonic_in_advantage_before_clip():
    adv = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    w = awr_weights(adv, beta=1.0, max_weight=1000.0)
    for i in range(len(w) - 1):
        assert w[i] <= w[i + 1], (i, w[i], w[i + 1])


def test_large_beta_collapses_to_uniform():
    adv = np.array([-5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    w = awr_weights(adv, beta=1e6, max_weight=10.0)
    assert np.allclose(w, w[0], atol=1e-3)


def test_invalid_beta_raises():
    adv = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError):
        awr_weights(adv, beta=0.0)
    with pytest.raises(ValueError):
        awr_weights(adv, beta=-1.0)


def test_numerical_stability_with_huge_advantages():
    # Without subtracting max(adv), exp(1000) would overflow to inf.
    adv = np.array([1000.0, 999.0, 998.0], dtype=np.float32)
    w = awr_weights(adv, beta=1.0, max_weight=20.0)
    assert np.isfinite(w).all()
    assert (w > 0).all()
