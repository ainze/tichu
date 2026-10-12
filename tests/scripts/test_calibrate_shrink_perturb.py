"""scripts/calibrate_shrink_perturb.py: the pick rule and the summary.

The tournament itself is the ship-check harness (`collect_pair_deltas`), tested
where it lives. What is new here is the decision: which alpha becomes arm P's
blend. Arm P v1 failed because its blend was never measured, so the rule that
turns measurements into a pick is pinned.
"""

import numpy as np
import pytest

from scripts import calibrate_shrink_perturb as csp


def _row(alpha, mean):
    return {"alpha": alpha, "mean": mean, "ci_lo": mean - 2, "ci_hi": mean + 2, "n_obs": 100}


def test_pick_is_the_strongest_perturbation_inside_the_band():
    rows = [_row(0.95, -1.0), _row(0.9, -4.0), _row(0.85, -12.0), _row(0.8, -40.0)]
    assert csp.pick_alpha(rows, lo=3, hi=15)["alpha"] == 0.85


def test_band_edges_are_inclusive():
    assert csp.pick_alpha([_row(0.9, -3.0)], lo=3, hi=15)["alpha"] == 0.9
    assert csp.pick_alpha([_row(0.8, -15.0)], lo=3, hi=15)["alpha"] == 0.8


def test_no_pick_when_every_alpha_misses_the_band():
    rows = [_row(0.95, -0.5), _row(0.7, -60.0)]
    assert csp.pick_alpha(rows, lo=3, hi=15) is None


def test_a_blend_that_beats_the_champion_is_not_a_pick():
    assert csp.pick_alpha([_row(0.95, +5.0)], lo=3, hi=15) is None


def test_bad_band_is_refused():
    with pytest.raises(ValueError):
        csp.pick_alpha([_row(0.9, -5.0)], lo=15, hi=3)


def test_calibrate_evaluates_every_alpha_once_strongest_last():
    seen = []

    def fake(alpha):
        seen.append(alpha)
        return np.full(20, -100 * (1 - alpha))

    rows = csp.calibrate([0.8, 0.95, 0.9], fake)
    assert seen == [0.95, 0.9, 0.8]
    assert [r["alpha"] for r in rows] == [0.95, 0.9, 0.8]
    assert rows[2]["mean"] == pytest.approx(-20.0)


def test_summary_uses_the_paired_cluster_ci():
    # Seat-swap pairs whose card luck cancels exactly: every DEAL means +1, so the
    # paired CI collapses onto +1 while a flat resample of the arrangements would not.
    rng = np.random.default_rng(0)
    luck = rng.normal(0, 100, 500)
    totals = np.empty(1000)
    totals[0::2] = 1 + luck
    totals[1::2] = 1 - luck
    s = csp.summarise(totals, iters=2000)
    assert s["mean"] == pytest.approx(1.0)
    assert s["ci_lo"] == pytest.approx(1.0) and s["ci_hi"] == pytest.approx(1.0)
    assert s["n_obs"] == 1000
