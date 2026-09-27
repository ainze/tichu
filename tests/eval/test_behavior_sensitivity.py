"""Behavior Sensitivity Probe — pairing arms, per-arm effects, Holm, verdicts.

Hand-built logs whose answer is known; see the pre-registration
(docs/notes/2026-09-27-behavior-sensitivity-preregistration.md)."""

import numpy as np
import pandas as pd
import pytest

from tichu_eval.behavior_sensitivity import (
    LEVER_METRICS,
    arm_effect,
    holm,
    pair_arms,
    verdicts,
)
from tichu_eval.drift_arms import DriftLog


def _round(arm, deal, half, margin):
    team = half % 2
    totals = (margin, 0) if team == 0 else (0, margin)
    return {"arm": arm, "deal": deal, "half": half, "subject_team": team,
            "total_0": totals[0], "total_1": totals[1], "call_bonus_0": 0, "call_bonus_1": 0}


def _follow(arm, deal, action):
    return {"arm": arm, "deal": deal, "half": 0, "seat": 0, "kind": "play", "is_subject": True,
            "forced": False, "holder": "opponent", "has_beat": True, "action": action,
            "trick_points": 0}


def _arm(name, margins, passes):
    """`margins[d]` per deal (both halves); `passes[d]` of 4 opponent-follows pass."""
    rounds = [_round(name, d, h, m) for d, m in enumerate(margins) for h in (0, 1)]
    decisions = [_follow(name, d, "pass" if i < p else "single")
                 for d, p in enumerate(passes) for i in range(4)]
    return DriftLog(decisions=pd.DataFrame(decisions), rounds=pd.DataFrame(rounds))


def test_every_lever_names_a_real_drift_metric():
    from tichu_eval.drift_metrics import metric

    for lever, (name, _cell) in LEVER_METRICS.items():
        metric(name)       # raises StopIteration if it does not exist


def test_pair_arms_labels_the_reference_bc_and_the_treatment_subject():
    log = pair_arms(_arm("ref", [0, 0], [1, 1]), _arm("t", [5, 5], [2, 2]))
    assert set(log.rounds.arm) == {"bc", "subject"}
    assert (log.rounds[log.rounds.arm == "subject"].total_0.max()) == 5


def test_arm_effect_reads_the_ev_and_the_lever_rate_change():
    rng = np.random.default_rng(0)
    n = 300
    ref_m = rng.normal(0, 50, n)
    ref = _arm("ref", ref_m, rng.integers(0, 2, n))            # ~12.5% pass
    treat = _arm("t", ref_m + 10.0, np.full(n, 2))             # +10/Round, 50% pass
    e = arm_effect(ref, treat, "pass_vs_opponent", n_boot=500, seed=0)
    assert e["d_ev"] == pytest.approx(10.0)
    assert e["d_ev_lo"] == pytest.approx(10.0) and e["d_ev_hi"] == pytest.approx(10.0)
    assert e["rate_ref"] < 0.2 and e["rate_treat"] == pytest.approx(0.5)
    assert e["d_rate"] == pytest.approx(e["rate_treat"] - e["rate_ref"])
    # slope: points per Round per percentage point of the lever rate
    assert e["slope"] == pytest.approx(10.0 / (100 * e["d_rate"]))
    assert e["p"] < 0.01


def test_holm_step_down():
    assert holm([0.001, 0.02, 0.04, 0.5], alpha=0.05).tolist() == [True, False, False, False]
    assert holm([0.001, 0.012, 0.3], alpha=0.05).tolist() == [True, True, False]
    assert holm([np.nan, 0.001], alpha=0.05).tolist() == [False, True]


def _row(lever, sign, d_ev, lo, hi, d_rate=0.10, p=None):
    if p is None:
        p = 0.0001 if (lo > 0 or hi < 0) else 0.5
    return {"lever": lever, "sign": sign, "d_ev": d_ev, "d_ev_lo": lo, "d_ev_hi": hi,
            "d_rate": d_rate, "p": p}


def test_verdicts_follow_the_preregistered_rules():
    rows = pd.DataFrame([
        _row("a", "+", 2.0, 1.2, 2.8), _row("a", "-", -1.0, -2.0, 0.1),        # lever found
        _row("b", "+", -3.0, -4.0, -2.0), _row("b", "-", -2.5, -3.5, -1.5),    # sharp optimum
        _row("c", "+", 0.2, -0.8, 1.2), _row("c", "-", -0.1, -1.1, 0.9),       # flat
        _row("d", "+", 0.5, -0.5, 1.5, d_rate=0.03), _row("d", "-", 0.0, -1.0, 1.0),
    ])
    v = verdicts(rows)
    assert v.levers == {"a": "lever found", "b": "sharp optimum", "c": "flat optimum",
                        "d": "flat optimum"}
    assert v.failed_manipulation == [("d", "+")]
    assert not v.global_null                       # a gained


def test_global_null_needs_every_upper_bound_below_two():
    flat = [_row(x, s, 0.1, -1.0, 1.2) for x in "ab" for s in "+-"]
    assert verdicts(pd.DataFrame(flat)).global_null
    wide = flat[:-1] + [_row("b", "-", 0.5, -1.5, 2.5)]
    assert not verdicts(pd.DataFrame(wide)).global_null
