"""Behavioral Drift Benchmark — the statistics.

Every metric reduces to per-deal `(events, opportunities)` counts per arm (and per
cell for a distribution). `summarise_cells` turns those into the tidy table:
Exposure + Conditional Rate per arm (ratio of sums), a paired Δ, deal-cluster
bootstrap 95% CIs, suppression below the opportunity floor. See CONTEXT.md
§"Exposure" / §"Conditional Rate".
"""

import numpy as np
import pandas as pd
import pytest

from tichu_eval.drift_stats import benjamini_hochberg, summarise_cells


def _counts(rows):
    return pd.DataFrame(rows, columns=["deal", "arm", "cell", "events", "opportunities"])


def _row(summary, arm):
    return summary[summary.arm == arm].iloc[0]


def test_rate_is_ratio_of_sums_not_mean_of_per_deal_rates():
    # Deal 0: 1/1, deal 1: 0/9. Mean of rates = 0.5; ratio of sums = 0.1.
    c = _counts([(0, "bc", "all", 1, 1), (1, "bc", "all", 0, 9),
                 (0, "subject", "all", 1, 1), (1, "subject", "all", 0, 9)])
    s = summarise_cells(c, n_boot=200, seed=0, floor=0)
    assert _row(s, "bc").rate == pytest.approx(0.1)


def test_exposure_is_opportunities_per_100_seat_rounds():
    # 3 opportunities on each of 5 deals; 5 more deals where it never arises.
    c = _counts([(d, arm, "all", 0, 3) for d in range(5) for arm in ("bc", "subject")])
    # 10 deals x 2 halves x 2 Subject seats = 40 seat-Rounds per arm; 15 opps.
    s = summarise_cells(c, deals=range(10), n_boot=100, seed=0, floor=0)
    assert _row(s, "bc").exposure == pytest.approx(37.5)


def test_delta_is_subject_minus_bc_with_a_ci_that_brackets_it():
    rng = np.random.default_rng(1)
    rows = []
    for d in range(400):
        opp = int(rng.integers(1, 6))
        rows.append((d, "bc", "all", int(rng.binomial(opp, 0.3)), opp))
        rows.append((d, "subject", "all", int(rng.binomial(opp, 0.5)), opp))
    s = summarise_cells(_counts(rows), n_boot=500, seed=0, floor=0)
    d = _row(s, "delta")
    assert d.rate == pytest.approx(_row(s, "subject").rate - _row(s, "bc").rate)
    assert d.rate_lo < d.rate < d.rate_hi
    assert d.rate_lo > 0.1 and d.rate_hi < 0.3   # true Δ is 0.2


def test_paired_bootstrap_is_tighter_than_unpaired_when_deals_correlate():
    # Deal difficulty drives both arms (the reason deals are shared): the paired
    # Δ CI must be far narrower than the per-arm CIs would suggest.
    rng = np.random.default_rng(2)
    rows = []
    for d in range(300):
        p = rng.uniform(0.05, 0.95)          # shared deal effect
        rows.append((d, "bc", "all", int(p * 100), 100))
        rows.append((d, "subject", "all", int(p * 100) + 2, 100))
    s = summarise_cells(_counts(rows), n_boot=500, seed=0, floor=0)
    d, b = _row(s, "delta"), _row(s, "bc")
    assert (d.rate_hi - d.rate_lo) < 0.2 * (b.rate_hi - b.rate_lo)


def test_cells_below_the_opportunity_floor_are_suppressed():
    c = _counts([(0, "bc", "rare", 1, 5), (0, "subject", "rare", 2, 5),
                 (0, "bc", "common", 100, 300), (0, "subject", "common", 90, 300)])
    s = summarise_cells(c, n_boot=50, seed=0, floor=200)
    assert s[s.cell == "rare"].suppressed.all()
    assert not s[s.cell == "common"].suppressed.any()
    assert s[s.cell == "rare"].rate.isna().all()


def test_benjamini_hochberg_keeps_only_the_sturdy_discoveries():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # Classic worked example: at q=0.05 only the first two survive.
    assert benjamini_hochberg(p, q=0.05).tolist() == [True, True] + [False] * 8


# --- The human reference: unpaired, clustered by Game ----------------------------

def _ref_rounds(games):
    """Reference Rounds: `games` maps game id -> number of Rounds; every Round is
    measured from both teams (two rows, half 0 / 1)."""
    rows, deal = [], 0
    for g, n in games.items():
        for _ in range(n):
            rows += [{"arm": "humans", "deal": deal, "half": h, "game": g} for h in (0, 1)]
            deal += 1
    return pd.DataFrame(rows)


def test_reference_rate_and_exposure_use_the_measured_seat_rounds():
    from tichu_eval.drift_stats import summarise_reference

    rounds = _ref_rounds({"a": 3, "b": 2})          # 5 Rounds x 2 teams x 2 seats = 20
    c = _counts([(d, "humans", "all", 1, 2) for d in range(5)])
    s = summarise_reference(c, rounds, n_boot=100, seed=0, floor=0)
    r = s.iloc[0]
    assert r.arm == "humans" and r.rate == pytest.approx(0.5)
    assert r.exposure == pytest.approx(100 * 10 / 20)
    assert np.isnan(r.p) and not r.bh_survives


def test_reference_ci_resamples_whole_games():
    # Two Games: every Round of one says yes, every Round of the other says no.
    # Only 2 independent units exist, so the CI must span [0, 1] — a per-Round
    # bootstrap would report a falsely tight interval around 0.5.
    from tichu_eval.drift_stats import summarise_reference

    rounds = _ref_rounds({"a": 50, "b": 50})
    c = _counts([(d, "humans", "all", int(d < 50), 1) for d in range(100)])
    r = summarise_reference(c, rounds, n_boot=400, seed=0, floor=0).iloc[0]
    assert r.rate_lo == pytest.approx(0.0) and r.rate_hi == pytest.approx(1.0)


def test_reference_cells_below_the_floor_are_suppressed():
    from tichu_eval.drift_stats import summarise_reference

    rounds = _ref_rounds({"a": 5})
    c = _counts([(d, "humans", "rare", 1, 1) for d in range(5)])
    r = summarise_reference(c, rounds, n_boot=50, seed=0, floor=200).iloc[0]
    assert r.suppressed and np.isnan(r.rate)
