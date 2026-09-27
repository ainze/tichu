"""Behavioral Drift Benchmark — Fixed-Opponent arms.

Subject arm: the Subject team (both partner seats) vs a BC opponent team, each
Pool deal played twice with Seat-Swap. BC arm: BC+BC vs BC+BC — deterministic, so
both Seat-Swap halves are the same game; it is played once and read from both
teams. See CONTEXT.md §"Behavioral Drift Benchmark".
"""

from functools import partial

import pandas as pd

from tichu_eval.drift_arms import run_drift_arms
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_ml.rule_agent import RuleAgent

from tests.eval.test_play_full import ScriptedCaller

# The "Subject" differs from the "BC" only by always calling Tichu, so which
# seats it sat in is visible in the outcome (only its seats can ever call).
SUBJECT = partial(ScriptedCaller, tichu=True)
BC = RuleAgent


def _log(n=8, **kw):
    return run_drift_arms(SUBJECT, BC, generate_full_position_pool(seed=0, n=n), **kw)


def test_both_arms_cover_every_deal_in_both_halves():
    log = _log()
    counts = log.rounds.groupby(["arm", "half"]).size().to_dict()
    assert counts == {("bc", 0): 8, ("bc", 1): 8, ("subject", 0): 8, ("subject", 1): 8}


def test_subject_seats_flip_between_halves_in_the_subject_arm():
    log = _log()
    subj = log.rounds[log.rounds.arm == "subject"]
    for _, r in subj.iterrows():
        seats = (0, 2) if r.half == 0 else (1, 3)
        assert r.subject_team == seats[0] % 2
        assert set(r.tichu_callers) <= set(seats)   # only the Subject ever calls
    rows = log.decisions[log.decisions.arm == "subject"]
    assert (rows.is_subject == (rows.seat % 2 == rows.half)).all()


def test_bc_arm_reads_one_game_from_both_teams():
    log = _log()
    bc = log.rounds[log.rounds.arm == "bc"]
    assert not any(len(t) for t in bc.tichu_callers)   # BC never calls here
    h0 = bc[bc.half == 0].set_index("deal")
    h1 = bc[bc.half == 1].set_index("deal")
    assert (h0.total_0 == h1.total_0).all() and (h0.out_order == h1.out_order).all()
    assert (h0.subject_team == 0).all() and (h1.subject_team == 1).all()
    d = log.decisions[log.decisions.arm == "bc"]
    assert (d.is_subject == (d.seat % 2 == d.half)).all()


def test_bc_arm_shortcut_equals_actually_replaying_the_swap():
    # The one-game shortcut is only valid for deterministic agents: guard it by
    # replaying half 1 for real and demanding the identical game.
    from tichu_eval.drift_recorder import record_round

    positions = generate_full_position_pool(seed=0, n=8)
    log = run_drift_arms(SUBJECT, BC, positions)
    bc = log.rounds[(log.rounds.arm == "bc") & (log.rounds.half == 1)].set_index("deal")
    for i, pos in enumerate(positions):
        real = record_round(tuple(BC() for _ in range(4)), pos, deal=i, half=1,
                            subject_seats=(1, 3))
        assert real.round["out_order"] == bc.loc[i, "out_order"]
        assert real.round["total_1"] == bc.loc[i, "total_1"]


def test_parallel_run_equals_serial_run():
    positions = generate_full_position_pool(seed=0, n=10)
    serial = run_drift_arms(SUBJECT, BC, positions)
    parallel = run_drift_arms(SUBJECT, BC, positions, workers=3)
    key_d = ["arm", "deal", "half"]
    pd_eq = lambda a, b, k: a.sort_values(k, kind="stable").reset_index(drop=True).equals(
        b.sort_values(k, kind="stable").reset_index(drop=True))
    assert pd_eq(serial.rounds, parallel.rounds, key_d)
    assert pd_eq(serial.decisions, parallel.decisions, key_d)


def test_aa_null_run_plays_bc_on_two_disjoint_deal_sets():
    from tichu_eval.drift_arms import run_aa_null

    # Pool seed s deals `s + i`, so a disjoint second set starts at s + n.
    a = generate_full_position_pool(seed=0, n=6)
    b = generate_full_position_pool(seed=6, n=6)
    log = run_aa_null(BC, a, b)
    assert set(log.rounds.arm) == {"bc", "subject"}
    ref_a = run_drift_arms(BC, BC, a).rounds
    ref_b = run_drift_arms(BC, BC, b).rounds
    cols = ["deal", "half", "total_0", "total_1"]
    got_bc = log.rounds[log.rounds.arm == "bc"][cols].reset_index(drop=True)
    got_s = log.rounds[log.rounds.arm == "subject"][cols].reset_index(drop=True)
    assert got_bc.equals(ref_a[ref_a.arm == "bc"][cols].reset_index(drop=True))
    assert got_s.equals(ref_b[ref_b.arm == "bc"][cols].reset_index(drop=True))


def test_aa_null_run_finds_no_drift():
    # End-to-end smoke of the acceptance gate: the same policy on two deal sets
    # must produce no Benjamini–Hochberg survivors and CIs that mostly cover 0.
    from tichu_eval.drift_arms import run_aa_null
    from tichu_eval.drift_metrics import summarise

    log = run_aa_null(SUBJECT, generate_full_position_pool(seed=0, n=150),
                      generate_full_position_pool(seed=150, n=150))
    s = summarise(log, n_boot=300, seed=0, floor=50)
    d = s[(s.arm == "delta") & ~s.suppressed]
    assert len(d) >= 20
    assert not d.bh_survives.any()
    covers = (d.rate_lo <= 0) & (d.rate_hi >= 0)
    assert covers.mean() >= 0.9


def test_a_saved_log_reloads_to_the_same_summary(tmp_path):
    # Re-analysis without replay is the point of logging: a log read back from
    # Parquet must summarise exactly like the in-memory one.
    from tichu_eval.drift_arms import load_drift_log, save_drift_log
    from tichu_eval.drift_metrics import summarise

    log = run_drift_arms(SUBJECT, BC, generate_full_position_pool(seed=0, n=12))
    save_drift_log(log, tmp_path)
    back = load_drift_log(tmp_path)
    a = summarise(log, n_boot=50, seed=0, floor=0)
    b = summarise(back, n_boot=50, seed=0, floor=0)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
