"""Behavioral Drift Benchmark — metric definitions and the panel summary.

Each metric is a pure function of the `DriftLog`, tested here on small hand-built
logs whose answer is known. See CONTEXT.md §"Behavioral Drift Benchmark".
"""

from functools import partial

import pandas as pd

from tichu_eval.drift_arms import DriftLog, run_drift_arms
from tichu_eval.drift_metrics import METRICS, metric, summarise
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_ml.rule_agent import RuleAgent

from tests.eval.test_play_full import ScriptedCaller


def _play(arm, deal, *, holder, action, has_beat=True, forced=False, subject=True, **kw):
    return {"arm": arm, "deal": deal, "half": 0, "seat": 0, "kind": "play",
            "is_subject": subject, "holder": holder, "action": action,
            "has_beat": has_beat, "forced": forced, **kw}


def _log(decisions=(), rounds=()):
    return DriftLog(decisions=pd.DataFrame(list(decisions)), rounds=pd.DataFrame(list(rounds)))


def _cells(counts):
    """{(arm, cell): (events, opportunities)} summed over deals."""
    g = counts.groupby(["arm", "cell"])[["events", "opportunities"]].sum()
    return {k: tuple(int(x) for x in v) for k, v in g.iterrows()}


def test_pass_despite_beat_splits_by_trick_holder_and_skips_forced_and_opponents():
    log = _log(decisions=[
        _play("bc", 0, holder="opponent", action="pass"),
        _play("bc", 0, holder="opponent", action="single"),
        _play("bc", 1, holder="partner", action="pass"),
        _play("bc", 1, holder="partner", action="pass", forced=True),      # forced: out
        _play("bc", 1, holder="opponent", action="pass", has_beat=False),  # no beat: out
        _play("bc", 1, holder="none", action="single"),                    # a lead: out
        _play("bc", 1, holder="opponent", action="pass", subject=False),   # BC seat: out
        _play("subject", 0, holder="opponent", action="single"),
    ])
    got = _cells(metric("pass_despite_beat").fn(log))
    assert got == {("bc", "opponent"): (1, 2), ("bc", "partner"): (1, 1),
                   ("subject", "opponent"): (0, 1)}


def test_slam_for_and_against_are_read_from_the_subject_team():
    rounds = [
        {"arm": "bc", "deal": 0, "half": 0, "subject_team": 0, "slam_team": 0},
        {"arm": "bc", "deal": 0, "half": 1, "subject_team": 1, "slam_team": 0},
        {"arm": "subject", "deal": 0, "half": 0, "subject_team": 0, "slam_team": None},
        {"arm": "subject", "deal": 0, "half": 1, "subject_team": 1, "slam_team": 1},
    ]
    got = _cells(metric("slam").fn(_log(rounds=rounds)))
    assert got == {("bc", "for"): (1, 2), ("bc", "against"): (1, 2),
                   ("subject", "for"): (1, 2), ("subject", "against"): (0, 2)}


def test_summarise_reports_every_metric_with_a_bh_flag():
    log = run_drift_arms(partial(ScriptedCaller, tichu=True), RuleAgent,
                         generate_full_position_pool(seed=0, n=12))
    s = summarise(log, n_boot=100, seed=0, floor=0)
    assert set(s.metric) == {m.name for m in METRICS} | {"game"}
    assert set(s.arm) == {"bc", "subject", "delta"}
    assert {"family", "bh_survives", "rate", "rate_lo", "rate_hi"} <= set(s.columns)
    assert not s[s.arm != "delta"].bh_survives.any()   # the flag lives on Δ rows only


# --- Distributions share one denominator --------------------------------------

def test_a_distribution_divides_every_category_by_all_decisions():
    lead = lambda arm, deal, action: _play(arm, deal, holder="none", action=action)
    log = _log(decisions=[lead("bc", 0, "single"), lead("bc", 0, "single"),
                          lead("bc", 1, "pair"), lead("subject", 0, "dog")])
    got = _cells(metric("lead_type").fn(log))
    # Every category any arm used appears for both arms, over that arm's 3 / 1 leads.
    assert got[("bc", "single")] == (2, 3) and got[("bc", "pair")] == (1, 3)
    assert got[("bc", "dog")] == (0, 3)
    assert got[("subject", "dog")] == (1, 1) and got[("subject", "single")] == (0, 1)


# --- Metrics that join Decisions to outcomes -----------------------------------

def _round(arm, deal, half, *, out_order, subject_team=0, **kw):
    return {"arm": arm, "deal": deal, "half": half, "subject_team": subject_team,
            "out_order": out_order, **kw}


def test_caller_overtake_then_fail_is_a_round_level_two_by_two():
    opp = lambda arm, deal, half, seat, action: {
        **_play(arm, deal, holder="partner", action=action, partner_called=True),
        "half": half, "seat": seat}
    log = _log(
        decisions=[
            # Round (bc, 0, 0), seat 0: overtook once and yielded once -> "overtook".
            opp("bc", 0, 0, 0, "single"), opp("bc", 0, 0, 0, "pass"),
            # Round (bc, 0, 1), seat 2: always yielded.
            opp("bc", 0, 1, 2, "pass"), opp("bc", 0, 1, 2, "pass"),
        ],
        rounds=[_round("bc", 0, 0, out_order=(1, 2, 3)),     # partner 2 not first: failed
                _round("bc", 0, 1, out_order=(0, 1, 3))],    # partner 0 first: made it
    )
    got = _cells(metric("caller_overtake_then_fail").fn(log))
    assert got == {("bc", "overtook"): (1, 1), ("bc", "yielded"): (0, 1)}


def test_wish_given_to_an_opponent_joins_the_seat_s_own_schupfen():
    sch = {"arm": "bc", "deal": 0, "half": 0, "seat": 0, "kind": "schupfen",
           "is_subject": True, "give_next": "7", "give_partner": "A", "give_previous": "3"}
    wish = lambda rank, deal=0: {"arm": "bc", "deal": deal, "half": 0, "seat": 0,
                                 "kind": "wish", "is_subject": True, "wish": rank}
    sch1 = {**sch, "deal": 1}
    sch2 = {**sch, "deal": 2}
    log = _log(decisions=[sch, wish("7"), sch1, wish("A", deal=1), sch2, wish("none", deal=2)])
    got = _cells(metric("wish_gave_opponent").fn(log))
    assert got == {("bc", "all"): (1, 2)}   # "A" went to the partner; "none" is no wish


def test_wish_fulfilled_is_read_from_the_subject_s_wishing_rounds():
    wish = lambda deal, seat, rank: {"arm": "bc", "deal": deal, "half": 0, "seat": seat,
                                     "kind": "wish", "is_subject": seat in (0, 2), "wish": rank}
    log = _log(
        decisions=[wish(0, 0, "7"), wish(1, 2, "9"), wish(2, 1, "5")],
        rounds=[_round("bc", 0, 0, out_order=(), wish_fulfilled=True),
                _round("bc", 1, 0, out_order=(), wish_fulfilled=False),
                _round("bc", 2, 0, out_order=(), wish_fulfilled=True)],   # BC seat's wish
    )
    assert _cells(metric("wish_fulfilled").fn(log)) == {("bc", "all"): (1, 2)}


def test_out_position_counts_each_subject_seat_once_per_round():
    log = _log(rounds=[
        _round("bc", 0, 0, out_order=(0, 1, 3)),          # seat 0 1st, seat 2 4th
        _round("bc", 0, 1, subject_team=1, out_order=(1, 3)),   # Slam: 1st and 2nd
    ])
    got = _cells(metric("out_position").fn(log))
    assert got[("bc", "1st")] == (2, 4) and got[("bc", "2nd")] == (1, 4)
    assert got[("bc", "last")] == (1, 4) and got[("bc", "3rd")] == (0, 4)


def test_every_metric_runs_on_a_real_log():
    log = run_drift_arms(partial(ScriptedCaller, tichu=True), RuleAgent,
                         generate_full_position_pool(seed=0, n=10))
    for m in METRICS:
        counts = m.fn(log)
        assert list(counts.columns[:5]) == ["deal", "arm", "cell", "events", "opportunities"], m.name
        assert (counts.events <= counts.opportunities).all() or not m.conditional, m.name


def test_phoenix_combination_when_legal_counts_only_decisions_where_one_was_legal():
    log = _log(decisions=[
        _play("bc", 0, holder="none", action="pair", plays_phoenix=True, phoenix_combo_legal=True),
        _play("bc", 0, holder="none", action="single", plays_phoenix=True, phoenix_combo_legal=True),
        _play("bc", 0, holder="none", action="pair", plays_phoenix=False, phoenix_combo_legal=True),
        _play("bc", 0, holder="none", action="single", plays_phoenix=True,
              phoenix_combo_legal=False),                                  # none legal: out
        _play("bc", 0, holder="none", action="pair", plays_phoenix=True,
              phoenix_combo_legal=True, forced=True),                      # forced: out
    ])
    got = _cells(metric("phoenix_combination_when_legal").fn(log))
    assert got == {("bc", "all"): (1, 3)}
