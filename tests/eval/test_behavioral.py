"""Behavioral telemetry: engine trick-winner info, per-Round capture, aggregation."""

from tichu_engine.state import deal_for_schupfen
from tichu_eval.behavioral import BehavioralProfile, _fold_round, profile_agent_self_play
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import RoundTelemetry, play_full_round
from tichu_ml.rule_agent import RuleAgent

from tests.eval.test_play_full import ScriptedCaller, _grand_prefixes


def _seeds_positions(n=6):
    return generate_full_position_pool(seed=0, n=n)


def test_telemetry_absent_by_default_present_when_requested():
    start = deal_for_schupfen(seed=0)
    agents = tuple(RuleAgent() for _ in range(4))
    assert play_full_round(agents, start, _grand_prefixes(start)).telemetry is None
    tel = play_full_round(
        agents, start, _grand_prefixes(start), collect_telemetry=True
    ).telemetry
    assert isinstance(tel, RoundTelemetry)


def test_default_result_equality_unaffected_by_telemetry_field():
    # The existing reproducibility contract: two no-telemetry results compare equal.
    start = deal_for_schupfen(seed=9)
    mk = lambda: (ScriptedCaller(grand=True), RuleAgent(), RuleAgent(), RuleAgent())
    pref = _grand_prefixes(start)
    assert play_full_round(mk(), start, pref) == play_full_round(mk(), start, pref)


def test_tricks_won_sum_equals_tricks_total():
    # Every resolved Trick is attributed to exactly one seat (engine `info`).
    for pos in _seeds_positions():
        agents = tuple(RuleAgent() for _ in range(4))
        tel = play_full_round(
            agents, pos.state, pos.grand_prefixes, collect_telemetry=True
        ).telemetry
        assert sum(tel.tricks_won) == tel.tricks_total
        assert tel.tricks_total > 0  # a played-out Round always resolves Tricks


def test_bombs_played_never_exceed_legal_opportunities():
    for pos in _seeds_positions():
        agents = tuple(RuleAgent() for _ in range(4))
        tel = play_full_round(
            agents, pos.state, pos.grand_prefixes, collect_telemetry=True
        ).telemetry
        for seat in range(4):
            assert tel.bombs_played[seat] <= tel.bomb_legal_decisions[seat]


def test_caller_pass_events_never_exceed_opportunities():
    for pos in _seeds_positions():
        agents = tuple(RuleAgent() for _ in range(4))
        tel = play_full_round(
            agents, pos.state, pos.grand_prefixes, collect_telemetry=True
        ).telemetry
        for seat in range(4):
            assert tel.caller_pass_events[seat] <= tel.caller_pass_opportunities[seat]


def test_partner_steal_events_never_exceed_opportunities():
    # Invariant: you cannot overtake more winning-partner tricks than you faced,
    # and the caller subset is bounded by the base opportunities.
    for pos in _seeds_positions():
        agents = tuple(RuleAgent() for _ in range(4))
        tel = play_full_round(
            agents, pos.state, pos.grand_prefixes, collect_telemetry=True
        ).telemetry
        for seat in range(4):
            assert tel.partner_steal_events[seat] <= tel.partner_steal_opportunities[seat]
            assert (
                tel.partner_steal_caller_opportunities[seat]
                <= tel.partner_steal_opportunities[seat]
            )
            assert (
                tel.partner_steal_caller_events[seat]
                <= tel.partner_steal_caller_opportunities[seat]
            )


def test_partner_steal_opportunities_actually_occur():
    # Coverage: across a handful of self-play Rounds the situation arises, so the
    # metric is exercised end-to-end (not silently always zero).
    total_opps = 0
    for pos in _seeds_positions(n=12):
        agents = tuple(RuleAgent() for _ in range(4))
        tel = play_full_round(
            agents, pos.state, pos.grand_prefixes, collect_telemetry=True
        ).telemetry
        total_opps += sum(tel.partner_steal_opportunities)
    assert total_opps > 0


def test_grand_call_attributed_to_calling_seat():
    start = deal_for_schupfen(seed=3)
    agents = (ScriptedCaller(grand=True), RuleAgent(), RuleAgent(), RuleAgent())
    tel = play_full_round(
        agents, start, _grand_prefixes(start), collect_telemetry=True
    ).telemetry
    assert tel.grand_called[0] is True
    assert not any(tel.grand_called[s] for s in (1, 2, 3))


def test_fold_round_derives_rates():
    # Seat 0 called+won Grand, bombed once (1 of 2 legal chances), won 3 of 4 tricks.
    tel = RoundTelemetry(
        grand_called=(True, False, False, False),
        tichu_called=(False, False, False, False),
        first_out=0,
        out_order=(0, 1, 2, 3),       # not a slam (0 and 1 are opponents)
        bombs_played=(1, 0, 0, 0),
        bomb_legal_decisions=(2, 0, 0, 0),
        tricks_won=(3, 1, 0, 0),
        tricks_total=4,
        caller_pass_opportunities=(4, 0, 0, 0),
        caller_pass_events=(1, 0, 0, 0),
    )
    p = BehavioralProfile()
    _fold_round(p, tel, seat=0)
    assert p.seat_rounds == 1
    assert p.grand_call_rate == 1.0
    assert p.grand_success_rate == 1.0       # first_out == 0
    assert p.bomb_when_legal_rate == 0.5     # 1 of 2
    assert p.trick_win_rate == 0.75          # 3 of 4
    assert p.out_first_rate == 1.0
    assert p.slam_rate == 0.0
    assert p.caller_passivity_rate == 0.25   # ceded 1 of 4 winnable caller tricks


def test_fold_round_derives_partner_steal_rate():
    # Seat 0 followed its winning partner with a legal beat 4 times and overtook
    # 3 of them (partner_steal_rate 0.75); of the 2 where partner was a caller it
    # overtook both (partner_steal_caller_rate 1.0 — the sharp blunder).
    tel = RoundTelemetry(
        grand_called=(False,) * 4,
        tichu_called=(False,) * 4,
        first_out=1,
        out_order=(1, 0, 2, 3),
        bombs_played=(0,) * 4,
        bomb_legal_decisions=(0,) * 4,
        tricks_won=(0, 0, 0, 0),
        tricks_total=4,
        caller_pass_opportunities=(0,) * 4,
        caller_pass_events=(0,) * 4,
        partner_steal_opportunities=(4, 0, 0, 0),
        partner_steal_events=(3, 0, 0, 0),
        partner_steal_caller_opportunities=(2, 0, 0, 0),
        partner_steal_caller_events=(2, 0, 0, 0),
    )
    p = BehavioralProfile()
    _fold_round(p, tel, seat=0)
    assert p.partner_steal_rate == 0.75
    assert p.partner_steal_caller_rate == 1.0


def test_fold_round_credits_slam_to_both_winning_seats():
    tel = RoundTelemetry(
        grand_called=(False,) * 4,
        tichu_called=(False,) * 4,
        first_out=0,
        out_order=(0, 2, 1, 3),       # 0 and 2 are partners -> Doppelsieg for team 0
        bombs_played=(0,) * 4,
        bomb_legal_decisions=(0,) * 4,
        tricks_won=(2, 0, 2, 0),
        tricks_total=4,
        caller_pass_opportunities=(0,) * 4,
        caller_pass_events=(0,) * 4,
    )
    for seat, expect in [(0, 1), (2, 1), (1, 0), (3, 0)]:
        p = BehavioralProfile()
        _fold_round(p, tel, seat=seat)
        assert p.slam_for == expect, f"seat {seat}"


def test_profile_agent_self_play_accumulates_all_seat_rounds():
    positions = _seeds_positions(n=5)
    p = profile_agent_self_play(RuleAgent, positions)
    assert p.seat_rounds == 5 * 4                 # 4 seats per Round
    assert 0.0 <= p.trick_win_rate <= 1.0
    assert p.tricks_available > 0
