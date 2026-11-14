"""Full-State Value + Value Search (test build).

`resume_round` must be the same game as `play_full_round`, or every label and
every search leaf is about a different game. The override rule must not move on
noise — a plain argmax over per-candidate means is what cost piKL 58 a round.
"""

import random

import numpy as np

from tichu_engine.legality import legal_actions_for
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_ml.rule_agent import RuleAgent
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.search.belief_chooser import asked_tichu_from_public
from tichu_training.search.full_state_value import (
    emit_deal,
    resume_round,
    round_totals,
)
from tichu_training.search.value_search import (
    ValueSearchAgent,
    pick_action,
    pick_action_mixed,
)


def _record(agents, position):
    seen = []
    result = play_full_round(
        agents, position.state, position.grand_prefixes,
        state_observer=lambda seat, gs, action: seen.append((seat, gs, action)))
    return result, seen


def test_resuming_from_any_decision_reproduces_the_round():
    agents = [RuleAgent()] * 4
    for position in generate_full_position_pool(seed=11, n=3):
        result, seen = _record(agents, position)
        for seat, gs, action in seen[:: max(1, len(seen) // 8)]:
            state, done = resume_round(
                agents, gs, forced_action=action,
                asked_tichu=asked_tichu_from_public(gs.public))
            assert done
            assert round_totals(state.public.scores,
                                position.state.public.scores) == result.total


def test_depth_zero_stops_at_the_next_recorded_decision():
    agents = [RuleAgent()] * 4
    position = generate_full_position_pool(seed=5, n=1)[0]
    _result, seen = _record(agents, position)
    for (seat, gs, action), (_s, next_gs, _a) in zip(seen, seen[1:]):
        state, done = resume_round(
            agents, gs, forced_action=action,
            asked_tichu=asked_tichu_from_public(gs.public), max_play_decisions=0)
        assert not done
        assert state.hands == next_gs.hands
        assert state.public.current_player == next_gs.public.current_player


def test_emit_deal_labels_are_exact_and_the_pair_points_at_its_rows():
    agents = [RuleAgent()] * 4
    position = generate_full_position_pool(seed=21, n=1)[0]
    result, _seen = _record(agents, position)
    out = emit_deal(agents, position, random.Random(0), rows_per_traj=6)
    rows, pair = out["rows"], out["pair"]
    assert rows and all(r[0].shape == (PERFECT_INFO_DIM,) for r in rows)
    spread = abs(result.total[0] - result.total[1]) / 100.0
    assert all(abs(abs(r[1]) - spread) < 1e-6 for r in rows if r[2] == 0)
    assert pair is not None
    for key in ("base_row", "alt_row"):
        assert -1 <= pair[key] < len(rows)
    if pair["alt_row"] >= 0:
        assert rows[pair["alt_row"]][2] == 1
        assert abs(rows[pair["alt_row"]][1] - pair["alt_sign"] * pair["z_alt"]) < 1e-6


def test_policy_choice_stands_unless_an_alternative_clears_both_bars():
    base = np.linspace(-80.0, 80.0, 20)
    lottery = np.full(20, -10.0)
    lottery[0] = 790.0  # mean +30, carried by one world
    values = np.stack([
        base,
        base + 3.0 + (np.arange(20) % 2),  # solid but tiny
        base + lottery,                    # big mean, not solid
    ])
    assert pick_action(values, delta_threshold=10.0, t_threshold=2.5) == 0


def test_the_best_clearing_alternative_is_played():
    base = np.zeros(20)
    values = np.stack([base, base + 15.0 + np.arange(20) % 3, base + 40.0 + np.arange(20) % 5])
    assert pick_action(values, delta_threshold=10.0, t_threshold=2.5) == 2


def test_emit_siblings_labels_each_candidate_with_its_own_exact_outcome():
    from tichu_training.search.full_state_value import emit_siblings

    agents = [RuleAgent()] * 4
    position = generate_full_position_pool(seed=21, n=1)[0]
    result, _seen = _record(agents, position)
    out = emit_siblings(agents, position, random.Random(0), decisions=3, top_k=2, plies=3)
    rows, cands = out["rows"], out["cands"]
    assert cands and rows
    spread = abs(result.total[0] - result.total[1]) / 100.0
    zd = {(d, c): z for d, c, z, _n, _p in cands}
    assert all(abs(abs(z) - spread) < 1e-6 for (d, c), z in zd.items() if c == 0)
    for feats, z, d, c, ply, sign, _ph in rows:
        assert feats.shape == (PERFECT_INFO_DIM,) and 0 <= ply < 3
        assert abs(z - sign * zd[(d, c)]) < 1e-6
    assert sum(n for _d, _c, _z, n, _p in cands) == len(rows)


def test_decision_targets_score_the_chosen_move_with_the_round_it_led_to():
    """Runs only where the champion export is on disk (it needs a real MLAgent
    for the policy's own inputs and action indices)."""
    import pathlib

    import pytest

    export = pathlib.Path(r"C:\workbench\tichu\data\runs\cotrain_v7_gated\export\iter_15360")
    if not export.exists():
        pytest.skip("champion export not on disk")
    from tichu_training.ppo.pmcpa import build_export_agent
    from tichu_training.search.search_distill import emit_decision_targets

    agent = build_export_agent(export_dir=export)
    position = generate_full_position_pool(seed=4, n=1)[0]
    result, _seen = _record([agent] * 4, position)
    items = emit_decision_targets(agent, position, random.Random(0), decisions=3, top_k=3)
    assert items
    spread = abs(result.total[0] - result.total[1]) / 100.0
    for item in items:
        n = int((item["cand"] >= 0).sum())
        assert n >= 2 and len(set(item["cand"][:n].tolist())) == n
        assert all(item["mask"][item["cand"][:n]])
        assert abs(abs(item["q"][0]) - spread) < 1e-6
        assert np.isfinite(item["q"][:n]).all() and np.isnan(item["q"][n:]).all()


class _ConstantValue:
    def predict(self, features):
        return np.zeros(len(features))


def test_an_uninformative_value_net_leaves_play_unchanged():
    """With a flat value net no alternative can clear the bar on leaf scores
    alone, and exact terminal outcomes still flow through."""
    policy = RuleAgent()
    agent = ValueSearchAgent(policy, _ConstantValue(), worlds=4, top_k=2, depth=0,
                             delta_threshold=10.0, t_threshold=2.5)
    position = generate_full_position_pool(seed=3, n=1)[0]
    _result, seen = _record([policy] * 4, position)
    checked = 0
    for seat, gs, action in seen:
        view = gs.private_view(seat)
        if len(legal_actions_for(view)) > 1 and len(gs.hands[seat]) > 6:
            assert agent.act(view) == action
            checked += 1
            if checked == 5:
                break
    assert checked


def test_mixed_rule_lets_value_break_policy_ties_but_not_overrule_a_strong_preference():
    values = np.stack([np.zeros(20), np.full(20, 30.0)])
    assert pick_action_mixed(values, np.log([0.5, 0.4]), tau=50.0) == 1
    assert pick_action_mixed(values, np.log([0.9, 0.05]), tau=50.0) == 0
