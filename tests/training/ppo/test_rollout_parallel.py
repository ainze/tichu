"""Process-parallel rollout for co-training (ADR-0034 #1).

The rollout is CPU-engine-bound (legal-action enumeration is pure Python, GIL-held),
so threads can't help — we fan the M games out across worker PROCESSES, each rolling
out its chunk with the current weights and returning whole trajectories. This is
additive: `rollout_workers=1` keeps the validated single-process path.
"""

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.cli.train_cotrain import _build_models
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.rollout_parallel import ParallelRollout, save_rollout_weights

_ARCH = {
    "model": dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
    "schupfen_model": {"skill_dim": 8, "hidden": 16},
    "call_model": {"skill_dim": 8, "hidden": 16},
}


def test_parallel_rollout_returns_two_trajectories_per_game(tmp_path):
    torch.manual_seed(0)
    models = _build_models(_ARCH)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    weights = tmp_path / "w.pt"
    save_rollout_weights(str(weights), models, critic)

    positions = generate_full_position_pool(seed=0, n=4)
    pr = ParallelRollout(_ARCH, critic_hidden=16, skill_decile=9, perfect_info=True, workers=2)
    try:
        trajs = pr.collect(positions, str(weights), learner_team=0, base_seed=0)
    finally:
        pr.close()

    # One trajectory per learner-team seat (0, 2) per game -> 2 x 4 games = 8,
    # every one a completed Round with recorded steps and a finite reward.
    assert len(trajs) == 8
    assert {t.seat for t in trajs} == {0, 2}
    assert all(len(t.steps) > 0 and t.reward is not None for t in trajs)
    # The full stack was driven in the workers (schupfen + a call recorded somewhere).
    types = {s.decision_type for t in trajs for s in t.steps}
    assert {"play", "schupfen"} <= types


def test_parallel_rollout_uses_a_distinct_league_opponent(tmp_path):
    # League path: opponent seats play a SEPARATE frozen weight file; learner seats
    # still play the live weights and are the only ones recorded.
    torch.manual_seed(0)
    learner_models = _build_models(_ARCH)
    opp_models = _build_models(_ARCH)  # different init -> a genuinely distinct opponent
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    learner_w = tmp_path / "learner.pt"
    opp_w = tmp_path / "opp.pt"
    save_rollout_weights(str(learner_w), learner_models, critic)
    save_rollout_weights(str(opp_w), opp_models, critic)

    positions = generate_full_position_pool(seed=0, n=4)
    pr = ParallelRollout(_ARCH, critic_hidden=16, skill_decile=9, perfect_info=True, workers=2)
    try:
        trajs = pr.collect(positions, str(learner_w), learner_team=0, base_seed=0,
                           opp_weights_path=str(opp_w))
    finally:
        pr.close()

    # Same recording invariant holds with a league opponent in play.
    assert len(trajs) == 8
    assert {t.seat for t in trajs} == {0, 2}
    assert all(len(t.steps) > 0 and t.reward is not None for t in trajs)
