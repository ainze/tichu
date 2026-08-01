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
from tichu_training.ppo.rollout import _supports_wish
from tichu_training.ppo.rollout_parallel import (
    ParallelRollout,
    _init_worker,
    _league_opponent,
    save_rollout_weights,
)

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


def test_league_opponent_honors_train_wish(tmp_path):
    # The Mahjong-wish is part of the policy (ADR-0034 addendum): when the run
    # co-trains the wish, a league opponent must wish via its own head exactly like
    # the learner — falling back to the frozen always-decline seat agent would be a
    # train/eval mismatch (the eval master wishes via its BC head too).
    torch.manual_seed(0)
    models = _build_models(_ARCH)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    opp_w = tmp_path / "opp.pt"
    save_rollout_weights(str(opp_w), models, critic)

    _init_worker(_ARCH, 16, 1, True)  # (arch, critic_hidden, critic_depth, perfect_info)
    wishing = _league_opponent(str(opp_w), skill_decile=9, seed=0, train_wish=True)
    declining = _league_opponent(str(opp_w), skill_decile=9, seed=0, train_wish=False)
    assert _supports_wish(wishing)
    assert not _supports_wish(declining)


def test_worker_task_carries_skip_forced_play_across_the_spawn_boundary(tmp_path):
    # The spawn pool ships a RolloutTask verbatim to the worker. Running the
    # worker body in-process (the pool itself needs spawned children) proves the
    # flag survives the hand-off and actually suppresses forced rows there.
    from tichu_training.ppo.rollout_parallel import RolloutTask, _rollout_chunk

    torch.manual_seed(0)
    models = _build_models(_ARCH)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    weights = tmp_path / "w.pt"
    save_rollout_weights(str(weights), models, critic)
    _init_worker(_ARCH, 16, 1, True)

    positions = generate_full_position_pool(seed=0, n=4)

    def run(skip: bool):
        return _rollout_chunk(
            RolloutTask(
                positions=positions, weights_path=str(weights),
                opp_weights_path=None, learner_team=0, skill_decile=9, seed=0,
                train_wish=False, skip_forced_play=skip,
            )
        )

    baseline = run(False)
    skipped = run(True)

    def play_rows(trajs):
        return sum(
            1 for t in trajs for s in t.steps if s.decision_type == "play"
        )

    assert play_rows(skipped) < play_rows(baseline)
    assert play_rows(skipped) > 0
    # Forced rows are exactly the single-legal ones; none may survive.
    for t in skipped:
        for s in t.steps:
            if s.decision_type == "play" and s.legal_mask is not None:
                assert int(s.legal_mask.sum()) > 1
