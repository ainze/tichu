"""Freezing call/schupfen nets in co-train (play-only isolation).

`freeze_nets` holds named nets at their warm-started BC weights: they're still
driven in the rollout (full stack plays at BC quality) but excluded from the
optimizer, so only the remaining nets + the shared critic learn. The hypothesis
it serves: co-training all four nets lets the call/schupfen policies drift, making
play's environment non-stationary; freezing them lets play train against a fixed
teammate. These tests pin the mechanism — a frozen net does NOT move even though
its loss is still computed, while play and the critic do.
"""

import copy

import pytest
import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models, _build_optimizer
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.cotrain import (
    BatchedCoTrainPolicy,
    build_cotrain_batch,
    cotrain_update,
)
from tichu_training.ppo.rollout import collect_rollout

_ARCH = {
    "model": dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
    "schupfen_model": {"skill_dim": 8, "hidden": 16},
    "call_model": {"skill_dim": 8, "hidden": 16},
}
_FROZEN = ["schupfen", "tichu", "grand"]


def _snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


def _changed(model, before) -> bool:
    return any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


def _one_update(models, critic, optimizer):
    positions = generate_full_position_pool(seed=0, n=6)
    policy = BatchedCoTrainPolicy(
        models["play"], models["schupfen"], models["tichu"], models["grand"],
        critic, skill_decile=9, perfect_info=True,
    )
    trajs = collect_rollout(positions, policy, learner_team=0)
    batch = build_cotrain_batch(trajs, skill_decile=9, gamma=1.0, lam=0.95)
    bc = {dt: copy.deepcopy(models[dt]) for dt in _NET_TYPES}
    cotrain_update(
        models, bc, critic, optimizer, batch,
        clip_eps=0.1, vf_coef=0.5,
        ent_coefs={dt: 0.01 for dt in _NET_TYPES},
        kl_coefs={dt: 1.0 for dt in _NET_TYPES}, epochs=2,
    )
    return batch


def test_build_optimizer_excludes_frozen_nets():
    models = _build_models(_ARCH)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    _opt, trainable = _build_optimizer(
        models, critic, policy_lr=1e-2, critic_lr=1e-2, freeze_nets=_FROZEN)
    assert trainable == ["play"]
    for dt in _FROZEN:
        assert all(not p.requires_grad for p in models[dt].parameters())
    assert all(p.requires_grad for p in models["play"].parameters())


def test_frozen_nets_do_not_move_play_and_critic_do():
    torch.manual_seed(0)
    models = _build_models(_ARCH)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    optimizer, _ = _build_optimizer(
        models, critic, policy_lr=1e-2, critic_lr=1e-2, freeze_nets=_FROZEN)

    before = {dt: _snapshot(models[dt]) for dt in _NET_TYPES}
    crit_before = _snapshot(critic)
    batch = _one_update(models, critic, optimizer)

    # Sanity: the schupfen net actually had steps in the batch, so "unchanged" is a
    # real freeze, not a vacuous "no data" pass (every game has a Schupfen phase).
    assert "schupfen" in batch.nets and batch.nets["schupfen"].features.shape[0] > 0

    for dt in _FROZEN:
        assert not _changed(models[dt], before[dt]), f"{dt} moved but should be frozen"
    assert _changed(models["play"], before["play"]), "play should have trained"
    assert _changed(critic, crit_before), "critic should have trained"


def test_default_trains_all_nets():
    """No freeze_nets => the existing behaviour: all four nets trainable."""
    models = _build_models(_ARCH)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    _opt, trainable = _build_optimizer(models, critic, policy_lr=1e-2, critic_lr=1e-2)
    assert trainable == list(_NET_TYPES)


def test_cannot_freeze_play():
    models = _build_models(_ARCH)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    with pytest.raises(ValueError, match="play"):
        _build_optimizer(models, critic, policy_lr=1e-2, critic_lr=1e-2, freeze_nets=["play"])
