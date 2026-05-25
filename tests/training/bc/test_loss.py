"""Masked cross-entropy and multi-head wrapper behaviour."""

import torch

from tichu_training.action_space import ACTION_SPACE_SIZE
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS
from tichu_training.bc.loss import masked_cross_entropy
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM


def test_masked_loss_zero_grad_on_illegal_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, requires_grad=True)
    legal = torch.tensor([
        [True, True, False, False, True, False, False, False],
        [False, False, True, True, False, False, False, False],
    ])
    target = torch.tensor([0, 2], dtype=torch.long)
    weight = torch.tensor([1.0, 1.0])
    loss = masked_cross_entropy(logits, target, legal, weight)
    loss.backward()
    assert logits.grad is not None
    # Gradient on masked positions must be exactly zero.
    masked_grad = logits.grad[~legal]
    assert torch.equal(masked_grad, torch.zeros_like(masked_grad))


def test_masked_loss_returns_scalar_with_grad():
    logits = torch.zeros(2, 4, requires_grad=True)
    legal = torch.ones(2, 4, dtype=torch.bool)
    target = torch.tensor([0, 1])
    weight = torch.ones(2)
    loss = masked_cross_entropy(logits, target, legal, weight)
    assert loss.dim() == 0
    assert loss.requires_grad


def test_sample_weight_scales_loss():
    logits = torch.zeros(2, 4)
    legal = torch.ones(2, 4, dtype=torch.bool)
    target = torch.tensor([0, 1])
    base = masked_cross_entropy(logits, target, legal, torch.ones(2))
    doubled = masked_cross_entropy(logits, target, legal, torch.full((2,), 2.0))
    assert torch.isclose(doubled, base * 2)


def test_head_logit_dims_match_spec():
    assert HEAD_LOGIT_DIMS["play"] == ACTION_SPACE_SIZE
    assert HEAD_LOGIT_DIMS["schupfen"] == 3
    assert HEAD_LOGIT_DIMS["wish"] == 14
    assert HEAD_LOGIT_DIMS["dragon_assignment"] == 2


def test_bcmodel_forward_returns_all_heads():
    # Use a tiny trunk for speed.
    model = BCModel(
        feature_dim=32,
        skill_buckets=10,
        skill_dim=4,
        trunk_hidden=16,
        trunk_depth=1,
        trunk_out_dim=8,
    )
    features = torch.randn(2, 32)
    skill = torch.tensor([0, 10], dtype=torch.long)  # one rated, one neutral
    out = model(features, skill)
    assert set(out.keys()) == set(HEAD_LOGIT_DIMS.keys())
    for name, dim in HEAD_LOGIT_DIMS.items():
        assert out[name].shape == (2, dim)


def test_bcmodel_heads_have_deterministic_order():
    model = BCModel(feature_dim=8, skill_buckets=10, skill_dim=2,
                    trunk_hidden=4, trunk_depth=1, trunk_out_dim=4)
    # Always in the canonical order play → schupfen → wish → dragon_assignment.
    assert list(model.heads.keys()) == ["play", "schupfen", "wish", "dragon_assignment"]


def test_bcmodel_accepts_default_dims():
    # Real-world dims: matches the v1 featurizer + action space.
    model = BCModel(feature_dim=FEATURIZER_OUTPUT_DIM, skill_buckets=10, skill_dim=64,
                    trunk_hidden=64, trunk_depth=1, trunk_out_dim=64)
    out = model(
        torch.randn(1, FEATURIZER_OUTPUT_DIM),
        torch.tensor([10], dtype=torch.long),
    )
    assert out["play"].shape == (1, ACTION_SPACE_SIZE)
