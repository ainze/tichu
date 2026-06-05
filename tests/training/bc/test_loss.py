"""Masked cross-entropy and multi-head wrapper behaviour."""

import torch

from tichu_training.action_space import ACTION_SPACE_SIZE
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS
from tichu_training.bc.loss import masked_cross_entropy, masked_kl_divergence
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM


def test_kl_is_zero_when_prediction_matches_target():
    # Uniform logits over the legal set => uniform predicted distribution; a uniform
    # target distribution then has zero divergence. The policy-improvement loss must
    # bottom out at 0 when the net already reproduces the visit distribution.
    logits = torch.zeros(1, 4)
    legal = torch.ones(1, 4, dtype=torch.bool)
    target = torch.full((1, 4), 0.25)
    loss = masked_kl_divergence(logits, target, legal, torch.ones(1))
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_kl_ignores_illegal_action_logits():
    # Two logit tensors differing ONLY on illegal positions must give the same loss —
    # the net can put anything on actions the engine forbade; it cannot move the loss.
    legal = torch.tensor([[True, True, False, False]])
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    base = masked_kl_divergence(torch.zeros(1, 4), target, legal, torch.ones(1))
    spiked = torch.zeros(1, 4)
    spiked[0, 2] = spiked[0, 3] = 1000.0  # huge logits on illegal actions
    other = masked_kl_divergence(spiked, target, legal, torch.ones(1))
    assert torch.isclose(base, other, atol=1e-6)


def test_kl_with_one_hot_target_equals_cross_entropy():
    # A one-hot visit distribution is just a hard label; KL must then collapse to the
    # existing masked_cross_entropy (target-entropy term is 0), so the soft-target loss
    # is a strict generalisation of the hard-label one the BC trainer already uses.
    torch.manual_seed(0)
    logits = torch.randn(3, 6)
    legal = torch.tensor([
        [True, True, True, False, False, False],
        [True, False, True, True, False, False],
        [False, True, True, True, True, False],
    ])
    target_idx = torch.tensor([0, 2, 3])
    weight = torch.tensor([1.0, 0.5, 2.0])
    one_hot = torch.zeros(3, 6).scatter_(1, target_idx.unsqueeze(1), 1.0)

    ce = masked_cross_entropy(logits, target_idx, legal, weight)
    kl = masked_kl_divergence(logits, one_hot, legal, weight)
    assert torch.isclose(ce, kl, atol=1e-6)


def test_kl_sample_weight_scales_loss():
    logits = torch.zeros(2, 4)  # uniform pred; non-uniform target => positive KL
    legal = torch.ones(2, 4, dtype=torch.bool)
    target = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.4, 0.2, 0.2, 0.2]])
    base = masked_kl_divergence(logits, target, legal, torch.ones(2))
    doubled = masked_kl_divergence(logits, target, legal, torch.full((2,), 2.0))
    assert base > 0
    assert torch.isclose(doubled, base * 2)


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
    assert HEAD_LOGIT_DIMS["wish"] == 14
    assert HEAD_LOGIT_DIMS["dragon_assignment"] == 2
    # Schupfen is a standalone Schupfen Network per ADR-0012, not a BC head.
    assert "schupfen" not in HEAD_LOGIT_DIMS


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
    # Always in the canonical order play → wish → dragon_assignment.
    # Schupfen is omitted — it lives in a standalone Schupfen Network per ADR-0012.
    assert list(model.heads.keys()) == ["play", "wish", "dragon_assignment"]


def test_bcmodel_accepts_default_dims():
    # Real-world dims: matches the v1 featurizer + action space.
    model = BCModel(feature_dim=FEATURIZER_OUTPUT_DIM, skill_buckets=10, skill_dim=64,
                    trunk_hidden=64, trunk_depth=1, trunk_out_dim=64)
    out = model(
        torch.randn(1, FEATURIZER_OUTPUT_DIM),
        torch.tensor([10], dtype=torch.long),
    )
    assert out["play"].shape == (1, ACTION_SPACE_SIZE)
