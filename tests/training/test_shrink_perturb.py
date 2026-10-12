"""Shrink-and-Perturb: `p <- alpha*p_trained + beta*p_fresh`, parameters only.

The probe it powers (docs/notes/2026-10-10-train-from-random-options-memo.md)
reads "did the perturbed champion climb past the plateau?". That reading is only
sound if the control arm (alpha=1, beta=0) is EXACTLY the champion and the
perturbed arm is exactly the stated blend, so both are pinned here.
"""

import pytest
import torch

from tichu_training.bc.heads import BCModel
from tichu_training.shrink_perturb import shrink_perturb_


def _net(seed: int) -> BCModel:
    torch.manual_seed(seed)
    return BCModel(feature_dim=12, skill_buckets=10, skill_dim=4, trunk_hidden=16,
                   trunk_depth=2, trunk_out_dim=8, head_hidden=8)


def test_identity_blend_leaves_the_net_unchanged():
    trained, fresh = _net(0), _net(1)
    before = {k: v.clone() for k, v in trained.state_dict().items()}
    stats = shrink_perturb_(trained, fresh, alpha=1.0, beta=0.0)
    for k, v in trained.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert stats["rel_change"] == 0.0


def test_every_parameter_is_the_stated_blend():
    trained, fresh = _net(0), _net(1)
    t0 = {k: v.clone() for k, v in trained.named_parameters()}
    f0 = dict(fresh.named_parameters())
    shrink_perturb_(trained, fresh, alpha=0.6, beta=0.4)
    for k, v in trained.named_parameters():
        torch.testing.assert_close(v, 0.6 * t0[k] + 0.4 * f0[k].detach())


def test_fresh_net_is_only_read():
    trained, fresh = _net(0), _net(1)
    f0 = {k: v.clone() for k, v in fresh.state_dict().items()}
    shrink_perturb_(trained, fresh, alpha=0.5, beta=0.5)
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, f0[k]), k


def test_buffers_are_not_blended():
    trained, fresh = torch.nn.BatchNorm1d(4), torch.nn.BatchNorm1d(4)
    trained.running_mean.fill_(3.0)
    shrink_perturb_(trained, fresh, alpha=0.5, beta=0.5)
    assert torch.equal(trained.running_mean, torch.full((4,), 3.0))


def test_rel_change_matches_the_parameter_delta():
    trained, fresh = _net(0), _net(1)
    t0 = torch.cat([p.detach().flatten().clone() for p in trained.parameters()])
    stats = shrink_perturb_(trained, fresh, alpha=0.6, beta=0.4)
    t1 = torch.cat([p.detach().flatten() for p in trained.parameters()])
    assert stats["rel_change"] == pytest.approx(float((t1 - t0).norm() / t0.norm()), rel=1e-4)


def test_architecture_mismatch_is_refused():
    other = BCModel(feature_dim=12, skill_buckets=10, skill_dim=4, trunk_hidden=32,
                    trunk_depth=2, trunk_out_dim=8, head_hidden=8)
    with pytest.raises(ValueError):
        shrink_perturb_(_net(0), other, alpha=0.6, beta=0.4)


@pytest.mark.parametrize("alpha,beta", [(-0.1, 0.4), (1.1, 0.0), (0.6, -0.1)])
def test_out_of_range_coefficients_are_refused(alpha, beta):
    with pytest.raises(ValueError):
        shrink_perturb_(_net(0), _net(1), alpha=alpha, beta=beta)
