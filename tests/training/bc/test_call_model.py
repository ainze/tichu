"""CallNetwork shape and parameter checks."""

import torch

from tichu_training.bc.call_model import (
    CallNetwork,
    GrandTichuCallNetwork,
    TichuCallNetwork,
)


def test_call_network_forward_shape():
    net = CallNetwork(feature_dim=64, skill_buckets=10, skill_dim=8, hidden=32)
    out = net(torch.randn(4, 64), torch.tensor([0, 10, 5, 9], dtype=torch.long))
    assert out.shape == (4, 2)


def test_call_network_is_differentiable():
    net = CallNetwork(feature_dim=16, skill_dim=4, hidden=8)
    x = torch.randn(2, 16, requires_grad=True)
    out = net(x, torch.tensor([0, 10], dtype=torch.long))
    out.sum().backward()
    assert x.grad is not None


def test_neutral_skill_row_used_for_cold_start():
    torch.manual_seed(0)
    net = CallNetwork(feature_dim=8, skill_dim=4, hidden=8)
    features = torch.randn(1, 8)
    # Two cold-start calls should be identical.
    a = net(features, torch.tensor([10], dtype=torch.long))
    b = net(features, torch.tensor([10], dtype=torch.long))
    assert torch.equal(a, b)


def test_grand_and_regular_have_independent_weights():
    torch.manual_seed(0)
    grand = GrandTichuCallNetwork(feature_dim=8, skill_dim=2, hidden=4)
    torch.manual_seed(0)
    regular = TichuCallNetwork(feature_dim=8, skill_dim=2, hidden=4)
    # Same init seed → identical weights, but distinct module instances.
    for p1, p2 in zip(grand.parameters(), regular.parameters()):
        assert torch.equal(p1, p2)
    # Mutating one must not touch the other.
    with torch.no_grad():
        for p in grand.parameters():
            p.add_(1.0)
    assert any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(grand.parameters(), regular.parameters())
    )


def test_module_classes_distinct():
    assert GrandTichuCallNetwork is not TichuCallNetwork
    assert issubclass(GrandTichuCallNetwork, CallNetwork)
    assert issubclass(TichuCallNetwork, CallNetwork)


def test_default_is_non_residual_and_keys_unchanged():
    """`residual=False` (default) keeps the original `net.*` layout so existing
    checkpoints still load."""
    net = CallNetwork(feature_dim=16, skill_dim=4, hidden=8)
    assert net.residual is False
    keys = set(net.state_dict())
    assert any(k.startswith("net.") for k in keys)
    assert not any(k.startswith("input_proj") or k.startswith("blocks") for k in keys)


def test_residual_call_network_forward_and_depth():
    net = CallNetwork(feature_dim=16, skill_dim=4, hidden=8, depth=4, residual=True)
    assert net.residual is True and len(net.blocks) == 4
    keys = set(net.state_dict())
    assert any(k.startswith("input_proj") for k in keys)
    assert not any(k.startswith("net.") for k in keys)
    out = net(torch.randn(3, 16), torch.tensor([0, 5, 10], dtype=torch.long))
    assert out.shape == (3, 2)
