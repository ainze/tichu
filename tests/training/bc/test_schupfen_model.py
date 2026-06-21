"""SchupfenNetwork forward contract.

Q1 of the schupfen design pass: three independent 56-way heads on a shared
MLP trunk, conditioned on the Skill Embedding — same shape as a Call
Network but with a 3-tuple output instead of a single 2-logit output.
"""

import torch

from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.card_slots import CARD_SLOTS


def test_schupfen_network_forward_returns_three_56_logit_tensors():
    net = SchupfenNetwork(feature_dim=32, skill_buckets=10, skill_dim=4, hidden=16)
    features = torch.zeros(7, 32)
    skill = torch.zeros(7, dtype=torch.long)
    logits = net(features, skill)
    assert isinstance(logits, tuple)
    assert len(logits) == 3
    for direction_logits in logits:
        assert direction_logits.shape == (7, CARD_SLOTS)


def test_default_is_non_residual_and_keys_unchanged():
    """`residual=False` (default) must keep the exact original layout so every
    existing checkpoint still loads."""
    net = SchupfenNetwork(feature_dim=32, skill_dim=4, hidden=16)
    assert net.residual is False
    keys = set(net.state_dict())
    assert any(k.startswith("trunk.") for k in keys)
    assert not any(k.startswith("input_proj") or k.startswith("blocks") for k in keys)


def test_residual_trunk_forward_and_depth():
    net = SchupfenNetwork(feature_dim=32, skill_dim=4, hidden=16, depth=4, residual=True)
    assert net.residual is True
    assert len(net.blocks) == 4
    keys = set(net.state_dict())
    assert any(k.startswith("input_proj") for k in keys)
    assert not any(k.startswith("trunk.") for k in keys)
    logits = net(torch.randn(5, 32), torch.zeros(5, dtype=torch.long))
    assert len(logits) == 3 and all(l.shape == (5, CARD_SLOTS) for l in logits)
