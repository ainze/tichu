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
