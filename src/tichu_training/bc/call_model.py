"""Tichu / Grand Tichu binary call networks.

Separate from the BC trunk because the input phase is different (8-card
hand for Grand Tichu, 14-card hand after schupfen for regular Tichu). The
featurizer's `own_hand` section captures the hand-size distinction, so
both networks consume the same `FEATURIZER_OUTPUT_DIM` input plus the
skill embedding.
"""

import torch
from torch import nn

from tichu_training.bc.model import SkillEmbedding


class CallNetwork(nn.Module):
    """Small MLP classifier producing two logits (call / no-call)."""

    def __init__(
        self,
        feature_dim: int,
        *,
        skill_buckets: int = 10,
        skill_dim: int = 64,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.skill = SkillEmbedding(num_buckets=skill_buckets, dim=skill_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + skill_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, features: torch.Tensor, skill_decile: torch.Tensor) -> torch.Tensor:
        skill_emb = self.skill(skill_decile)
        joint = torch.cat([features, skill_emb], dim=-1)
        return self.net(joint)


class GrandTichuCallNetwork(CallNetwork):
    """Grand Tichu network — trained on the 8-card-hand decision."""


class TichuCallNetwork(CallNetwork):
    """Regular Tichu network — trained on the post-schupfen 14-card-hand decision."""
