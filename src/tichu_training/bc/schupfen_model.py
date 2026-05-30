"""Schupfen Network — standalone (ADR-0012, Q1 of the design pass).

Three independent 56-way heads on a shared MLP trunk: one head per
direction (`to_next`, `to_partner`, `to_previous`). Each head produces
a distribution over the 56-card identity space. Hand-masking and
distinct-card decoding happen outside `forward` — this module is the
raw tensor contract.

Same input shape as a Call Network: `FEATURIZER_OUTPUT_DIM` features +
Skill Embedding.
"""

import torch
from torch import nn

from tichu_training.bc.model import SkillEmbedding
from tichu_training.card_slots import CARD_SLOTS


class SchupfenNetwork(nn.Module):
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
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + skill_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head_to_next = nn.Linear(hidden, CARD_SLOTS)
        self.head_to_partner = nn.Linear(hidden, CARD_SLOTS)
        self.head_to_previous = nn.Linear(hidden, CARD_SLOTS)

    def forward(
        self, features: torch.Tensor, skill_decile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skill_emb = self.skill(skill_decile)
        joint = torch.cat([features, skill_emb], dim=-1)
        body = self.trunk(joint)
        return (
            self.head_to_next(body),
            self.head_to_partner(body),
            self.head_to_previous(body),
        )
