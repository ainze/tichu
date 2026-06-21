"""Tichu / Grand Tichu binary call networks.

Separate from the BC trunk because the input phase is different (8-card
hand for Grand Tichu, 14-card hand after schupfen for regular Tichu). The
featurizer's `own_hand` section captures the hand-size distinction, so
both networks consume the same `FEATURIZER_OUTPUT_DIM` input plus the
skill embedding.
"""

import torch
from torch import nn

from tichu_training.bc.model import SkillEmbedding, _ResidualBlock


class CallNetwork(nn.Module):
    """Small MLP classifier producing two logits (call / no-call).

    Default (`residual=False`) is the original 2-hidden-layer non-residual
    MLP — state-dict-compatible with existing checkpoints. `residual=True`
    swaps in a pre-norm residual trunk (`input_proj` → `depth` residual
    blocks → 2-logit head). The capacity probe (DEBUG-cap7e2) showed this
    helps the Tichu call (+0.007 AUC) but does NOTHING for Grand Tichu
    (information-limited), so enable it for tichu only. `depth` is ignored
    when `residual=False`.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        skill_buckets: int = 10,
        skill_dim: int = 64,
        hidden: int = 256,
        depth: int = 4,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.skill = SkillEmbedding(num_buckets=skill_buckets, dim=skill_dim)
        self.residual = bool(residual)
        if self.residual:
            self.input_proj = nn.Linear(feature_dim + skill_dim, hidden)
            self.blocks = nn.ModuleList([_ResidualBlock(hidden) for _ in range(depth)])
            self.out = nn.Linear(hidden, 2)
        else:
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
        if self.residual:
            h = self.input_proj(joint)
            for block in self.blocks:
                h = block(h)
            return self.out(h)
        return self.net(joint)


class GrandTichuCallNetwork(CallNetwork):
    """Grand Tichu network — trained on the 8-card-hand decision."""


class TichuCallNetwork(CallNetwork):
    """Regular Tichu network — trained on the post-schupfen 14-card-hand decision."""
