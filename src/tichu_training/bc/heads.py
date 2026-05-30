"""Multi-head BC model: shared trunk + per-decision-type linear heads.

Heads are kept in deterministic order keyed by `decision_type`:
  play → wish → dragon_assignment

Schupfen is **not** a BC head — it is served by a standalone Schupfen
Network per [ADR-0012](../../../docs/adr/0012-schupfen-is-a-standalone-network.md).
The schupfen parquet shard feeds `train_schupfen`, not `train_bc`.
"""

from collections import OrderedDict

import torch
from torch import nn

# Re-exported for backward compatibility — `HEAD_LOGIT_DIMS` lives in
# `bc/decision_types.py` so torch-free readers (bc/dataset, bc/materialised,
# bsw/to_parquet, parse_bsw) can import it without pulling torch.
from tichu_training.bc.decision_types import HEAD_LOGIT_DIMS
from tichu_training.bc.model import SkillEmbedding, TichuTrunk

__all__ = ["HEAD_LOGIT_DIMS", "BCModel"]


class _Head(nn.Module):
    def __init__(self, trunk_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCModel(nn.Module):
    """Trunk + skill embedding + four task heads.

    Forward returns a dict keyed by decision_type → logits. Loss should be
    computed per-decision-type with `masked_cross_entropy` from `bc.loss`.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        skill_buckets: int = 10,
        skill_dim: int = 64,
        trunk_hidden: int = 1024,
        trunk_depth: int = 4,
        trunk_out_dim: int = 512,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.skill = SkillEmbedding(num_buckets=skill_buckets, dim=skill_dim)
        self.trunk = TichuTrunk(
            in_dim=feature_dim + skill_dim,
            hidden=trunk_hidden,
            depth=trunk_depth,
            out_dim=trunk_out_dim,
        )
        self.heads = nn.ModuleDict(OrderedDict([
            (name, _Head(trunk_out_dim, dim, hidden=head_hidden))
            for name, dim in HEAD_LOGIT_DIMS.items()
        ]))

    def forward(
        self, features: torch.Tensor, skill_decile: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        skill_emb = self.skill(skill_decile)
        joint = torch.cat([features, skill_emb], dim=-1)
        h = self.trunk(joint)
        return {name: head(h) for name, head in self.heads.items()}
