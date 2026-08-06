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

__all__ = ["HEAD_LOGIT_DIMS", "BCModel", "forward_play_model",
           "play_mask_for_trunk"]


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


def play_mask_for_trunk(head: str, legal_mask: torch.Tensor) -> torch.Tensor:
    """The 1809-wide play-legality mask for the trunk input; zeros for any other
    head (v7, ADR-0044).

    The trunk is shared and fixed-width, but head masks are 1809 / 14 / 2 wide.
    Zero-padding a 14-wide wish mask into play's action-index space would be
    actively misleading — those indices name entirely different actions, so the
    trunk would read a wish's legal ranks as a claim about Singles and Pairs.
    All-zeros says "no play-legality information here", which is true, and
    `phase` already tells the trunk which decision type it is looking at.
    Non-play rows are 3.2% of the corpus.
    """
    if head == "play":
        return legal_mask
    return legal_mask.new_zeros((legal_mask.shape[0], HEAD_LOGIT_DIMS["play"]))


def forward_play_model(model, features, skill, legal_masks=None):
    """Forward a `BCModel` that MAY consume the legal mask as a trunk input.

    v7 (ADR-0044). One definition for every caller — the PPO rollout, the PPO
    update, the KL-anchor forward, the wish head and vine all forward the play
    BCModel, and a per-call-site fix is precisely how the `team_scores` and
    schupfen `current_player` skews happened.

    `legal_masks=None` means "not a Play Decision" and feeds ZEROS, matching
    `play_mask_for_trunk` in training. A mask-less model ignores the argument
    entirely, so this is a no-op on every pre-v7 net.
    """
    if not getattr(model, "use_legal_mask", False):
        return model(features, skill)
    if legal_masks is None:
        mask = features.new_zeros((features.shape[0], HEAD_LOGIT_DIMS["play"]))
    else:
        mask = legal_masks.to(features.dtype)
    return model(features, skill, mask)


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
        use_legal_mask: bool = False,
    ) -> None:
        super().__init__()
        self.skill = SkillEmbedding(num_buckets=skill_buckets, dim=skill_dim)
        # v7 (ADR-0044 / PR #80): the legal-Intent mask as a trunk INPUT, not
        # only an output mask. A model-arch flag — the mask is already in every
        # batch, so this costs no disk and no featurizer bump. Non-play rows pass
        # zeros: head masks are 1809 / 14 / 2 wide and the trunk needs a fixed
        # width, play is 96.8% of rows, and `phase` already tells the trunk the
        # decision type, so "no play-legality information here" is both true and
        # unambiguous.
        self.use_legal_mask = bool(use_legal_mask)
        mask_dim = HEAD_LOGIT_DIMS["play"] if self.use_legal_mask else 0
        self.trunk = TichuTrunk(
            in_dim=feature_dim + skill_dim + mask_dim,
            hidden=trunk_hidden,
            depth=trunk_depth,
            out_dim=trunk_out_dim,
        )
        self.heads = nn.ModuleDict(OrderedDict([
            (name, _Head(trunk_out_dim, dim, hidden=head_hidden))
            for name, dim in HEAD_LOGIT_DIMS.items()
        ]))

    def forward(
        self, features: torch.Tensor, skill_decile: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        skill_emb = self.skill(skill_decile)
        parts = [features, skill_emb]
        if self.use_legal_mask:
            if legal_mask is None:
                # Loudly, not silently. A zero-filled fallback here would serve a
                # policy that sees no legal actions where training saw the real
                # set — invisible in every metric until strength quietly drops.
                raise ValueError(
                    "BCModel was built with use_legal_mask=True but forward() got "
                    "legal_mask=None. Pass the play-legality mask (or rebuild the "
                    "model with use_legal_mask=False)."
                )
            parts.append(legal_mask.to(features.dtype))
        joint = torch.cat(parts, dim=-1)
        h = self.trunk(joint)
        return {name: head(h) for name, head in self.heads.items()}
