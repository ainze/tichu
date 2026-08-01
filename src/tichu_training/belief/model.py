"""Belief model — per-opponent per-card occupancy MLP.

Multi-label binary classification: for each (opponent_i, card_j) the model
outputs a logit for `P(opponent_i holds card_j)`. The loss masks out
positions where the card is in the acting player's own hand or has already
been played; only opponent-vs-elsewhere uncertainty contributes gradient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefModel(nn.Module):
    """Per-opponent per-card occupancy logits.

    Default (`residual=False`) is the original 2-layer MLP — state-dict
    compatible with every existing checkpoint. `residual=True` swaps in the play
    **Policy Network**'s own `TichuTrunk` (`input_proj` → `depth` residual blocks
    → `output_proj`), so belief can be given the same trunk shape the policy uses
    (`hidden=1024, depth=4, trunk_out_dim=512`).

    The distinction matters because the ADR-0041 capacity sweep varied *width*
    only and found `+0.024` flat from 64 to 512 — while this project's
    head-capacity probe found schupfen / tichu-call are **depth**-limited and
    that width alone underperforms. `depth` and `trunk_out_dim` are ignored when
    `residual=False`.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        num_opponents: int = 3,
        num_cards: int = 56,
        hidden: int = 256,
        depth: int = 4,
        trunk_out_dim: int | None = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.num_opponents = num_opponents
        self.num_cards = num_cards
        self.residual = bool(residual)
        out_dim = num_opponents * num_cards
        if self.residual:
            from tichu_training.bc.model import TichuTrunk

            head_in = int(trunk_out_dim or hidden)
            self.trunk = TichuTrunk(
                in_dim=feature_dim, hidden=hidden, depth=depth, out_dim=head_in,
            )
            self.head = nn.Linear(head_in, out_dim)
        else:
            self.fc1 = nn.Linear(feature_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.residual:
            h = self.trunk(features)
        else:
            h = F.gelu(self.fc1(features))
            h = F.gelu(self.fc2(h))
        logits = self.head(h)
        return logits.view(features.shape[0], self.num_opponents, self.num_cards)


def belief_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked binary cross-entropy.

    Positions where `mask == False` are excluded from the average; if all
    positions are masked out, the loss is 0.
    """
    per_pos = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none"
    )
    mask_f = mask.float()
    denom = mask_f.sum()
    if float(denom) == 0.0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return (per_pos * mask_f).sum() / denom


def holder_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Per-card distribution over the three opponents — `softmax` across the
    opponent axis rather than an independent `sigmoid` per position.

    A card has **exactly one** holder, and the **Determinization Sampler**
    consumes these values as draw weights. Independent sigmoids never learn the
    constraint: the fitted v6 model carried 0.982 (sd 0.068) of mass per unseen
    card and missed the *publicly known* `hand_sizes` by up to 2.4 cards.
    Normalising here makes "exactly one holder" hold by construction.
    """
    return torch.softmax(logits, dim=1)


def belief_holder_loss(
    logits: torch.Tensor, labels: torch.Tensor, card_mask: torch.Tensor
) -> torch.Tensor:
    """Per-card 3-way cross-entropy over **Unseen Cards** — the loss counterpart
    of `holder_probabilities`.

    `card_mask` is `(batch, num_cards)`: True where some opponent holds the card.
    Cards the actor can see contribute nothing, so nothing is credited for
    common knowledge.
    """
    mask = card_mask.bool()
    if not bool(mask.any()):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    # (B, opp, card) -> (B*card, opp), keeping only the masked-in cards.
    flat_logits = logits.permute(0, 2, 1)[mask]
    flat_target = labels.permute(0, 2, 1)[mask].argmax(dim=-1)
    return F.cross_entropy(flat_logits, flat_target)


def belief_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> float:
    """Fraction of masked-in positions where `(logits > 0)` matches `labels`."""
    with torch.no_grad():
        pred = (logits > 0).to(labels.dtype)
        matches = (pred == labels) & mask
        denom = mask.sum()
        if int(denom) == 0:
            return 0.0
        return float(matches.sum().float() / denom.float())
