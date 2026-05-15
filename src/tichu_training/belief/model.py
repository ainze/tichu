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
    def __init__(
        self,
        feature_dim: int,
        *,
        num_opponents: int = 3,
        num_cards: int = 56,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.num_opponents = num_opponents
        self.num_cards = num_cards
        out_dim = num_opponents * num_cards
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
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
