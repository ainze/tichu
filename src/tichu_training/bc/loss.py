"""Masked cross-entropy loss for BC heads.

Illegal-action logits are shifted to `-inf` *before* softmax, so the
probability assigned to them is exactly 0 and the upstream gradient
through those entries is exactly 0.
"""

import torch
from torch.nn.functional import log_softmax


_NEG_INF = -1e9


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Sample-weight-weighted cross-entropy with illegal-action masking.

    Shapes:
      logits: (B, K) float, requires_grad
      target: (B,)   long, in [0, K)
      legal_mask: (B, K) bool — True for legal actions
      sample_weight: (B,) float
    """
    masked = logits.masked_fill(~legal_mask, _NEG_INF)
    log_probs = log_softmax(masked, dim=-1)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return (nll * sample_weight).mean()
