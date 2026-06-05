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


def masked_kl_divergence(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    legal_mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Sample-weight-weighted KL(target ‖ pred) with illegal-action masking.

    The policy-improvement loss for the search+learning loop (ADR-0031): train the
    play head toward an MCTS visit distribution ``target_probs`` rather than a single
    argmax label. KL (not bare cross-entropy) so the reported value bottoms out at 0
    when the net already reproduces the target — the cross-entropy term carries the
    gradient, the target-entropy term makes the scale interpretable.

    Shapes:
      logits: (B, K) float, requires_grad
      target_probs: (B, K) float — non-negative, each row sums to 1 over legal actions
      legal_mask: (B, K) bool — True for legal actions
      sample_weight: (B,) float
    """
    masked = logits.masked_fill(~legal_mask, _NEG_INF)
    log_pred = log_softmax(masked, dim=-1)
    log_target = torch.where(
        target_probs > 0, target_probs.log(), torch.zeros_like(target_probs)
    )
    per = target_probs * (log_target - log_pred)
    # Illegal positions contribute nothing (target there is 0 by contract; guard the
    # -1e9 * 0 product against propagating into the sum).
    per = per.masked_fill(~legal_mask, 0.0)
    kl = per.sum(dim=-1)
    return (kl * sample_weight).mean()
