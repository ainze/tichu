"""PPO update for PPO Refine (ADR-0029).

Turns collected self-play trajectories into a policy/critic update: GAE
advantages over the sparse terminal `round_outcome` reward, a clipped surrogate
policy loss, a critic (value) loss, an entropy bonus, and a KL-anchor to the
frozen BC policy that keeps the sharpened policy human-plausible.
"""

from typing import NamedTuple

import torch


class PPOBatch(NamedTuple):
    """A flat batch of learner transitions for one PPO update.

    All tensors share leading dim N (transitions). `actions` are play-Intent
    indices; `old_logp` is the legal-masked log-prob recorded at rollout time;
    `advantages` / `returns` come from `compute_gae`.
    """

    features: torch.Tensor    # (N, F)
    skill: torch.Tensor       # (N,) long
    legal_masks: torch.Tensor  # (N, action_dim) bool
    actions: torch.Tensor     # (N,) long
    old_logp: torch.Tensor    # (N,)
    advantages: torch.Tensor  # (N,)
    returns: torch.Tensor     # (N,)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over one episode.

    `rewards` and `values` are 1-D, length-T (time-ordered). `last_value` is the
    bootstrap value of the state after the final transition — 0.0 for a
    terminated Round (one-Round episodes, ADR-0029). Returns `(advantages,
    returns)`, both length-T, with `returns == advantages + values`.
    """
    horizon = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype)
    next_value = torch.as_tensor(last_value, dtype=rewards.dtype)
    for t in range(horizon - 1, -1, -1):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def clipped_policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float,
) -> torch.Tensor:
    """PPO clipped-surrogate policy loss (to be minimized)."""
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def kl_anchor_loss(
    new_logits: torch.Tensor,
    bc_logits: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    """Mean `KL(pi_theta || pi_BC)` over the legal-masked play distribution.

    `new_logits` (the learner) and `bc_logits` (the frozen BC policy) are
    `(B, action_dim)`; `legal_masks` is the boolean legal-Intent mask. Illegal
    Intents carry zero probability on both sides and contribute nothing — the
    anchor keeps the sharpened policy close to BC over the choices that matter.
    """
    new_raw = torch.log_softmax(new_logits.masked_fill(~legal_masks, float("-inf")), dim=-1)
    bc_raw = torch.log_softmax(bc_logits.masked_fill(~legal_masks, float("-inf")), dim=-1)
    probs = new_raw.exp()  # 0 at illegal Intents
    # Replace the -inf logp at illegal slots with a finite 0 BEFORE any multiply:
    # those slots carry zero probability so contribute nothing, and this keeps
    # the unselected autograd branch from producing NaN (0 * -inf) gradients.
    new_logp = new_raw.masked_fill(~legal_masks, 0.0)
    bc_logp = bc_raw.masked_fill(~legal_masks, 0.0)
    return (probs * (new_logp - bc_logp)).sum(dim=-1).mean()


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Critic loss: mean squared error of value estimates against GAE returns."""
    return ((values - returns) ** 2).mean()


def masked_entropy(logits: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    """Mean entropy of the legal-masked play distribution (the exploration term;
    higher entropy => more willingness to try under-explored Intents like bombs)."""
    raw = torch.log_softmax(logits.masked_fill(~legal_masks, float("-inf")), dim=-1)
    probs = raw.exp()  # 0 at illegal Intents
    logp = raw.masked_fill(~legal_masks, 0.0)  # finite; illegal slots add 0 (see KL note)
    return -(probs * logp).sum(dim=-1).mean()


def _masked_logp_at(logits, legal_masks, actions):
    logp = torch.log_softmax(logits.masked_fill(~legal_masks, float("-inf")), dim=-1)
    return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def ppo_update(
    model,
    critic,
    bc_model,
    optimizer,
    batch: PPOBatch,
    *,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    kl_coef: float,
    epochs: int = 1,
    normalize_advantages: bool = True,
) -> dict[str, float]:
    """Run `epochs` PPO update passes over `batch`, combining the clipped policy
    loss, the critic value loss, an entropy bonus, and the KL-anchor to frozen
    BC (ADR-0029). Updates `model` + `critic` in place via `optimizer`; the
    frozen `bc_model` is read-only (the anchor reference). Returns last-epoch
    loss components for logging.
    """
    with torch.no_grad():
        bc_logits = bc_model(batch.features, batch.skill)["play"]

    advantages = batch.advantages
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    stats: dict[str, float] = {}
    for _ in range(epochs):
        play_logits = model(batch.features, batch.skill)["play"]
        new_logp = _masked_logp_at(play_logits, batch.legal_masks, batch.actions)
        values = critic(batch.features)

        policy = clipped_policy_loss(new_logp, batch.old_logp, advantages, clip_eps=clip_eps)
        value = value_loss(values, batch.returns)
        entropy = masked_entropy(play_logits, batch.legal_masks)
        kl = kl_anchor_loss(play_logits, bc_logits, batch.legal_masks)

        loss = policy + vf_coef * value - ent_coef * entropy + kl_coef * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats = {
            "loss": float(loss.detach()),
            "policy_loss": float(policy.detach()),
            "value_loss": float(value.detach()),
            "entropy": float(entropy.detach()),
            "kl_to_bc": float(kl.detach()),
        }
    return stats
