"""Full-stack co-training PPO primitives (ADR-0034).

Sharpens play + the Schupfen Network + the Tichu/Grand Call Networks together in
one self-play run. This module holds the pieces that are *new* relative to the
play-only PPO Refine harness (ADR-0029): the Schupfen action's
without-replacement sampling / log-prob, the per-net policy-gradient losses, and
the batched co-training policy bundle. Play reuses `ppo/policy.py` + `ppo/update.py`
unchanged; the collector lives in `ppo/rollout.py`.
"""

from dataclasses import dataclass

import numpy as np
import torch

from tichu_engine.legality import SchupfenPass
from tichu_training.card_slots import CARD_SLOTS, card_slot, slot_to_card
from tichu_training.featurizer import featurize
from tichu_training.perfect_info import featurize_perfect_info
from tichu_training.ppo.policy import BatchedPolicy
from tichu_training.ppo.rollout import CallChoice, SchupfenChoice
from tichu_training.ppo.update import (
    _masked_logp_at,
    clipped_policy_loss,
    compute_gae,
    kl_anchor_loss,
    masked_entropy,
    value_loss,
)

_NUM_DIRECTIONS = 3  # to_next, to_partner, to_previous


def sample_schupfen(
    head_logits: torch.Tensor,
    hand_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a Schupfen action: 3 DISTINCT in-hand cards, one per direction head.

    `head_logits` is `(B, 3, CARD_SLOTS)` (next / partner / previous); `hand_mask`
    is `(B, CARD_SLOTS)` bool (cards in hand). Directions are sampled in order with
    the chosen card removed from the available set each step (without-replacement,
    so the three cards are distinct — the constraint `ml_agent._act_schupfen`
    enforces greedily). Returns `(indices (B,3) long, logprob (B,))`, where the
    log-prob is the sum of the three sequential masked-softmax log-probs — the exact
    factorization the PPO importance ratio needs.
    """
    rows = head_logits.shape[0]
    avail = hand_mask.clone()  # True == still selectable
    indices = torch.empty(rows, _NUM_DIRECTIONS, dtype=torch.long, device=head_logits.device)
    total_logp = torch.zeros(rows, dtype=head_logits.dtype, device=head_logits.device)
    for d in range(_NUM_DIRECTIONS):
        logits_d = head_logits[:, d, :].masked_fill(~avail, float("-inf"))
        logp = torch.log_softmax(logits_d, dim=-1)
        idx = torch.multinomial(logp.exp(), num_samples=1, generator=generator).squeeze(-1)
        total_logp = total_logp + logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        indices[:, d] = idx
        avail = avail.scatter(1, idx.unsqueeze(-1), False)
    return indices, total_logp


@dataclass
class NetBatch:
    """One decision-type's transitions for its policy-gradient sub-update (ADR-0034).

    `features` (observable) are re-forwarded through that net; `actions` is `(n,)`
    int for play / calls or `(n, 3)` card slots for schupfen; `masks` is the
    legal-Intent mask (play) / hand mask (schupfen) / `None` (binary calls)."""

    features: torch.Tensor
    skill: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor | None
    old_logp: torch.Tensor
    advantages: torch.Tensor


@dataclass
class CoTrainBatch:
    """A co-training update's data: per-net policy sub-batches keyed by decision
    type, plus the SHARED critic's pooled inputs (every step's perfect-info
    features) and GAE returns (ADR-0034)."""

    nets: dict[str, NetBatch]
    critic_features: torch.Tensor
    returns: torch.Tensor


def build_cotrain_batch(trajectories, *, skill_decile: int, gamma: float, lam: float) -> CoTrainBatch:
    """Flatten heterogeneous learner trajectories into a `CoTrainBatch`.

    GAE runs over each whole per-seat trajectory in time order (grand -> schupfen
    -> tichu -> plays) against its terminal `round_outcome`, so every decision —
    one-shot calls/schupfen included — gets its advantage from the shared critic's
    value chain (ADR-0034; under gamma=1 this reduces to `R - V(state)`). Steps are
    then grouped by `decision_type` for the per-net policy losses, while the critic
    pools all steps."""
    crit_feats: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    groups: dict[str, dict[str, list]] = {}

    for traj in trajectories:
        steps = traj.steps
        if not steps:
            continue
        values = torch.tensor([s.value for s in steps], dtype=torch.float32)
        rewards = torch.zeros(len(steps), dtype=torch.float32)
        rewards[-1] = float(traj.reward)
        adv, ret = compute_gae(rewards, values, gamma=gamma, lam=lam, last_value=0.0)
        for i, s in enumerate(steps):
            cf = s.critic_features if s.critic_features is not None else s.features
            crit_feats.append(torch.as_tensor(cf, dtype=torch.float32))
            returns.append(ret[i])
            g = groups.setdefault(
                s.decision_type,
                {"features": [], "actions": [], "masks": [], "old_logp": [], "adv": []},
            )
            g["features"].append(torch.as_tensor(s.features, dtype=torch.float32))
            g["actions"].append(torch.as_tensor(s.intent_index, dtype=torch.long))
            g["masks"].append(
                None if s.legal_mask is None
                else torch.as_tensor(s.legal_mask, dtype=torch.bool)
            )
            g["old_logp"].append(float(s.logprob))
            g["adv"].append(adv[i])

    nets: dict[str, NetBatch] = {}
    for dt, g in groups.items():
        features = torch.stack(g["features"])
        masks = None if g["masks"][0] is None else torch.stack(g["masks"])
        nets[dt] = NetBatch(
            features=features,
            skill=torch.full((features.shape[0],), int(skill_decile), dtype=torch.long),
            actions=torch.stack(g["actions"]),
            masks=masks,
            old_logp=torch.tensor(g["old_logp"], dtype=torch.float32),
            advantages=torch.stack(g["adv"]),
        )
    return CoTrainBatch(
        nets=nets,
        critic_features=torch.stack(crit_feats),
        returns=torch.stack(returns),
    )


def schupfen_logp(
    head_logits: torch.Tensor,
    hand_mask: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Log-prob of a fixed Schupfen action under `head_logits`, replaying the
    without-replacement masking in the recorded direction order (next, partner,
    previous). `indices` is `(B, 3)`. Returns `(B,)` — differentiable, for the PPO
    re-forward. Mirrors `sample_schupfen`'s factorization exactly."""
    rows = head_logits.shape[0]
    avail = hand_mask.clone()
    total_logp = torch.zeros(rows, dtype=head_logits.dtype, device=head_logits.device)
    for d in range(_NUM_DIRECTIONS):
        logits_d = head_logits[:, d, :].masked_fill(~avail, float("-inf"))
        logp = torch.log_softmax(logits_d, dim=-1)
        idx = indices[:, d]
        total_logp = total_logp + logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        avail = avail.scatter(1, idx.unsqueeze(-1), False)
    return total_logp


def _net_logits(model, features: torch.Tensor, skill: torch.Tensor, decision_type: str) -> torch.Tensor:
    """Forward one net to logits, normalizing shapes across the three net families:
    play -> (B, PLAY_DIM); schupfen -> (B, 3, CARD_SLOTS); call -> (B, 2)."""
    if decision_type == "play":
        return model(features, skill)["play"]
    if decision_type == "schupfen":
        return torch.stack(model(features, skill), dim=1)
    return model(features, skill)  # tichu / grand Call Network


def _policy_kl_entropy(decision_type, logits, bc_logits, nb: "NetBatch"):
    """(new_logp, kl_to_bc, entropy) for one net's batch. Schupfen sums the three
    direction heads (masked to the hand); play/calls are a single masked categorical
    (calls use a trivial all-legal binary mask)."""
    if decision_type == "schupfen":
        hand = nb.masks
        new_logp = schupfen_logp(logits, hand, nb.actions)
        kl = sum(kl_anchor_loss(logits[:, d, :], bc_logits[:, d, :], hand) for d in range(_NUM_DIRECTIONS))
        ent = sum(masked_entropy(logits[:, d, :], hand) for d in range(_NUM_DIRECTIONS))
        return new_logp, kl, ent
    mask = nb.masks if nb.masks is not None else torch.ones_like(logits, dtype=torch.bool)
    new_logp = _masked_logp_at(logits, mask, nb.actions)
    kl = kl_anchor_loss(logits, bc_logits, mask)
    ent = masked_entropy(logits, mask)
    return new_logp, kl, ent


def cotrain_update(
    models: dict,
    bc_models: dict,
    critic,
    optimizer,
    batch: CoTrainBatch,
    *,
    clip_eps: float,
    vf_coef: float,
    ent_coefs: dict,
    kl_coefs: dict,
    epochs: int = 1,
    normalize_advantages: bool = True,
    device: str = "cpu",
) -> dict:
    """One co-training PPO update (ADR-0034). Combines the SHARED critic's value
    loss with a per-net clipped surrogate + KL-anchor-to-its-frozen-BC + entropy,
    each net using its own `kl_coefs[dt]` / `ent_coefs[dt]`. Updates every present
    net + the critic in place via the single `optimizer`. Advantages are normalized
    per net. Returns per-net + critic loss components for logging.

    With `device != "cpu"` the heavy batched forward+backward runs on that device:
    the nets, the frozen BC anchors, the critic, the optimizer moment state, and the
    batch are moved there for the update and the nets/critic/optimizer are restored
    to CPU afterwards (in a `finally`) — so the engine-bound CPU rollout, and the
    CPU-only Resume Bundle, are unaffected. Profiling showed only this step is worth
    offloading; the rollout is CPU-engine-bound (ADR-0034)."""
    dev = torch.device(device)
    if dev.type != "cpu":
        for m in models.values():
            m.to(dev)
        for m in bc_models.values():
            m.to(dev)
        critic.to(dev)
        _move_optimizer_state(optimizer, dev)
        batch = _batch_to(batch, dev)
    try:
        return _cotrain_update_body(
            models, bc_models, critic, optimizer, batch,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coefs=ent_coefs,
            kl_coefs=kl_coefs, epochs=epochs, normalize_advantages=normalize_advantages,
        )
    finally:
        if dev.type != "cpu":
            cpu = torch.device("cpu")
            for m in models.values():
                m.to(cpu)
            for m in bc_models.values():
                m.to(cpu)
            critic.to(cpu)
            _move_optimizer_state(optimizer, cpu)


def _move_optimizer_state(optimizer, dev) -> None:
    """Move an optimizer's per-parameter state tensors (Adam's exp_avg/exp_avg_sq …)
    to `dev`, so moving the params between devices doesn't desync the moments."""
    for state in optimizer.state.values():
        for key, val in state.items():
            if torch.is_tensor(val):
                state[key] = val.to(dev)


def _batch_to(batch: CoTrainBatch, dev) -> CoTrainBatch:
    return CoTrainBatch(
        nets={
            dt: NetBatch(
                features=nb.features.to(dev), skill=nb.skill.to(dev),
                actions=nb.actions.to(dev),
                masks=None if nb.masks is None else nb.masks.to(dev),
                old_logp=nb.old_logp.to(dev), advantages=nb.advantages.to(dev),
            )
            for dt, nb in batch.nets.items()
        },
        critic_features=batch.critic_features.to(dev),
        returns=batch.returns.to(dev),
    )


def _cotrain_update_body(
    models, bc_models, critic, optimizer, batch, *,
    clip_eps, vf_coef, ent_coefs, kl_coefs, epochs, normalize_advantages,
) -> dict:
    with torch.no_grad():
        bc_logits = {
            dt: _net_logits(bc_models[dt], nb.features, nb.skill, dt)
            for dt, nb in batch.nets.items()
        }
    advs: dict[str, torch.Tensor] = {}
    for dt, nb in batch.nets.items():
        a = nb.advantages
        if normalize_advantages and a.numel() > 1:
            a = (a - a.mean()) / (a.std() + 1e-8)
        advs[dt] = a

    stats: dict[str, float] = {}
    for _ in range(epochs):
        values = critic(batch.critic_features)
        vloss = value_loss(values, batch.returns)
        total = vf_coef * vloss
        for dt, nb in batch.nets.items():
            logits = _net_logits(models[dt], nb.features, nb.skill, dt)
            new_logp, kl, ent = _policy_kl_entropy(dt, logits, bc_logits[dt], nb)
            ploss = clipped_policy_loss(new_logp, nb.old_logp, advs[dt], clip_eps=clip_eps)
            total = total + ploss - ent_coefs.get(dt, 0.0) * ent + kl_coefs.get(dt, 0.0) * kl
            stats[f"{dt}_policy_loss"] = float(ploss.detach())
            stats[f"{dt}_kl"] = float(kl.detach())
            stats[f"{dt}_entropy"] = float(ent.detach())
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        stats["value_loss"] = float(vloss.detach())
        stats["loss"] = float(total.detach())
    return stats


class BatchedCoTrainPolicy:
    """The full-stack learner bundle the collector drives (ADR-0034).

    Wraps the four policy nets (play / schupfen / tichu / grand) and the SHARED
    Perfect-Info Critic, exposing the `RolloutPolicy` interface: `act_play_batch`
    (delegated to the play `BatchedPolicy`), `act_schupfen_batch`, and
    `act_call_batch`. Each method samples its net's distribution, records the PPO
    bookkeeping, and values the state with the critic (perfect-info features when
    enabled, observable otherwise). The critic is train-time only — never shipped."""

    def __init__(
        self, play_model, schupfen_model, tichu_model, grand_model, critic,
        *, skill_decile: int = 9, perfect_info: bool = True, generator=None,
    ) -> None:
        self._play = BatchedPolicy(
            play_model, critic, skill_decile=skill_decile,
            perfect_info=perfect_info, generator=generator,
        )
        self.play_model = play_model
        self.schupfen_model = schupfen_model
        self.call_models = {"tichu": tichu_model, "grand": grand_model}
        self.critic = critic
        self._skill = int(skill_decile)
        self._perfect_info = bool(perfect_info)
        self._gen = generator

    def act_play_batch(self, decisions):
        return self._play.act_play_batch(decisions)

    def _stack_features(self, decisions):
        feats = [featurize(ps) for _seat, ps, _gs in decisions]
        features = torch.from_numpy(np.stack(feats))
        if self._perfect_info:
            pi = [featurize_perfect_info(gs, seat) for seat, _ps, gs in decisions]
            crit_in = torch.from_numpy(np.stack(pi))
        else:
            crit_in = features
        skill = torch.full((len(decisions),), self._skill, dtype=torch.long)
        return features, crit_in, skill

    def act_schupfen_batch(self, decisions):
        features, crit_in, skill = self._stack_features(decisions)
        hand_masks = torch.zeros(len(decisions), CARD_SLOTS, dtype=torch.bool)
        for row, (_seat, ps, _gs) in enumerate(decisions):
            for card in ps.hand:
                hand_masks[row, card_slot(card)] = True
        with torch.no_grad():
            head_logits = torch.stack(self.schupfen_model(features, skill), dim=1)
            indices, logp = sample_schupfen(head_logits, hand_masks, generator=self._gen)
            values = self.critic(crit_in)
        choices = []
        for row in range(len(decisions)):
            slots = indices[row].tolist()
            action = SchupfenPass(
                to_next=slot_to_card(slots[0]),
                to_partner=slot_to_card(slots[1]),
                to_previous=slot_to_card(slots[2]),
            )
            choices.append(SchupfenChoice(
                concrete_action=action,
                card_indices=indices[row],
                logprob=float(logp[row]),
                value=float(values[row]),
                features=features[row],
                legal_masks=hand_masks[row],
                critic_features=(crit_in[row] if self._perfect_info else None),
            ))
        return choices

    def act_call_batch(self, kind, decisions):
        model = self.call_models[kind]
        features, crit_in, skill = self._stack_features(decisions)
        with torch.no_grad():
            logp_all = torch.log_softmax(model(features, skill), dim=-1)
            actions = torch.multinomial(
                logp_all.exp(), num_samples=1, generator=self._gen
            ).squeeze(-1)
            chosen_logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            values = self.critic(crit_in)
        return [
            CallChoice(
                called=bool(actions[row].item()),
                logprob=float(chosen_logp[row]),
                value=float(values[row]),
                features=features[row],
                critic_features=(crit_in[row] if self._perfect_info else None),
            )
            for row in range(len(decisions))
        ]


def train_cotrain(
    models: dict,
    bc_models: dict,
    critic,
    sample_positions,
    *,
    optimizer,
    kl_controllers: dict,
    ent_coefs: dict,
    iterations: int,
    gamma: float,
    lam: float,
    clip_eps: float,
    vf_coef: float,
    ppo_epochs: int,
    skill_decile: int = 9,
    learner_team: int = 0,
    perfect_info: bool = True,
    opponent_policy_provider=None,
    generator=None,
    on_iteration=None,
    start_iter: int = 0,
    update_device: str = "cpu",
) -> list[dict]:
    """Run `iterations` of full-stack co-training self-play (ADR-0034).

    Each iteration: build a `BatchedCoTrainPolicy` over the live weights, sample
    Starting Positions, roll out self-play (opponents from
    `opponent_policy_provider`, else pure self-play), `build_cotrain_batch`, and one
    `cotrain_update` using each net's current adaptive `kl_coef`. The measured
    per-net KL-to-BC then steers that net's controller for the next iteration
    (schupfen anchored tightest is just a smaller `target`/`coef`, ADR-0034).
    `start_iter` continues the deal-pool seed sequence + budget accounting on resume."""
    from tichu_training.ppo.rollout import collect_rollout

    history: list[dict] = []
    for it in range(start_iter, start_iter + iterations):
        policy = BatchedCoTrainPolicy(
            models["play"], models["schupfen"], models["tichu"], models["grand"],
            critic, skill_decile=skill_decile, perfect_info=perfect_info, generator=generator,
        )
        opponent = opponent_policy_provider(it) if opponent_policy_provider is not None else policy
        positions = sample_positions(it)
        trajs = collect_rollout(
            positions, policy, opponent_policy=opponent, learner_team=learner_team
        )
        batch = build_cotrain_batch(trajs, skill_decile=skill_decile, gamma=gamma, lam=lam)
        kl_coefs = {dt: kl_controllers[dt].coef for dt in models}
        stats = cotrain_update(
            models, bc_models, critic, optimizer, batch,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coefs=ent_coefs,
            kl_coefs=kl_coefs, epochs=ppo_epochs, device=update_device,
        )
        for dt in models:
            stats[f"{dt}_kl_coef"] = kl_controllers[dt].coef
            kl_controllers[dt].update(stats[f"{dt}_kl"])
        stats["iter"] = it
        history.append(stats)
        if on_iteration is not None:
            on_iteration(it, stats)
    return history
