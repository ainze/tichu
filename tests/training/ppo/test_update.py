"""PPO update for PPO Refine (ADR-0029): GAE, clipped surrogate, value loss,
entropy, and the KL-anchor to the frozen BC policy."""

import copy
import math

import torch

from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.heads import HEAD_LOGIT_DIMS, BCModel
from tichu_training.ppo.rollout import Trajectory, TrajectoryStep
from tichu_training.ppo.update import (
    PPOBatch,
    build_batch,
    clipped_policy_loss,
    compute_gae,
    masked_entropy,
    kl_anchor_loss,
    ppo_update,
    value_loss,
)


def _step(intent, logp, value, feature_dim=4, action_dim=6):
    return TrajectoryStep(
        intent_index=intent,
        logprob=logp,
        value=value,
        features=torch.full((feature_dim,), float(value)),
        legal_mask=torch.ones(action_dim, dtype=torch.bool),
    )


def _legal_prob(model, features, skill, mask, action) -> float:
    with torch.no_grad():
        logits = model(features.unsqueeze(0), skill.unsqueeze(0))["play"][0]
        logp = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    return float(logp[action].exp())


def test_compute_gae_matches_monte_carlo_when_lambda_is_one():
    # lambda=1, gamma=1 -> GAE reduces to Monte-Carlo: return is the full
    # terminal reward at every step; advantage is return minus the value.
    values = torch.tensor([1.0, 2.0, 3.0])
    rewards = torch.tensor([0.0, 0.0, 5.0])  # terminal-only reward (one Round)

    adv, ret = compute_gae(rewards, values, gamma=1.0, lam=1.0)

    assert torch.allclose(adv, torch.tensor([4.0, 3.0, 2.0]))
    assert torch.allclose(ret, torch.tensor([5.0, 5.0, 5.0]))
    # returns are advantages plus the value baseline, by construction.
    assert torch.allclose(ret, adv + values)


def test_clipped_policy_loss_at_ratio_one_is_negative_mean_advantage():
    advantages = torch.tensor([2.0, -1.0, 0.5])
    logp = torch.tensor([-0.5, -1.0, -2.0])
    # new == old -> ratio 1 (inside any clip band) -> loss = -mean(advantage).
    loss = clipped_policy_loss(logp, logp, advantages, clip_eps=0.2)
    assert torch.allclose(loss, -advantages.mean())


def test_clipped_policy_loss_caps_a_positive_advantage_update():
    advantages = torch.tensor([3.0])
    old = torch.tensor([-1.0])
    new = torch.tensor([0.0])  # ratio = e^1 ~ 2.718, far above 1 + eps

    loss = clipped_policy_loss(new, old, advantages, clip_eps=0.2)

    # surrogate = min(ratio*adv, clamp(ratio, 0.8, 1.2)*adv) = min(8.15, 3.6).
    # Clipping caps the objective at 3.6, so the loss is -3.6 (not -8.15).
    assert torch.allclose(loss, torch.tensor(-3.6), atol=1e-5)


def test_kl_anchor_is_zero_when_policy_equals_bc():
    logits = torch.randn(4, 6)
    masks = torch.ones(4, 6, dtype=torch.bool)
    kl = kl_anchor_loss(logits, logits.clone(), masks)
    assert torch.allclose(kl, torch.zeros(()), atol=1e-6)


def test_kl_anchor_matches_hand_computed_value_over_legal_actions():
    # policy = uniform [0.5, 0.5]; BC = [0.25, 0.75] (logits [0, ln 3]).
    new_logits = torch.tensor([[0.0, 0.0]])
    bc_logits = torch.tensor([[0.0, math.log(3.0)]])
    masks = torch.ones(1, 2, dtype=torch.bool)

    kl = kl_anchor_loss(new_logits, bc_logits, masks)

    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert torch.allclose(kl, torch.tensor(expected), atol=1e-6)


def test_value_loss_is_mean_squared_error_to_returns():
    values = torch.tensor([1.0, 2.0, 3.0])
    returns = torch.tensor([1.5, 2.0, 5.0])
    loss = value_loss(values, returns)
    assert torch.allclose(loss, torch.tensor((0.25 + 0.0 + 4.0) / 3.0))


def test_masked_entropy_of_uniform_over_k_legal_actions_is_log_k():
    logits = torch.zeros(2, 6)  # equal logits -> uniform over the legal subset
    masks = torch.zeros(2, 6, dtype=torch.bool)
    masks[:, :4] = True  # four legal actions per row
    ent = masked_entropy(logits, masks)
    assert torch.allclose(ent, torch.tensor(math.log(4.0)), atol=1e-6)


def test_build_batch_flattens_one_trajectory_with_gae():
    traj = Trajectory(
        seat=0, team=0, reward=5.0,
        steps=[_step(1, -0.5, 1.0), _step(2, -0.6, 2.0), _step(3, -0.7, 3.0)],
    )
    batch = build_batch([traj], skill_decile=9, gamma=1.0, lam=1.0)

    assert isinstance(batch, PPOBatch)
    assert batch.features.shape == (3, 4)
    assert batch.legal_masks.shape == (3, 6)
    assert torch.equal(batch.actions, torch.tensor([1, 2, 3]))
    assert torch.allclose(batch.old_logp, torch.tensor([-0.5, -0.6, -0.7]))
    assert batch.skill.shape == (3,) and torch.all(batch.skill == 9)
    # GAE over (values=[1,2,3], terminal reward 5): adv=[4,3,2], returns=[5,5,5].
    assert torch.allclose(batch.advantages, torch.tensor([4.0, 3.0, 2.0]))
    assert torch.allclose(batch.returns, torch.tensor([5.0, 5.0, 5.0]))


def test_build_batch_computes_gae_per_trajectory_independently():
    a = Trajectory(seat=0, team=0, reward=10.0, steps=[_step(0, -0.1, 1.0), _step(1, -0.1, 1.0)])
    b = Trajectory(seat=2, team=0, reward=0.0, steps=[_step(2, -0.1, 0.0)])

    batch = build_batch([a, b], skill_decile=9, gamma=1.0, lam=1.0)

    # Rows concatenate (2 + 1); trajectory a's terminal reward must not bleed
    # into trajectory b's advantage.
    assert batch.advantages.shape == (3,)
    assert torch.allclose(batch.advantages, torch.tensor([9.0, 9.0, 0.0]))
    assert torch.allclose(batch.returns, torch.tensor([10.0, 10.0, 0.0]))


def test_ppo_update_raises_probability_of_a_positive_advantage_action():
    torch.manual_seed(0)
    feature_dim = 16
    action_dim = HEAD_LOGIT_DIMS["play"]
    model = BCModel(
        feature_dim, skill_dim=8, trunk_hidden=32, trunk_depth=1,
        trunk_out_dim=16, head_hidden=16,
    )
    critic = ValueBaseline(feature_dim, hidden=16)
    bc_model = copy.deepcopy(model)  # frozen BC reference for the KL anchor

    # Two transitions at the SAME state, two legal Intents (5, 9): Intent 5 has
    # positive advantage, Intent 9 negative.
    features = torch.randn(1, feature_dim).repeat(2, 1)
    skill = torch.full((2,), 9, dtype=torch.long)
    masks = torch.zeros(2, action_dim, dtype=torch.bool)
    masks[:, [5, 9]] = True
    actions = torch.tensor([5, 9])
    advantages = torch.tensor([1.0, -1.0])
    returns = torch.tensor([1.0, 1.0])

    with torch.no_grad():
        logits0 = model(features, skill)["play"]
        logp0 = torch.log_softmax(logits0.masked_fill(~masks, float("-inf")), dim=-1)
        old_logp = logp0.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    before = _legal_prob(model, features[0], skill[0], masks[0], 5)

    batch = PPOBatch(
        features=features, skill=skill, legal_masks=masks, actions=actions,
        old_logp=old_logp, advantages=advantages, returns=returns,
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(critic.parameters()), lr=0.05
    )
    ppo_update(
        model, critic, bc_model, optimizer, batch,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, kl_coef=0.0, epochs=20,
    )

    after = _legal_prob(model, features[0], skill[0], masks[0], 5)
    assert after > before


def test_kl_anchor_ignores_illegal_actions():
    # Adding an illegal third action with wildly different BC mass must not
    # change the KL — masked actions carry zero probability on both sides.
    new_logits = torch.tensor([[0.0, 0.0, 5.0]])
    bc_logits = torch.tensor([[0.0, math.log(3.0), -4.0]])
    masks = torch.tensor([[True, True, False]])

    kl = kl_anchor_loss(new_logits, bc_logits, masks)

    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert torch.allclose(kl, torch.tensor(expected), atol=1e-6)
