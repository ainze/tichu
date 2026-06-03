"""Batched play-policy sampler for PPO Refine (ADR-0029).

The sampler runs the policy forward once over a batch, masks illegal Intents,
samples, and reports the action log-probs (and critic values) PPO needs.
"""

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.heads import HEAD_LOGIT_DIMS, BCModel
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.ppo.policy import BatchedPolicy, sample_masked
from tichu_training.ppo.rollout import collect_rollout


def _tiny_policy() -> BatchedPolicy:
    model = BCModel(
        FEATURIZER_OUTPUT_DIM,
        skill_dim=8,
        trunk_hidden=32,
        trunk_depth=1,
        trunk_out_dim=16,
        head_hidden=16,
    )
    critic = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=16)
    return BatchedPolicy(model, critic, skill_decile=9)


def test_batched_policy_drives_a_full_round_with_only_legal_actions():
    torch.manual_seed(0)
    pos = generate_full_position_pool(seed=0, n=1)[0]

    trajs = collect_rollout([pos], _tiny_policy(), learner_team=0)

    # A Round that completes proves every sampled Intent resolved to a legal
    # action (the engine raises ValueError on an illegal step), and the learner
    # recorded real PPO bookkeeping at each of its Play Decisions.
    assert {t.seat for t in trajs} == {0, 2}
    for t in trajs:
        assert len(t.steps) > 0
        for s in t.steps:
            assert isinstance(s.intent_index, int)
            assert s.logprob is not None and s.value is not None


def test_sample_masked_respects_legal_mask_and_reports_logprobs():
    action_dim = 8
    logits = torch.zeros(2, action_dim)
    logits[0] = torch.tensor([5.0, 9.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    logits[1] = torch.arange(action_dim, dtype=torch.float) * 0.1

    masks = torch.zeros(2, action_dim, dtype=torch.bool)
    masks[0, 3] = True            # row 0: a single legal Intent
    masks[1, [1, 4, 7]] = True    # row 1: several legal Intents

    gen = torch.Generator().manual_seed(7)
    idx, logp = sample_masked(logits, masks, generator=gen)

    assert idx.shape == (2,)
    # Row 0 has one legal action -> it must be chosen, with log-prob 0 (p=1),
    # even though index 1 has the highest raw logit.
    assert idx[0].item() == 3
    assert torch.allclose(logp[0], torch.tensor(0.0), atol=1e-6)

    # Row 1: the sampled index is legal, and its log-prob equals the masked
    # log-softmax (illegal Intents carry zero probability mass).
    assert bool(masks[1, idx[1]])
    expected = torch.log_softmax(
        logits[1].masked_fill(~masks[1], float("-inf")), dim=-1
    )
    assert torch.allclose(logp[1], expected[idx[1]], atol=1e-6)


def test_batched_policy_returns_actions_logprobs_and_critic_values():
    torch.manual_seed(0)
    feature_dim = 16
    model = BCModel(
        feature_dim,
        skill_dim=8,
        trunk_hidden=32,
        trunk_depth=1,
        trunk_out_dim=16,
        head_hidden=16,
    )
    critic = ValueBaseline(feature_dim, hidden=16)
    policy = BatchedPolicy(model, critic)

    batch = 5
    features = torch.randn(batch, feature_dim)
    skill = torch.full((batch,), 9, dtype=torch.long)  # decile-9 (ADR-0029)
    masks = torch.zeros(batch, HEAD_LOGIT_DIMS["play"], dtype=torch.bool)
    masks[:, :3] = True  # first three Intents legal in every row

    out = policy.sample(features, skill, masks, generator=torch.Generator().manual_seed(1))

    assert out.indices.shape == (batch,)
    assert out.logprobs.shape == (batch,)
    assert out.values.shape == (batch,)
    # A sampled Intent is always legal.
    assert bool(masks[torch.arange(batch), out.indices].all())
    # The value is the separate critic's estimate on these features.
    with torch.no_grad():
        assert torch.allclose(out.values, critic(features))


def test_sample_masked_is_deterministic_under_a_seed():
    torch.manual_seed(0)
    logits = torch.randn(16, 12)
    masks = torch.zeros(16, 12, dtype=torch.bool)
    masks[:, :5] = True  # first five Intents legal in every row

    a_idx, a_logp = sample_masked(logits, masks, generator=torch.Generator().manual_seed(99))
    b_idx, b_logp = sample_masked(logits, masks, generator=torch.Generator().manual_seed(99))

    assert torch.equal(a_idx, b_idx)
    assert torch.allclose(a_logp, b_logp)
