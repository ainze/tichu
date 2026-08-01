"""Batched play-policy sampler for PPO Refine (ADR-0029).

The sampler runs the policy forward once over a batch, masks illegal Intents,
samples, and reports the action log-probs (and critic values) PPO needs.
"""

import copy
import math

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.heads import HEAD_LOGIT_DIMS, BCModel
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.ppo.policy import BatchedPolicy, sample_masked
from tichu_training.ppo.rollout import collect_rollout
from tichu_training.ppo.update import build_batch, ppo_update


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


def test_real_policy_records_features_and_mask_for_the_update():
    torch.manual_seed(0)
    pos = generate_full_position_pool(seed=0, n=1)[0]

    trajs = collect_rollout([pos], _tiny_policy(), learner_team=0)

    # PPO's update re-runs the forward at each visited state, so every learner
    # step must carry the featurized state and its legal-Intent mask.
    for t in trajs:
        for s in t.steps:
            assert s.features is not None
            assert tuple(s.features.shape) == (FEATURIZER_OUTPUT_DIM,)
            assert s.legal_mask is not None
            assert tuple(s.legal_mask.shape) == (HEAD_LOGIT_DIMS["play"],)
            # the recorded action is legal under the recorded mask
            assert bool(s.legal_mask[s.intent_index])


def test_full_training_step_rollout_build_batch_then_ppo_update():
    torch.manual_seed(0)
    positions = generate_full_position_pool(seed=0, n=4)

    model = BCModel(
        FEATURIZER_OUTPUT_DIM, skill_dim=8, trunk_hidden=32, trunk_depth=1,
        trunk_out_dim=16, head_hidden=16,
    )
    critic = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=16)
    bc_model = copy.deepcopy(model)  # frozen anchor reference
    policy = BatchedPolicy(model, critic, skill_decile=9)

    trajs = collect_rollout(positions, policy, learner_team=0)
    batch = build_batch(trajs, skill_decile=9, gamma=1.0, lam=0.95)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(critic.parameters()), lr=1e-4
    )
    stats = ppo_update(
        model, critic, bc_model, optimizer, batch,
        clip_eps=0.1, vf_coef=0.5, ent_coef=0.01, kl_coef=0.5, epochs=2,
    )

    # The whole pipeline runs and produces finite, sane training stats.
    assert set(stats) == {"loss", "policy_loss", "value_loss", "entropy", "kl_to_bc"}
    assert all(math.isfinite(v) for v in stats.values())
    assert stats["kl_to_bc"] >= -1e-6  # KL divergence is non-negative
    assert stats["entropy"] >= 0.0


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


def _lead_state_with_bomb():
    """Seat 0 on lead: jade 3-7 is a straight-flush bomb, and the two off-suit
    5s make a pair of 5s realisable three ways — one Intent, three concretes."""
    from tichu_engine.cards import Card, Suit
    from tichu_engine.state import PrivateState, PublicState, Trick

    hand = frozenset(
        {Card(suit=Suit.JADE, rank=r) for r in range(3, 8)}
        | {Card(suit=Suit.SWORD, rank=5), Card(suit=Suit.STAR, rank=5)}
    )
    pub = PublicState(
        current_player=0, hand_sizes=(len(hand), 5, 5, 5), scores=(0, 0),
        trick=Trick(plays=(), leader=0),
    )
    return PrivateState(player=0, hand=hand, public=pub)


def test_rollout_resolves_the_sampled_intent_without_spending_the_bomb(monkeypatch):
    # The rollout resolver is separate from serving's: a sampled Intent index has
    # to be turned back into concrete cards here too. It must make the same
    # bomb-sparing choice, or the policy trains against a world that plays its
    # own hand worse than the served agent does.
    from tichu_engine.cards import Card, Suit
    from tichu_engine.combinations import Pair
    from tichu_engine.legality import _cards_in
    import tichu_training.ppo.policy as pol
    from tichu_training.action_space import play_intent_index

    jade_five = Card(suit=Suit.JADE, rank=5)
    pv = _lead_state_with_bomb()
    pair_of_fives = play_intent_index(
        Pair(Card(suit=Suit.SWORD, rank=5), Card(suit=Suit.STAR, rank=5))
    )

    # Hand the resolver the bomb-breaking variant first; otherwise the frozenset
    # iterates favourably and this passes with or without a resolver.
    real = pol.legal_actions_for
    monkeypatch.setattr(
        pol, "legal_actions_for",
        lambda ps: sorted(real(ps), key=lambda a: 0 if jade_five in _cards_in(a) else 1),
    )

    class _Peaked:
        def __call__(self, features, skill):
            play = torch.full((features.shape[0], HEAD_LOGIT_DIMS["play"]), -10.0)
            play[:, pair_of_fives] = 10.0
            return {"play": play}

    class _ZeroCritic:
        def __call__(self, features):
            return torch.zeros(features.shape[0])

    [choice] = BatchedPolicy(_Peaked(), _ZeroCritic()).act_play_batch([(0, pv, None)])

    assert choice.intent_index == pair_of_fives
    assert jade_five not in _cards_in(choice.concrete_action)
