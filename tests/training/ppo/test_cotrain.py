"""Full-stack co-training PPO primitives (ADR-0034).

The co-training run sharpens play + Schupfen + Tichu/Grand calls together. These
tests pin the genuinely new pieces: the Schupfen Network's without-replacement
action sampling (3 distinct in-hand cards across the 3 direction heads) and the
matching log-prob re-evaluator the PPO update re-forwards with.
"""

import copy
import math

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.heads import HEAD_LOGIT_DIMS, BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.card_slots import CARD_SLOTS
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.cotrain import (
    BatchedCoTrainPolicy,
    build_cotrain_batch,
    cotrain_update,
    sample_schupfen,
    schupfen_logp,
)
from tichu_training.ppo.rollout import Trajectory, TrajectoryStep, collect_rollout

_PLAY_DIM = HEAD_LOGIT_DIMS["play"]


def _tiny_cotrain_policy(*, perfect_info=True, seed=0):
    play = BCModel(FEATURIZER_OUTPUT_DIM, skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16)
    schupfen = SchupfenNetwork(FEATURIZER_OUTPUT_DIM, skill_dim=8, hidden=16)
    tichu = TichuCallNetwork(FEATURIZER_OUTPUT_DIM, skill_dim=8, hidden=16)
    grand = GrandTichuCallNetwork(FEATURIZER_OUTPUT_DIM, skill_dim=8, hidden=16)
    critic = ValueBaseline(PERFECT_INFO_DIM if perfect_info else FEATURIZER_OUTPUT_DIM, hidden=16)
    return BatchedCoTrainPolicy(
        play, schupfen, tichu, grand, critic,
        skill_decile=9, perfect_info=perfect_info,
        generator=torch.Generator().manual_seed(seed),
    )


def _hand_mask(rows, slots):
    mask = torch.zeros(rows, CARD_SLOTS, dtype=torch.bool)
    for r, ss in enumerate(slots):
        mask[r, list(ss)] = True
    return mask


def test_sample_schupfen_picks_three_distinct_in_hand_slots():
    torch.manual_seed(0)
    head_logits = torch.randn(4, 3, CARD_SLOTS)
    hand = _hand_mask(4, [range(5), range(3, 9), [0, 7, 13, 40, 55], range(10)])

    idx, logp = sample_schupfen(head_logits, hand, generator=torch.Generator().manual_seed(1))

    assert idx.shape == (4, 3)
    assert logp.shape == (4,)
    for r in range(4):
        chosen = idx[r].tolist()
        assert len(set(chosen)) == 3, "the 3 schupfen cards must be distinct"
        for slot in chosen:
            assert bool(hand[r, slot]), "every schupfen card must be in hand"


def test_schupfen_logp_reproduces_the_sampled_logprob():
    # The update re-forwards the nets and must recompute the EXACT log-prob of the
    # sampled without-replacement action — otherwise the PPO importance ratio is wrong.
    torch.manual_seed(0)
    head_logits = torch.randn(6, 3, CARD_SLOTS)
    hand = _hand_mask(6, [range(4, 12)] * 6)

    idx, sampled_logp = sample_schupfen(head_logits, hand, generator=torch.Generator().manual_seed(2))
    replayed = schupfen_logp(head_logits, hand, idx)

    assert torch.allclose(replayed, sampled_logp, atol=1e-6)


def test_sample_schupfen_is_deterministic_under_a_seed():
    torch.manual_seed(0)
    head_logits = torch.randn(8, 3, CARD_SLOTS)
    hand = _hand_mask(8, [range(14)] * 8)

    a_idx, a_logp = sample_schupfen(head_logits, hand, generator=torch.Generator().manual_seed(5))
    b_idx, b_logp = sample_schupfen(head_logits, hand, generator=torch.Generator().manual_seed(5))

    assert torch.equal(a_idx, b_idx)
    assert torch.allclose(a_logp, b_logp)


_F, _FC = 4, 6  # observable (policy) dim, perfect-info (critic) dim


def _mk_step(decision_type, action, mask, value, logp):
    feat = torch.ones(_F) * value
    crit = torch.ones(_FC) * value
    return TrajectoryStep(action, logp, value, feat, mask, crit, decision_type=decision_type)


def test_build_cotrain_batch_groups_steps_by_decision_type():
    play_mask = torch.zeros(_PLAY_DIM, dtype=torch.bool)
    play_mask[7] = True
    hand_mask = torch.zeros(CARD_SLOTS, dtype=torch.bool)
    hand_mask[[1, 2, 3, 4]] = True

    traj = Trajectory(
        seat=0, team=0, reward=10.0,
        steps=[
            _mk_step("grand", 1, None, 0.5, -1.0),
            _mk_step("schupfen", torch.tensor([1, 2, 3]), hand_mask, 0.4, -2.0),
            _mk_step("play", 7, play_mask, 0.2, -0.5),
            _mk_step("play", 7, play_mask, 0.3, -0.6),
            _mk_step("tichu", 0, None, 0.1, -0.3),
        ],
    )
    batch = build_cotrain_batch([traj], skill_decile=9, gamma=1.0, lam=0.95)

    # One sub-batch per decision type present, each net re-forwards on observable feats.
    assert set(batch.nets) == {"grand", "schupfen", "play", "tichu"}
    assert batch.nets["play"].features.shape == (2, _F)
    assert batch.nets["play"].actions.tolist() == [7, 7]
    assert batch.nets["play"].masks.shape == (2, _PLAY_DIM)
    # Schupfen carries its 3-card action and the hand mask.
    assert batch.nets["schupfen"].actions.shape == (1, 3)
    assert batch.nets["schupfen"].masks.shape == (1, CARD_SLOTS)
    # Binary calls have no action mask.
    assert batch.nets["grand"].actions.tolist() == [1]
    assert batch.nets["grand"].masks is None

    # The SHARED critic pools every step (all 5) on perfect-info features + returns.
    assert batch.critic_features.shape == (5, _FC)
    assert batch.returns.shape == (5,)
    # GAE identity holds on the pooled steps (returns = advantages + values).
    pooled_adv = torch.cat([nb.advantages for nb in batch.nets.values()])
    assert pooled_adv.shape == (5,)


def _cotrain_batch_with_all_nets():
    play_mask = torch.zeros(_PLAY_DIM, dtype=torch.bool)
    play_mask[[5, 7, 9]] = True
    hand_mask = torch.zeros(CARD_SLOTS, dtype=torch.bool)
    hand_mask[[1, 2, 3, 4, 5]] = True
    trajs = []
    for r in range(4):  # a few seats so each net sees >1 sample
        trajs.append(Trajectory(
            seat=0, team=0, reward=float(10 - r * 7),
            steps=[
                _mk_step("grand", r % 2, None, 0.5, -1.0),
                _mk_step("schupfen", torch.tensor([1, 2, 3]), hand_mask, 0.4, -2.0),
                _mk_step("play", 7, play_mask, 0.2, -0.5),
                _mk_step("tichu", (r + 1) % 2, None, 0.1, -0.3),
            ],
        ))
    return build_cotrain_batch(trajs, skill_decile=9, gamma=1.0, lam=0.95)


def test_cotrain_update_moves_every_net_and_the_shared_critic():
    torch.manual_seed(0)
    models = {
        "play": BCModel(_F, skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
        "schupfen": SchupfenNetwork(_F, skill_dim=8, hidden=16),
        "tichu": TichuCallNetwork(_F, skill_dim=8, hidden=16),
        "grand": GrandTichuCallNetwork(_F, skill_dim=8, hidden=16),
    }
    bc_models = {k: copy.deepcopy(m) for k, m in models.items()}  # frozen anchors
    critic = ValueBaseline(_FC, hidden=16)
    batch = _cotrain_batch_with_all_nets()

    params = list(critic.parameters())
    for m in models.values():
        params += list(m.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)

    before = {k: copy.deepcopy(m.state_dict()) for k, m in models.items()}
    crit_before = copy.deepcopy(critic.state_dict())

    stats = cotrain_update(
        models, bc_models, critic, optimizer, batch,
        clip_eps=0.1, vf_coef=0.5,
        ent_coefs={"play": 0.01, "schupfen": 0.01, "tichu": 0.01, "grand": 0.01},
        kl_coefs={"play": 0.5, "schupfen": 0.5, "tichu": 0.5, "grand": 0.5},
        epochs=2,
    )

    # Every net's policy stats + the shared critic loss are finite and sane.
    for dt in ("play", "schupfen", "tichu", "grand"):
        assert math.isfinite(stats[f"{dt}_policy_loss"])
        assert stats[f"{dt}_kl"] >= -1e-6      # KL is non-negative
        assert stats[f"{dt}_entropy"] >= -1e-6
    assert math.isfinite(stats["value_loss"]) and math.isfinite(stats["loss"])

    # The shared critic moved, and so did each policy net (gradient reached all four).
    assert any(not torch.equal(crit_before[k], v) for k, v in critic.state_dict().items())
    for dt, m in models.items():
        assert any(
            not torch.equal(before[dt][k], v) for k, v in m.state_dict().items()
        ), f"{dt} net did not update"


def test_cotrain_policy_drives_and_records_all_decision_types():
    torch.manual_seed(0)
    positions = generate_full_position_pool(seed=0, n=2)
    policy = _tiny_cotrain_policy(perfect_info=True, seed=1)

    trajs = collect_rollout(positions, policy, learner_team=0)

    types = {s.decision_type for t in trajs for s in t.steps}
    # The full stack is driven + recorded: play + schupfen always happen, and every
    # learner seat is asked Grand (recorded whether it calls or not).
    assert {"play", "schupfen", "grand"} <= types

    for t in trajs:
        for s in t.steps:
            assert s.logprob is not None and s.value is not None
            if s.decision_type == "schupfen":
                slots = s.intent_index.tolist()
                assert len(set(slots)) == 3, "schupfen action must be 3 distinct cards"
                assert tuple(s.legal_mask.shape) == (CARD_SLOTS,)
            elif s.decision_type in ("grand", "tichu"):
                assert int(s.intent_index) in (0, 1)
            else:  # play
                assert isinstance(int(s.intent_index), int)


def test_cotrain_policy_round_trips_through_build_and_update():
    # End-to-end: the policy's recorded trajectories flow through build_cotrain_batch
    # and one cotrain_update without shape errors (the full co-training step, ADR-0034).
    torch.manual_seed(0)
    positions = generate_full_position_pool(seed=1, n=3)
    policy = _tiny_cotrain_policy(perfect_info=True, seed=2)
    trajs = collect_rollout(positions, policy, learner_team=0)

    batch = build_cotrain_batch(trajs, skill_decile=9, gamma=1.0, lam=0.95)
    models = {
        "play": policy.play_model, "schupfen": policy.schupfen_model,
        "tichu": policy.call_models["tichu"], "grand": policy.call_models["grand"],
    }
    bc_models = {k: copy.deepcopy(m) for k, m in models.items()}
    params = list(policy.critic.parameters())
    for m in models.values():
        params += list(m.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    coefs = {dt: 0.01 for dt in models}
    kls = {dt: 0.5 for dt in models}
    stats = cotrain_update(
        models, bc_models, policy.critic, optimizer, batch,
        clip_eps=0.1, vf_coef=0.5, ent_coefs=coefs, kl_coefs=kls, epochs=1,
    )
    assert math.isfinite(stats["loss"]) and math.isfinite(stats["value_loss"])


def test_train_cotrain_runs_iterations_and_logs_per_net_stats():
    import copy as _copy

    from tichu_training.ppo.cotrain import train_cotrain
    from tichu_training.ppo.train import AdaptiveKLController

    torch.manual_seed(0)
    policy = _tiny_cotrain_policy(perfect_info=True, seed=3)
    models = {
        "play": policy.play_model, "schupfen": policy.schupfen_model,
        "tichu": policy.call_models["tichu"], "grand": policy.call_models["grand"],
    }
    bc_models = {k: _copy.deepcopy(m) for k, m in models.items()}
    params = list(policy.critic.parameters())
    for m in models.values():
        params += list(m.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)
    controllers = {dt: AdaptiveKLController(coef=0.5, target=0.02) for dt in models}

    seen = []
    history = train_cotrain(
        models, bc_models, policy.critic,
        lambda it: generate_full_position_pool(seed=it, n=2),
        optimizer=optimizer, kl_controllers=controllers,
        ent_coefs={dt: 0.01 for dt in models},
        iterations=2, gamma=1.0, lam=0.95, clip_eps=0.1, vf_coef=0.5, ppo_epochs=1,
        perfect_info=True, generator=torch.Generator().manual_seed(4),
        on_iteration=lambda it, st: seen.append(it),
    )

    assert len(history) == 2 and seen == [0, 1]
    for st in history:
        for dt in ("play", "schupfen", "tichu", "grand"):
            assert math.isfinite(st[f"{dt}_policy_loss"])
            assert f"{dt}_kl_coef" in st
        assert math.isfinite(st["value_loss"])
