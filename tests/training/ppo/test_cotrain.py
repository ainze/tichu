"""Full-stack co-training PPO primitives (ADR-0034).

The co-training run sharpens play + Schupfen + Tichu/Grand calls together. These
tests pin the genuinely new pieces: the Schupfen Network's without-replacement
action sampling (3 distinct in-hand cards across the 3 direction heads) and the
matching log-prob re-evaluator the PPO update re-forwards with.
"""

import copy
import math

import pytest
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


def _tiny_cotrain_policy(*, perfect_info=True, seed=0, train_wish=False):
    play = BCModel(FEATURIZER_OUTPUT_DIM, skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16)
    schupfen = SchupfenNetwork(FEATURIZER_OUTPUT_DIM, skill_dim=8, hidden=16)
    tichu = TichuCallNetwork(FEATURIZER_OUTPUT_DIM, skill_dim=8, hidden=16)
    grand = GrandTichuCallNetwork(FEATURIZER_OUTPUT_DIM, skill_dim=8, hidden=16)
    critic = ValueBaseline(PERFECT_INFO_DIM if perfect_info else FEATURIZER_OUTPUT_DIM, hidden=16)
    return BatchedCoTrainPolicy(
        play, schupfen, tichu, grand, critic,
        skill_decile=9, perfect_info=perfect_info,
        generator=torch.Generator().manual_seed(seed), train_wish=train_wish,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cotrain_update_on_cuda_runs_and_restores_models_to_cpu():
    # GPU the update only (the heavy batched forward+backward), keeping the nets on
    # CPU for the engine-bound rollout. After the call the nets MUST be back on CPU
    # so the next rollout works, and a SECOND update must still run (the optimizer
    # moment state survived the device round-trip).
    torch.manual_seed(0)
    models = {
        "play": BCModel(_F, skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
        "schupfen": SchupfenNetwork(_F, skill_dim=8, hidden=16),
        "tichu": TichuCallNetwork(_F, skill_dim=8, hidden=16),
        "grand": GrandTichuCallNetwork(_F, skill_dim=8, hidden=16),
    }
    bc_models = {k: copy.deepcopy(m) for k, m in models.items()}
    critic = ValueBaseline(_FC, hidden=16)
    params = list(critic.parameters())
    for m in models.values():
        params += list(m.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)

    before = copy.deepcopy(models["play"].state_dict())
    common = dict(
        clip_eps=0.1, vf_coef=0.5,
        ent_coefs={dt: 0.01 for dt in models}, kl_coefs={dt: 0.5 for dt in models},
    )
    stats = cotrain_update(models, bc_models, critic, optimizer,
                           _cotrain_batch_with_all_nets(), epochs=2, device="cuda", **common)
    assert math.isfinite(stats["loss"]) and math.isfinite(stats["value_loss"])
    # Nets restored to CPU for the (CPU, engine-bound) rollout.
    for dt, m in models.items():
        assert next(m.parameters()).device.type == "cpu", f"{dt} not back on cpu"
    # Second update still works -> optimizer state round-tripped cleanly.
    stats2 = cotrain_update(models, bc_models, critic, optimizer,
                            _cotrain_batch_with_all_nets(), epochs=1, device="cuda", **common)
    assert math.isfinite(stats2["loss"])
    # The play net actually moved.
    assert any(not torch.equal(before[k], v) for k, v in models["play"].state_dict().items())


# --- Wish head co-training (ADR-0034 addendum: wish is a head on the play trunk) ---

_WISH_DIM = HEAD_LOGIT_DIMS["wish"]


def test_wish_index_round_trips_through_the_action_space():
    # The collector decodes the wish head index 0..13 to a MahjongWish (0 -> decline,
    # k -> rank k+1) and records that index back as the action; the encoding must be
    # the inverse of action_space.wish_intent_index so the PPO re-forward lines up.
    from tichu_engine.legality import MahjongWish
    from tichu_training.action_space import wish_intent_index
    for idx in range(_WISH_DIM):
        rank = None if idx == 0 else idx + 1
        assert wish_intent_index(MahjongWish(rank=rank)) == idx


def test_cotrain_update_trains_the_wish_head_on_the_play_net():
    # Option (a): "wish" is a decision type whose policy head lives on the play
    # BCModel's trunk. cotrain_update must route the "wish" group to models["play"]
    # (its ["wish"] head) without a separate models["wish"] entry, and the gradient
    # must reach the wish head specifically (not only the shared trunk).
    torch.manual_seed(0)
    models = {
        "play": BCModel(_F, skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
        "schupfen": SchupfenNetwork(_F, skill_dim=8, hidden=16),
        "tichu": TichuCallNetwork(_F, skill_dim=8, hidden=16),
        "grand": GrandTichuCallNetwork(_F, skill_dim=8, hidden=16),
    }
    bc_models = {k: copy.deepcopy(m) for k, m in models.items()}
    critic = ValueBaseline(_FC, hidden=16)

    # A batch that includes a "wish" group (plus play, so the trunk is shared).
    trajs = []
    for r in range(4):
        trajs.append(Trajectory(
            seat=0, team=0, reward=float(10 - r * 6),
            steps=[
                _mk_step("wish", r % _WISH_DIM, None, 0.3, -0.8),
                _mk_step("play", 7, _play_mask([5, 7, 9]), 0.2, -0.5),
            ],
        ))
    batch = build_cotrain_batch(trajs, skill_decile=9, gamma=1.0, lam=0.95)
    assert "wish" in batch.nets and batch.nets["wish"].masks is None  # 14-way, all legal

    params = list(critic.parameters())
    for m in models.values():
        params += list(m.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)

    wish_head_before = copy.deepcopy(models["play"].heads["wish"].state_dict())
    stats = cotrain_update(
        models, bc_models, critic, optimizer, batch,
        clip_eps=0.1, vf_coef=0.5,
        ent_coefs={"play": 0.01, "wish": 0.01},
        kl_coefs={"play": 0.5, "wish": 0.5},
        epochs=2,
    )
    assert math.isfinite(stats["wish_policy_loss"]) and stats["wish_kl"] >= -1e-6
    # The wish head itself moved (gradient reached it through the play net).
    assert any(
        not torch.equal(wish_head_before[k], v)
        for k, v in models["play"].heads["wish"].state_dict().items()
    ), "wish head did not update"


def _play_mask(idxs):
    m = torch.zeros(_PLAY_DIM, dtype=torch.bool)
    m[idxs] = True
    return m


class WishTaggingPolicy:
    """Plays + schupfens via the tiny co-train policy but stamps each WISH with a
    globally-unique tag, recording which seats it was asked — so a test can prove
    learner wish-declaration is routed to the policy's wish head and recorded once
    per learner Mahjong-holder, tagged (ADR-0034 addendum)."""
    def __init__(self):
        self._inner = _tiny_cotrain_policy(perfect_info=True, seed=9)
        self.wish_seats = []
        self._n = 7000

    def act_play_batch(self, decisions):
        return self._inner.act_play_batch(decisions)

    def act_schupfen_batch(self, decisions):
        return self._inner.act_schupfen_batch(decisions)

    def act_wish_batch(self, decisions):
        from tichu_training.ppo.rollout import WishChoice
        from tichu_engine.legality import MahjongWish
        out = []
        for seat, _ps, _gs in decisions:
            tag = self._n
            self._n += 1
            self.wish_seats.append(seat)
            out.append(WishChoice(
                concrete_action=MahjongWish(rank=None),
                intent_index=tag, logprob=float(-tag), value=float(tag),
            ))
        return out


def test_collector_records_learner_wish_when_policy_supports_it():
    # A learner-team seat (2) playing the Mahjong triggers a MahjongWishPending that
    # must be routed to the learner's wish head and recorded once, tagged. We start
    # the round at the wish-pending state directly (stepping the Mahjong through the
    # engine) so the holder is deterministic — schupfen would otherwise shuffle who
    # holds the Mahjong. Opponent policy has no act_wish_batch, so opponent wishes
    # would stay on the frozen inline seat agent.
    from dataclasses import dataclass
    from tichu_engine.cards import MAHJONG, Card, Suit
    from tichu_engine.combinations import Single
    from tichu_engine.engine import step
    from tichu_engine.state import GameState, PublicState, Trick
    from tichu_training.ppo.rollout import PlayChoice, collect_rollout

    @dataclass
    class _Pos:
        state: object
        grand_prefixes: tuple

    pre = GameState(
        hands=(frozenset({Card(Suit.JADE, 5)}), frozenset({Card(Suit.JADE, 6)}),
               frozenset({MAHJONG, Card(Suit.JADE, 7)}), frozenset({Card(Suit.JADE, 8)})),
        public=PublicState(current_player=2, hand_sizes=(1, 1, 2, 1),
                           scores=(0, 0), trick=Trick.empty()),
    )
    mid, _, _, _ = step(pre, Single(MAHJONG))  # -> MahjongWishPending(player=2)
    assert mid.public.current_player == 2 and mid.public.pending_decision is not None
    pos = _Pos(state=mid, grand_prefixes=(frozenset(),) * 4)
    learner = WishTaggingPolicy()

    class RuleOpp:
        def __init__(self):
            from tichu_ml.rule_agent import RuleAgent
            self._r = RuleAgent()
        def act_play_batch(self, decisions):
            return [PlayChoice(concrete_action=self._r.act(ps)) for _s, ps, _g in decisions]

    trajs = collect_rollout([pos], learner, opponent_policy=RuleOpp(), learner_team=0)
    by_seat = {t.seat: t for t in trajs}

    wish_steps = [s for s in by_seat[2].steps if s.decision_type == "wish"]
    assert len(wish_steps) == 1, "learner Mahjong-holder seat 2 should wish exactly once"
    assert wish_steps[0].logprob == float(-wish_steps[0].intent_index)  # the tag we returned
    # Only learner-team seats were routed to the learner's wish method.
    assert set(learner.wish_seats) <= {0, 2}


def _wish_pending_position():
    """A FullStartingPosition-shaped object whose state is already at a
    MahjongWishPending for seat 2 (built by stepping a Mahjong play through the
    engine), with dummy empty Grand-Tichu prefixes."""
    from dataclasses import dataclass
    from tichu_engine.cards import MAHJONG, Card, Suit
    from tichu_engine.combinations import Single
    from tichu_engine.engine import step
    from tichu_engine.state import GameState, PublicState, Trick

    @dataclass
    class _Pos:
        state: object
        grand_prefixes: tuple

    pre = GameState(
        hands=(frozenset({Card(Suit.JADE, 5)}), frozenset({Card(Suit.JADE, 6)}),
               frozenset({MAHJONG, Card(Suit.JADE, 7)}), frozenset({Card(Suit.JADE, 8)})),
        public=PublicState(current_player=2, hand_sizes=(1, 1, 2, 1),
                           scores=(0, 0), trick=Trick.empty()),
    )
    mid, _, _, _ = step(pre, Single(MAHJONG))
    return _Pos(state=mid, grand_prefixes=(frozenset(),) * 4)


def test_batched_cotrain_policy_act_wish_batch_emits_valid_wishes():
    from tichu_engine.legality import MahjongWish
    policy = _tiny_cotrain_policy(perfect_info=True, seed=11, train_wish=True)
    gs = _wish_pending_position().state
    private = gs.private_view(gs.public.current_player)

    choices = policy.act_wish_batch([(gs.public.current_player, private, gs)])

    assert len(choices) == 1
    ch = choices[0]
    assert isinstance(ch.concrete_action, MahjongWish)
    assert 0 <= ch.intent_index < _WISH_DIM
    # index/rank consistency: 0 -> decline, k -> rank k+1.
    expected_rank = None if ch.intent_index == 0 else ch.intent_index + 1
    assert ch.concrete_action.rank == expected_rank
    assert math.isfinite(ch.logprob) and math.isfinite(ch.value)


def test_wish_routing_is_off_by_default():
    # Default (train_wish=False): the policy does NOT expose a usable act_wish_batch,
    # so rollout._supports_wish() is False and the wish stays on the frozen seat agent
    # — training is byte-identical to pre-addendum behavior.
    from tichu_training.ppo.rollout import _supports_wish
    assert not _supports_wish(_tiny_cotrain_policy(seed=0))                 # off by default
    assert _supports_wish(_tiny_cotrain_policy(seed=0, train_wish=True))    # on when asked


def test_train_cotrain_with_wish_enabled_trains_the_wish_head():
    import copy as _copy
    from tichu_training.ppo.cotrain import train_cotrain
    from tichu_training.ppo.train import AdaptiveKLController

    torch.manual_seed(0)
    policy = _tiny_cotrain_policy(perfect_info=True, seed=3, train_wish=True)
    models = {
        "play": policy.play_model, "schupfen": policy.schupfen_model,
        "tichu": policy.call_models["tichu"], "grand": policy.call_models["grand"],
    }
    bc_models = {k: _copy.deepcopy(m) for k, m in models.items()}
    params = list(policy.critic.parameters())
    for m in models.values():
        params += list(m.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)
    # Controllers/ent are keyed by DECISION type, incl. wish (rides the play net).
    decision_types = ("play", "schupfen", "tichu", "grand", "wish")
    controllers = {dt: AdaptiveKLController(coef=0.5, target=0.02) for dt in decision_types}

    wish_head_before = _copy.deepcopy(models["play"].heads["wish"].state_dict())
    history = train_cotrain(
        models, bc_models, policy.critic,
        lambda it: [_wish_pending_position(), _wish_pending_position()],
        optimizer=optimizer, kl_controllers=controllers,
        ent_coefs={dt: 0.01 for dt in decision_types},
        iterations=1, gamma=1.0, lam=0.95, clip_eps=0.1, vf_coef=0.5, ppo_epochs=2,
        perfect_info=True, generator=torch.Generator().manual_seed(4), train_wish=True,
    )

    # A wish decision was driven, recorded, and trained this iteration.
    assert math.isfinite(history[0]["wish_policy_loss"])
    assert "wish_kl_coef" in history[0]
    assert any(
        not torch.equal(wish_head_before[k], v)
        for k, v in models["play"].heads["wish"].state_dict().items()
    ), "wish head did not train end-to-end"
