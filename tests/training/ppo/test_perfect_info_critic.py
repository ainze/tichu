"""Asymmetric Perfect-Info Critic for PPO Refine (ADR-0033).

The critic sees all four hands (392-dim perfect-info features) at training time;
the policy stays observable (224). These tests cover the additive plumbing that
carries the critic's features separately from the policy's through
rollout -> trajectory -> batch -> update, with `critic_features=None` falling
back to the symmetric path (byte-identical to ADR-0029).
"""

from __future__ import annotations

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_ml.rule_agent import RuleAgent
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.heads import BCModel
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.policy import BatchedPolicy
from tichu_training.ppo.rollout import PlayChoice, TrajectoryStep, collect_rollout
from tichu_training.ppo.train import AdaptiveKLController, train_ppo
from tichu_training.ppo.update import build_batch, ppo_update


def _tiny_model():
    return BCModel(FEATURIZER_OUTPUT_DIM, skill_dim=8, trunk_hidden=32,
                   trunk_depth=1, trunk_out_dim=16, head_hidden=16)


def test_playchoice_and_step_carry_optional_critic_features():
    # Default None preserves the symmetric path.
    assert PlayChoice(concrete_action=object()).critic_features is None
    assert TrajectoryStep(intent_index=0, logprob=0.0, value=0.0).critic_features is None
    # When set, they carry the perfect-info feature vector.
    pc = PlayChoice(concrete_action=object(), critic_features=[1.0, 2.0])
    assert pc.critic_features == [1.0, 2.0]


class _GameStateRecorder:
    """Plays RuleAgent moves; records the (seat, private_state, game_state)
    triples the driver hands it — the perfect-info critic needs all four hands."""

    def __init__(self) -> None:
        self._rule = RuleAgent()
        self.seen: list = []

    def act_play_batch(self, decisions):
        out = []
        for seat, private_state, game_state in decisions:
            self.seen.append((seat, private_state, game_state))
            out.append(PlayChoice(concrete_action=self._rule.act(private_state)))
        return out


def test_rollout_passes_full_gamestate_to_the_policy():
    pos = generate_full_position_pool(seed=0, n=1)[0]
    rec = _GameStateRecorder()

    collect_rollout([pos], rec, learner_team=0)

    assert rec.seen
    for seat, private_state, game_state in rec.seen:
        # All four hands are visible to the critic...
        assert len(game_state.hands) == 4
        # ...and the triple is the live decision (its private view matches).
        assert game_state.private_view(seat).hand == private_state.hand


def test_perfect_info_policy_values_from_392_features_and_records_them():
    import torch

    torch.manual_seed(0)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)  # 392-input critic
    policy = BatchedPolicy(_tiny_model(), critic, skill_decile=9, perfect_info=True)
    pos = generate_full_position_pool(seed=0, n=1)[0]

    trajs = collect_rollout([pos], policy, learner_team=0)

    # The asymmetric critic's input is recorded per learner step at 392 dims,
    # and a value (from those features) is present for GAE / the update.
    assert any(t.steps for t in trajs)
    for t in trajs:
        for s in t.steps:
            assert s.value is not None
            assert s.critic_features is not None
            assert tuple(s.critic_features.shape) == (PERFECT_INFO_DIM,)
            # the policy's own re-forward features stay observable (224)
            assert tuple(s.features.shape) == (FEATURIZER_OUTPUT_DIM,)


def test_symmetric_policy_leaves_critic_features_none():
    # ADR-0029 path unchanged: no perfect_info => critic_features stays None.
    critic = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=16)
    policy = BatchedPolicy(_tiny_model(), critic, skill_decile=9)
    pos = generate_full_position_pool(seed=0, n=1)[0]

    trajs = collect_rollout([pos], policy, learner_team=0)

    assert any(t.steps for t in trajs)
    for t in trajs:
        for s in t.steps:
            assert s.value is not None
            assert s.critic_features is None


def test_build_batch_separates_critic_features_and_falls_back_when_symmetric():
    import torch

    torch.manual_seed(0)
    positions = generate_full_position_pool(seed=0, n=2)

    # Asymmetric: critic features (392) are carried separately from policy (224).
    pi_policy = BatchedPolicy(_tiny_model(), ValueBaseline(PERFECT_INFO_DIM, hidden=16),
                              skill_decile=9, perfect_info=True)
    pi_batch = build_batch(collect_rollout(positions, pi_policy, learner_team=0),
                           skill_decile=9, gamma=1.0, lam=0.95)
    assert pi_batch.features.shape[1] == FEATURIZER_OUTPUT_DIM
    assert pi_batch.critic_features.shape[1] == PERFECT_INFO_DIM
    assert pi_batch.critic_features.shape[0] == pi_batch.features.shape[0]

    # Symmetric: critic features fall back to the observable features.
    sym_policy = BatchedPolicy(_tiny_model(), ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=16),
                               skill_decile=9)
    sym_batch = build_batch(collect_rollout(positions, sym_policy, learner_team=0),
                            skill_decile=9, gamma=1.0, lam=0.95)
    assert sym_batch.critic_features.shape[1] == FEATURIZER_OUTPUT_DIM
    assert torch.equal(sym_batch.critic_features, sym_batch.features)


def test_ppo_update_values_perfect_info_critic_without_shape_clash():
    import copy
    import math

    import torch

    torch.manual_seed(0)
    positions = generate_full_position_pool(seed=0, n=3)
    model = _tiny_model()
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)   # 392-input critic
    bc_model = copy.deepcopy(model)
    policy = BatchedPolicy(model, critic, skill_decile=9, perfect_info=True)

    batch = build_batch(collect_rollout(positions, policy, learner_team=0),
                        skill_decile=9, gamma=1.0, lam=0.95)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(critic.parameters()), lr=1e-4
    )
    stats = ppo_update(
        model, critic, bc_model, optimizer, batch,
        clip_eps=0.1, vf_coef=0.5, ent_coef=0.01, kl_coef=0.5, epochs=2,
    )

    # Policy re-forwards on 224, value on 392 — no shape clash, finite stats.
    assert set(stats) == {"loss", "policy_loss", "value_loss", "entropy", "kl_to_bc"}
    assert all(math.isfinite(v) for v in stats.values())


def test_train_ppo_runs_end_to_end_with_perfect_info_critic():
    import copy
    import math

    import torch

    torch.manual_seed(0)
    model = _tiny_model()
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    bc_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(critic.parameters()), lr=1e-4
    )
    positions = generate_full_position_pool(seed=0, n=2)

    history = train_ppo(
        model, critic, bc_model, lambda _it: positions,
        optimizer=optimizer,
        kl_controller=AdaptiveKLController(coef=0.5, target=0.02),
        iterations=2, gamma=1.0, lam=0.95, clip_eps=0.1,
        vf_coef=0.5, ent_coef=0.01, ppo_epochs=1, perfect_info=True,
    )

    assert len(history) == 2
    assert all(math.isfinite(h["value_loss"]) for h in history)
