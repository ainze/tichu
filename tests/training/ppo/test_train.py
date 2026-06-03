"""PPO Refine training-loop orchestration (ADR-0029): the adaptive KL-anchor
controller and the collect -> build -> update iteration loop."""

import copy
import math

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.heads import BCModel
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.ppo.train import AdaptiveKLController, train_ppo


def test_adaptive_kl_raises_coef_when_measured_kl_exceeds_target():
    ctrl = AdaptiveKLController(coef=1.0, target=0.01)
    new = ctrl.update(measured_kl=0.1)  # well above 1.5 * target
    assert new > 1.0
    assert ctrl.coef == new


def test_adaptive_kl_lowers_coef_when_measured_kl_below_target():
    ctrl = AdaptiveKLController(coef=1.0, target=0.01)
    new = ctrl.update(measured_kl=0.0001)  # well below target / 1.5
    assert new < 1.0


def test_adaptive_kl_holds_coef_inside_the_band():
    ctrl = AdaptiveKLController(coef=1.0, target=0.01)
    new = ctrl.update(measured_kl=0.01)  # at target -> within the dead band
    assert new == 1.0


def test_adaptive_kl_clamps_to_bounds():
    ctrl = AdaptiveKLController(coef=1.0, target=0.01, factor=2.0, max_coef=1.5)
    assert ctrl.update(measured_kl=10.0) == 1.5  # would be 2.0, clamped to max


def _tiny_model():
    return BCModel(
        FEATURIZER_OUTPUT_DIM, skill_dim=8, trunk_hidden=32, trunk_depth=1,
        trunk_out_dim=16, head_hidden=16,
    )


def test_train_ppo_runs_iterations_adapts_kl_and_invokes_callback():
    torch.manual_seed(0)
    model = _tiny_model()
    critic = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=16)
    bc_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(critic.parameters()), lr=1e-3
    )
    # Near-zero KL target: any drift from BC exceeds it, so beta_KL keeps rising.
    ctrl = AdaptiveKLController(coef=1.0, target=1e-9, factor=2.0)
    seen_iters: list[int] = []

    history = train_ppo(
        model, critic, bc_model,
        sample_positions=lambda it: generate_full_position_pool(seed=10 * it, n=2),
        optimizer=optimizer,
        kl_controller=ctrl,
        iterations=3,
        gamma=1.0, lam=0.95,
        clip_eps=0.1, vf_coef=0.5, ent_coef=0.01, ppo_epochs=2,
        skill_decile=9, learner_team=0,
        on_iteration=lambda it, stats: seen_iters.append(it),
    )

    assert len(history) == 3
    assert seen_iters == [0, 1, 2]
    for s in history:
        assert "kl_coef" in s and math.isfinite(s["loss"])
    # The controller raised beta_KL across iterations (KL always beat the target).
    assert history[-1]["kl_coef"] > history[0]["kl_coef"]
