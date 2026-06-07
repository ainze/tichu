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


def test_target_constant_without_anneal_config():
    """Backward compat: no anneal params -> target is the fixed initial at every iter."""
    ctrl = AdaptiveKLController(coef=1.0, target=0.008)
    assert ctrl.target_at(0) == 0.008
    assert ctrl.target_at(10_000) == 0.008
    assert ctrl.target == 0.008


def test_target_anneals_linearly_and_clamps_at_both_ends():
    """Schupfen leash loosening (ADR-0034 follow-up): linearly raise the KL target
    from `target` to `target_final` over [anneal_start, anneal_start+anneal_iters],
    held flat outside that window."""
    ctrl = AdaptiveKLController(
        coef=1.0, target=0.008, target_final=0.028,
        anneal_start=100, anneal_iters=200,
    )
    assert ctrl.target_at(50) == 0.008                      # before window -> initial
    assert ctrl.target_at(100) == 0.008                     # at start -> initial
    assert abs(ctrl.target_at(200) - 0.018) < 1e-9          # midpoint -> halfway
    assert abs(ctrl.target_at(300) - 0.028) < 1e-9          # at end -> final
    assert ctrl.target_at(10_000) == 0.028                  # after window -> final (clamped)


def test_set_iteration_loosens_the_band_so_beta_stops_growing():
    """A measured KL that was ABOVE the band at the tight initial target falls INSIDE
    the band once the target has annealed up -> beta no longer doubles (the leash
    loosens, letting schupfen drift further from BC)."""
    ctrl = AdaptiveKLController(
        coef=1.0, target=0.008, target_final=0.028,
        anneal_start=0, anneal_iters=100,
    )
    # iter 0: target 0.008, measured 0.02 is > 1.5*target -> beta grows
    ctrl.set_iteration(0)
    assert ctrl.update(measured_kl=0.02) == 2.0
    # iter 100: target annealed to 0.028, 0.02 now within [target/1.5, target*1.5] -> hold
    ctrl.set_iteration(100)
    before = ctrl.coef
    assert ctrl.update(measured_kl=0.02) == before


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
