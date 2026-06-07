"""PPO Refine training-loop orchestration (ADR-0029).

Wraps the verified single-step primitives (`collect_rollout` ->
`build_batch` -> `ppo_update`) into an iteration loop, with an adaptive
KL-anchor controller that steers `beta_KL` toward a target KL-to-BC budget —
the human-plausibility leash that keeps the sharpened policy from drifting too
far from BC.
"""

from typing import Callable

from tichu_training.ppo.policy import BatchedPolicy
from tichu_training.ppo.rollout import collect_rollout
from tichu_training.ppo.update import build_batch, ppo_update


class AdaptiveKLController:
    """Schulman-style adaptive KL coefficient with an optional annealed target.

    Holds the current `beta_KL`. After each update, `update(measured_kl)` nudges
    it toward keeping KL-to-BC near the current `target`: above the band it
    multiplies by `factor`, below it divides, otherwise holds. Clamped to
    `[min_coef, max_coef]`.

    **Target annealing (ADR-0034 follow-up).** The cotrain_v5 diagnosis found the
    Schupfen net pinned at BC because its KL target (0.008) is anchored tightest —
    it can't drift to a better joint optimum. When `target_final` is set, the
    effective target is raised **linearly** from `target` to `target_final` across
    `[anneal_start, anneal_start + anneal_iters]` (held flat outside), loosening
    the leash gradually so the net can move without the immediate crater of
    unleashing it. The schedule is a pure function of the **global iteration**
    (`set_iteration(it)` / `target_at(it)`), so it is **resume-safe**: only `coef`
    persists in the Resume Bundle; the target is reconstructed from config + the
    restored iteration counter. With `target_final=None` the behaviour is
    identical to the fixed-target controller (the play-only `train_ppo` path).
    """

    def __init__(
        self,
        *,
        coef: float,
        target: float,
        factor: float = 2.0,
        min_coef: float = 1e-4,
        max_coef: float = 1e4,
        target_final: float | None = None,
        anneal_start: int = 0,
        anneal_iters: int = 0,
    ) -> None:
        self.coef = float(coef)
        self.target_initial = float(target)
        self.target = float(target)  # current effective target (annealed in place)
        self.factor = float(factor)
        self.min_coef = float(min_coef)
        self.max_coef = float(max_coef)
        self.target_final = None if target_final is None else float(target_final)
        self.anneal_start = int(anneal_start)
        self.anneal_iters = int(anneal_iters)

    def target_at(self, iteration: int) -> float:
        """The effective KL target at global `iteration` (linear anneal, clamped)."""
        if self.target_final is None or self.anneal_iters <= 0:
            return self.target_initial
        frac = (iteration - self.anneal_start) / self.anneal_iters
        frac = min(1.0, max(0.0, frac))
        return self.target_initial + frac * (self.target_final - self.target_initial)

    def set_iteration(self, iteration: int) -> float:
        """Advance the annealed target to global `iteration`; returns the new target."""
        self.target = self.target_at(iteration)
        return self.target

    def update(self, measured_kl: float) -> float:
        if measured_kl > self.target * 1.5:
            self.coef = min(self.coef * self.factor, self.max_coef)
        elif measured_kl < self.target / 1.5:
            self.coef = max(self.coef / self.factor, self.min_coef)
        return self.coef


def train_ppo(
    model,
    critic,
    bc_model,
    sample_positions: Callable[[int], list],
    *,
    optimizer,
    kl_controller: AdaptiveKLController,
    iterations: int,
    gamma: float,
    lam: float,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    ppo_epochs: int,
    skill_decile: int = 9,
    learner_team: int = 0,
    opponent_policy_provider: Callable[[int], object] | None = None,
    on_iteration: Callable[[int, dict], None] | None = None,
    perfect_info: bool = False,
) -> list[dict[str, float]]:
    """Run `iterations` of PPO Refine self-play.

    Each iteration: build a fresh learner `BatchedPolicy` over the live weights,
    sample Starting Positions, roll out self-play (opponents from
    `opponent_policy_provider`, else pure self-play), GAE-flatten, and run
    `ppo_epochs` of `ppo_update` with the controller's current `beta_KL`. The
    measured KL-to-BC then steers `beta_KL` for the next iteration. `on_iteration`
    is the hook for snapshotting into a league and logging the caller-passivity
    dials. Returns per-iteration stats (including the `kl_coef` actually used).
    """
    history: list[dict[str, float]] = []
    for it in range(iterations):
        policy = BatchedPolicy(
            model, critic, skill_decile=skill_decile, perfect_info=perfect_info
        )
        opponent = (
            opponent_policy_provider(it) if opponent_policy_provider is not None
            else policy
        )
        positions = sample_positions(it)
        trajectories = collect_rollout(
            positions, policy, opponent_policy=opponent, learner_team=learner_team
        )
        batch = build_batch(trajectories, skill_decile=skill_decile, gamma=gamma, lam=lam)

        used_coef = kl_controller.coef
        stats = ppo_update(
            model, critic, bc_model, optimizer, batch,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
            kl_coef=used_coef, epochs=ppo_epochs,
        )
        stats["kl_coef"] = used_coef
        kl_controller.update(stats["kl_to_bc"])

        history.append(stats)
        if on_iteration is not None:
            on_iteration(it, stats)
    return history
