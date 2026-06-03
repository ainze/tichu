"""Opponent league for PPO Refine (ADR-0029).

Opponents start as the frozen BC policy and grow to a small league: periodic
frozen snapshots of the improving learner, sampled as the opposing seats. Once
the learner bombs well, the league forces opponents to respect bombs — and it
guards against self-play rock-paper-scissors by keeping older referents around.
"""

import copy
import random

from tichu_training.ppo.policy import BatchedPolicy


class League:
    """A pool of opponent policies: fixed `base` opponents (e.g. frozen BC) plus
    up to `max_snapshots` frozen learner snapshots (oldest dropped — the league
    stays small). `sample` draws one opponent uniformly for an iteration."""

    def __init__(self, base, *, max_snapshots: int = 5, rng: random.Random | None = None) -> None:
        self._base = list(base)
        self._snapshots: list = []
        self._max_snapshots = int(max_snapshots)
        self._rng = rng or random.Random(0)

    def snapshot(self, model, critic, *, skill_decile: int = 9) -> None:
        """Freeze a deep copy of the current learner into the league."""
        frozen_model = copy.deepcopy(model).eval()
        frozen_critic = copy.deepcopy(critic).eval()
        for param in frozen_model.parameters():
            param.requires_grad_(False)
        for param in frozen_critic.parameters():
            param.requires_grad_(False)
        self._snapshots.append(
            BatchedPolicy(frozen_model, frozen_critic, skill_decile=skill_decile)
        )
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)  # drop the oldest snapshot — keep the league small

    def sample(self, iteration: int | None = None):
        return self._rng.choice(self._base + self._snapshots)

    def __len__(self) -> int:
        return len(self._base) + len(self._snapshots)
