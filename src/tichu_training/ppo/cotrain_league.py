"""File-based opponent league for full-stack co-training (ADR-0034, league lever).

Pure self-play optimizes "beat my current self" — which can plateau or cycle
(rock-paper-scissors) without becoming robustly stronger than `master`. A league
pins a frozen-BC (master-level) opponent in the training distribution and adds
periodic frozen snapshots of the improving learner, so the learner is always under
pressure to beat master-level play and the diversity damps cycling (ADR-0029's
non-transitivity guard).

Co-training opponents run inside spawned rollout workers, so — unlike the play-only
in-memory `League` — this league holds weight-FILE paths: a fixed `base`
(frozen-BC bundle, never evicted) plus up to `max_snapshots` learner snapshot files
(oldest dropped). It round-trips through the Resume Bundle via `state` / `from_state`.
Opt-in and parallel-only; `rollout_workers <= 1` keeps pure self-play.
"""

import os
import random
from pathlib import Path

from tichu_training.ppo.rollout_parallel import save_rollout_weights


class CoTrainLeague:
    def __init__(self, league_dir, base_path: str, *, max_snapshots: int = 3,
                 rng: random.Random | None = None) -> None:
        self._dir = Path(league_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._base = [str(base_path)]           # frozen-BC; never evicted
        self._snapshots: list[str] = []         # frozen learner snapshot files
        self._max = int(max_snapshots)
        self._rng = rng or random.Random(0)

    def snapshot(self, models: dict, critic, iteration: int) -> None:
        """Freeze the current learner weights into the league as a new file (the file
        IS the frozen copy — no in-memory deepcopy needed), evicting the oldest
        learner snapshot beyond `max_snapshots`."""
        path = str(self._dir / f"snap_iter_{iteration:05d}.pt")
        save_rollout_weights(path, models, critic)
        self._snapshots.append(path)
        while len(self._snapshots) > self._max:
            old = self._snapshots.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass

    def sample(self) -> str:
        """One opponent weight-file path for an iteration (uniform over base + snapshots)."""
        return self._rng.choice(self._base + self._snapshots)

    def members(self) -> list[str]:
        return self._base + self._snapshots

    def state(self) -> dict:
        return {"base": list(self._base), "snapshots": list(self._snapshots), "max": self._max}

    @classmethod
    def from_state(cls, state: dict, *, league_dir, rng: random.Random | None = None) -> "CoTrainLeague":
        league = cls(league_dir, state["base"][0], max_snapshots=int(state["max"]), rng=rng)
        league._base = list(state["base"])
        league._snapshots = list(state["snapshots"])
        return league
