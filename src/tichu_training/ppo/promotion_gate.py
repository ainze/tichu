"""Champion promotion gate for co-training — the rounds-as-gate ratchet.

Instead of bolting a separate tournament into the loop, the rollout opponent is the
frozen CHAMPION, so each iteration's learner `round_outcome`s ARE the candidate-vs-
champion margin (fresh deals each iter => an unbiased estimate). This gate
accumulates those per-game margins over a window and PROMOTES the candidate to
champion when the windowed margin's bootstrap CI lower bound clears a threshold — a
variance-robust ratchet that pierces the per-update noise floor using games you're
already playing. It is the principled sibling of the ungated `reanchor_play_every`
trail (ADR-0035): the reference only advances on a VALIDATED improvement.

Scope caveats (documented, not solved here):
  * Mild optimism — the margin is read off the training stream; the policy is
    selected to be good, though on fresh deals vs a frozen champion it's unbiased.
    The rigorous ship check stays the paired tournament + move-prediction drift flag.
  * A single champion can be cycled in a non-transitive game; mixing past champions
    (a league of frozen winners) is the robustness upgrade, not done here.
"""

import numpy as np


class PromotionGate:
    """Accumulate per-game candidate-vs-champion margins; promote on a CI-validated
    improvement.

    `window_games` is the minimum number of games to accumulate before a verdict is
    drawn (more games => tighter CI; ~M*K for K iterations at M positions). `promote`
    is True iff the bootstrap CI lower bound of the mean margin exceeds `threshold`
    (default 0 = any positive). Non-overlapping windows: `reset()` after each verdict.
    """

    def __init__(self, *, window_games: int, threshold: float = 0.0,
                 bootstrap_iters: int = 1000, seed: int = 0) -> None:
        if window_games < 1:
            raise ValueError(f"window_games must be >= 1, got {window_games}")
        self.window_games = int(window_games)
        self.threshold = float(threshold)
        self.bootstrap_iters = int(bootstrap_iters)
        self._rng = np.random.default_rng(seed)
        self._margins: list[float] = []

    def record(self, game_margins) -> None:
        """Add this iteration's per-game learner margins (one value per game)."""
        self._margins.extend(float(m) for m in game_margins)

    @property
    def n(self) -> int:
        return len(self._margins)

    def ready(self) -> bool:
        """True once enough games have accumulated to draw a verdict."""
        return self.n >= self.window_games

    def verdict(self) -> dict:
        """`{n, mean, ci_lo, ci_hi, promote}` over the accumulated margins. `promote`
        requires both a full window AND the 95% CI lower bound above `threshold`."""
        x = np.asarray(self._margins, dtype=float)
        n = int(x.size)
        if n == 0:
            return {"n": 0, "mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "promote": False}
        idx = self._rng.integers(0, n, size=(self.bootstrap_iters, n))
        boots = x[idx].mean(axis=1)
        lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
        return {
            "n": n, "mean": float(x.mean()), "ci_lo": lo, "ci_hi": hi,
            "promote": self.ready() and lo > self.threshold,
        }

    def reset(self) -> None:
        """Drop the accumulated window (call after each verdict)."""
        self._margins = []
