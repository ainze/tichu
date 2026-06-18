"""Champion promotion gate for co-training — the rounds-as-gate ratchet.

Instead of bolting a separate tournament into the loop, the rollout opponent(s) are
frozen, so each iteration's learner `round_outcome`s ARE the candidate-vs-opponent
margin (fresh deals each iter => an unbiased estimate). This gate accumulates those
per-game margins, per opponent, over a window and PROMOTES the candidate to champion
when EVERY opponent's bootstrap CI lower bound clears a threshold — a variance-robust
ratchet that pierces the per-update noise floor using games already being played. It
is the principled sibling of the ungated `reanchor_play_every` trail (ADR-0035): the
reference only advances on a VALIDATED improvement.

Multi-opponent (the anti-cycling lever): a single champion can be *cycled* in a
non-transitive game (beat-your-ancestor != get-stronger). Gating against a POOL —
e.g. {champion (moving), BC (fixed)} — means a promotion must beat all of them, so a
champion-beating *exploit* that doesn't also beat BC is rejected. With BC as a fixed
pool member the policy can DIVERGE from BC in policy-space (the KL anchor follows the
champion) yet must still BEAT BC in outcome-space — "play differently, but better."

Scope caveats (documented, not solved here): the margin is read off the training
stream (unbiased on fresh deals vs frozen opponents, mildly optimistic); the rigorous
ship check stays the paired tournament + move-prediction drift flag.
"""

import numpy as np


class PromotionGate:
    """Accumulate per-game candidate-vs-opponent margins for one or more opponents;
    promote on a CI-validated improvement against ALL of them.

    `opponents` names the streams (default a single `"champion"`). `window_games` is
    the minimum games PER OPPONENT before a verdict is drawn. `promote` is True iff
    every opponent has a full window AND its 95% bootstrap CI lower bound exceeds
    `threshold` (default 0 = any positive). Non-overlapping windows: `reset()` after
    each verdict.
    """

    def __init__(self, *, opponents=("champion",), window_games: int,
                 threshold: float = 0.0, bootstrap_iters: int = 1000, seed: int = 0) -> None:
        if window_games < 1:
            raise ValueError(f"window_games must be >= 1, got {window_games}")
        if not opponents:
            raise ValueError("need at least one opponent")
        self.opponents = tuple(opponents)
        self.window_games = int(window_games)
        self.threshold = float(threshold)
        self.bootstrap_iters = int(bootstrap_iters)
        self._rng = np.random.default_rng(seed)
        self._margins: dict[str, list[float]] = {o: [] for o in self.opponents}

    def record(self, opponent: str, margins) -> None:
        """Add this iteration's per-game learner margins (one value per game) for the
        opponent it played this iteration."""
        if opponent not in self._margins:
            raise KeyError(f"unknown opponent {opponent!r}; expected one of {self.opponents}")
        self._margins[opponent].extend(float(m) for m in margins)

    def n(self, opponent: str) -> int:
        return len(self._margins[opponent])

    def ready(self) -> bool:
        """True once EVERY opponent has accumulated a full window."""
        return all(len(self._margins[o]) >= self.window_games for o in self.opponents)

    def _ci(self, x: list[float]) -> tuple[float, float, float]:
        arr = np.asarray(x, dtype=float)
        n = arr.size
        if n == 0:
            return 0.0, 0.0, 0.0
        idx = self._rng.integers(0, n, size=(self.bootstrap_iters, n))
        boots = arr[idx].mean(axis=1)
        lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
        return float(arr.mean()), lo, hi

    def verdict(self) -> dict:
        """`{promote, opponents: {name: {n, mean, ci_lo, ci_hi}}}`. `promote` requires
        a full window for EVERY opponent AND every CI lower bound above `threshold`."""
        opp = {}
        for o in self.opponents:
            mean, lo, hi = self._ci(self._margins[o])
            opp[o] = {"n": len(self._margins[o]), "mean": mean, "ci_lo": lo, "ci_hi": hi}
        promote = self.ready() and all(opp[o]["ci_lo"] > self.threshold for o in self.opponents)
        return {"promote": promote, "opponents": opp}

    def reset(self) -> None:
        """Drop the accumulated windows for all opponents (call after each verdict)."""
        self._margins = {o: [] for o in self.opponents}
