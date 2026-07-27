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

    `observe_only` names a subset of `opponents` that are scored and reported in
    every verdict (so their margin lands in `promotion_gate.csv` and the plot) but
    are excluded from BOTH `ready()` and the promote decision — fixed visibility
    references (e.g. the shipped champion) that must never hard-block the ratchet.

    `pooled` is the **Pooled Verdict** mode (ADR-0040): margins accumulate across
    consecutive windows against the unchanged champion (fresh deals per window make
    pooling valid) and `conclude()` clears the pool only on a promotion — so a true
    +1-2 edge, invisible to any single window's CI, banks once the pooled CI
    resolves. Off (default) = the original one-window-per-verdict ratchet.

    `paired` is the seat-swap CLUSTER bootstrap. The GREEDY margin source records
    `2 * n_deals` values per window — the two seat arrangements of each deal, adjacent
    — and resampling those flat double-counts card luck the swap already cancelled.
    Resampling per-deal pair means instead is the correct estimator for the design.
    Measured on the 40k ship check (rho = -0.42): the CI narrows ~26% at 4096 deals,
    i.e. ~46% fewer deals for the same resolution, at an identical point estimate.
    Only valid for the greedy stream — the SAMPLED stream is one margin per game with
    no pairing, so this is off by default and the caller must opt in.

    NB the residual after pairing is dominated by CALL-bonus variance (sd 86 of 118
    on the ship check): each agent decides its own Tichu/Grand, so the +/-100/200
    does NOT cancel under seat-swap. Pairing alone moves the promote bar ~4.8 -> ~3.5;
    reaching the ~1.6 band needs `pooled` as well.
    """

    def __init__(self, *, opponents=("champion",), window_games: int,
                 threshold: float = 0.0, bootstrap_iters: int = 1000, seed: int = 0,
                 observe_only=(), pooled: bool = False, paired: bool = False) -> None:
        if window_games < 1:
            raise ValueError(f"window_games must be >= 1, got {window_games}")
        if not opponents:
            raise ValueError("need at least one opponent")
        self.opponents = tuple(opponents)
        self.observe_only = frozenset(observe_only)
        unknown = self.observe_only - set(self.opponents)
        if unknown:
            raise ValueError(
                f"observe_only names not in opponents: {sorted(unknown)}")
        self.required = tuple(o for o in self.opponents if o not in self.observe_only)
        if not self.required:
            raise ValueError(
                "every opponent is observe_only — need at least one required opponent")
        self.window_games = int(window_games)
        self.threshold = float(threshold)
        self.bootstrap_iters = int(bootstrap_iters)
        self.pooled = bool(pooled)
        self.paired = bool(paired)
        self._rng = np.random.default_rng(seed)
        self._margins: dict[str, list[float]] = {o: [] for o in self.opponents}
        # New-data latch: one verdict per batch of recorded margins. Historically
        # reset() enforced this implicitly (empty pool => not ready); a pooled HOLD
        # keeps the pool, so without the latch a caller polling ready() every
        # iteration would re-draw and re-log the same verdict until the next window.
        self._dirty = False

    def record(self, opponent: str, margins) -> None:
        """Add this iteration's per-game learner margins (one value per game) for the
        opponent it played this iteration."""
        if opponent not in self._margins:
            raise KeyError(f"unknown opponent {opponent!r}; expected one of {self.opponents}")
        self._margins[opponent].extend(float(m) for m in margins)
        self._dirty = True

    def n(self, opponent: str) -> int:
        return len(self._margins[opponent])

    def ready(self) -> bool:
        """True once every REQUIRED opponent has a full window AND new margins
        arrived since the last concluded verdict (observe-only streams never gate
        readiness; the new-data latch stops a pooled hold from re-drawing)."""
        return self._dirty and all(
            len(self._margins[o]) >= self.window_games for o in self.required)

    def _ci(self, x: list[float]) -> tuple[float, float, float]:
        arr = np.asarray(x, dtype=float)
        mean = float(arr.mean()) if arr.size else 0.0
        if self.paired:
            # Cluster the seat-swap pairs before resampling. Each greedy window
            # appends an even, pair-adjacent block, so concatenated windows (pooled
            # mode) stay correctly aligned. The mean is taken BEFORE clustering so
            # it is reported identically either way (pair means average to the same
            # value only when every pair is complete — which `pair_cluster` enforces).
            from tichu_eval.tournament import pair_cluster

            arr = pair_cluster(arr)
        n = arr.size
        if n == 0:
            return 0.0, 0.0, 0.0
        idx = self._rng.integers(0, n, size=(self.bootstrap_iters, n))
        boots = arr[idx].mean(axis=1)
        lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
        return mean, lo, hi

    def verdict(self) -> dict:
        """`{promote, opponents: {name: {n, mean, ci_lo, ci_hi}}}`. `promote` requires
        a full window for every REQUIRED opponent AND each of their CI lower bounds
        above `threshold`; observe-only streams are reported but never consulted."""
        opp = {}
        for o in self.opponents:
            mean, lo, hi = self._ci(self._margins[o])
            opp[o] = {"n": len(self._margins[o]), "mean": mean, "ci_lo": lo, "ci_hi": hi}
        promote = self.ready() and all(opp[o]["ci_lo"] > self.threshold for o in self.required)
        return {"promote": promote, "opponents": opp}

    def conclude(self, verdict: dict) -> None:
        """Close out a drawn verdict: clear the accumulated margins, EXCEPT in
        pooled mode after a hold — there the pool keeps accumulating until a
        promotion resolves it (the Pooled Verdict ratchet). Either way the
        new-data latch drops, so no second verdict is drawn from the same data."""
        self._dirty = False
        if self.pooled and not verdict["promote"]:
            return
        self.reset()

    def reset(self) -> None:
        """Drop the accumulated windows for all opponents (call after each verdict)."""
        self._margins = {o: [] for o in self.opponents}
        self._dirty = False

    def state(self) -> dict:
        """Serialisable pool state for the Resume Bundle (mirrors CoTrainLeague).

        Load-bearing ONLY under `pooled`: an unpooled gate resets every window, so a
        restart loses at most one partial window. A pooled gate accumulates across
        5-20 windows (640-2560 iterations at greedy_every=128) — without this, a
        Ctrl-C, crash or reboot mid-accumulation silently restarts the count from
        zero and the small edge pooling exists to bank never banks.

        Stored float64 so the round-trip is EXACT and a resumed verdict is bit-for-bit
        the one the uninterrupted run would have drawn. float32 would halve it, but the
        whole pool is ~4 MB at its deepest (20 windows x 8192 x 3 opponents) against a
        252 MB bundle — not worth trading exactness for 1.5%.
        """
        return {
            "margins": {o: np.asarray(v, dtype=np.float64) for o, v in self._margins.items()},
            "dirty": bool(self._dirty),
            "rng": self._rng.bit_generator.state,
        }

    def load_state(self, state: dict) -> None:
        """Restore a pool written by `state()`.

        Tolerant of an opponent-set change across the restart (a config edit adding or
        dropping an `extra_opponents` entry): streams absent from the payload start
        empty — `ready()` then waits for them, which is the conservative direction —
        and saved streams no longer configured are discarded. Restoring the bootstrap
        RNG keeps a resumed verdict identical to the uninterrupted one.
        """
        saved = state.get("margins") or {}
        self._margins = {o: [float(m) for m in saved.get(o, [])] for o in self.opponents}
        self._dirty = bool(state.get("dirty", False))
        if state.get("rng") is not None:
            self._rng.bit_generator.state = state["rng"]
