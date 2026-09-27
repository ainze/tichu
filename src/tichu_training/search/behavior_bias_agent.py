"""`ml_biased` — an MLAgent whose play logits carry a Behavior Bias.

Identical to `ml` except that, in the lever's situation, δ is added to the
target actions' play logits before the argmax (`behavior_bias`). Every other
Decision — Schupfen, Calls, Wish, Dragon — is the unbiased agent's. δ = 0 is
the unbiased agent exactly. See the Behavior Sensitivity Probe pre-registration
(docs/notes/2026-09-27-behavior-sensitivity-preregistration.md).
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from tichu_engine.legality import legal_actions_for
from tichu_inference.ml_agent import MLAgent
from tichu_ml.registry import register_agent
from tichu_training.search.behavior_bias import LEVERS, bias_play_logits, lever_gap


@register_agent("ml_biased")
class BiasedMLAgent(MLAgent):
    def __init__(self, checkpoint_path, *, lever: str, delta: float, **kwargs) -> None:
        self._lever = LEVERS[lever]
        self._delta = float(delta)
        super().__init__(checkpoint_path, **kwargs)

    def _play_logits(self, private_state) -> np.ndarray:
        logits = super()._play_logits(private_state)
        return bias_play_logits(private_state, list(legal_actions_for(private_state)),
                                logits, self._lever, self._delta)


@register_agent("ml_gaps")
class GapRecordingMLAgent(MLAgent):
    """The unbiased agent, recording every lever's `lever_gap` at each of its
    non-forced Play Decisions — the δ-calibration pass. Plays exactly like `ml`
    (the gap reads the same memoised forward the argmax then uses)."""

    def __init__(self, checkpoint_path, **kwargs) -> None:
        super().__init__(checkpoint_path, **kwargs)
        self.gaps: dict[str, list[float]] = {name: [] for name in LEVERS}

    def _act_on_play(self, private_state):
        legal = list(legal_actions_for(private_state))
        if len(legal) > 1:
            logits = self._play_logits(private_state)
            if np.isfinite(logits).all():
                for name, lever in LEVERS.items():
                    gap = lever_gap(private_state, legal, logits, lever)
                    if gap is not None:
                        self.gaps[name].append(gap)
        return super()._act_on_play(private_state)


def collect_lever_gaps(recorder_builder, opponent_builder, positions, *,
                       workers: int = 1) -> dict[str, list[float]]:
    """Every lever's gaps over `positions`: the recorder's team vs the opponent's,
    both Seat-Swap halves, in Position order. Builders are zero-arg module-level
    callables (spawned workers rebuild them), as in `run_drift_arms`."""
    if workers <= 1:
        return _gap_chunk(recorder_builder(), opponent_builder(), positions)
    from tichu_eval.tournament import _chunk_bounds

    bounds = _chunk_bounds(len(positions), workers, max_chunk=64)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_gap_worker_init,
                  initargs=(recorder_builder, opponent_builder, positions)) as pool:
        parts = [pool.apply_async(_gap_worker_chunk, b) for b in bounds]
        out: dict[str, list[float]] = {name: [] for name in LEVERS}
        for ar in parts:                      # submission order == Position order
            for name, gaps in ar.get().items():
                out[name].extend(gaps)
    return out


def _gap_chunk(recorder, opponent, positions) -> dict[str, list[float]]:
    from tichu_eval.play_full import play_full_round

    recorder.gaps = {name: [] for name in LEVERS}
    for pos in positions:
        for seats in ((0, 2), (1, 3)):
            agents = tuple(recorder if s in seats else opponent for s in range(4))
            play_full_round(agents, pos.state, pos.grand_prefixes)
    return recorder.gaps


_GAP_WORKER: dict = {}


def _gap_worker_init(recorder_builder, opponent_builder, positions) -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover
        pass
    _GAP_WORKER.update(recorder=recorder_builder(), opponent=opponent_builder(),
                       positions=positions)


def _gap_worker_chunk(start: int, end: int) -> dict[str, list[float]]:
    return _gap_chunk(_GAP_WORKER["recorder"], _GAP_WORKER["opponent"],
                      _GAP_WORKER["positions"][start:end])
