"""Behavioral Drift Benchmark — the Fixed-Opponent arms.

Two arms over the same Pool deals, both recorded by `drift_recorder`:

* **subject** — Subject+Subject vs BC+BC, every deal played twice with Seat-Swap
  (half 0: Subject in seats 0/2; half 1: Subject in seats 1/3).
* **bc** — BC+BC vs BC+BC. With deterministic (greedy) agents both Seat-Swap
  halves are the *same game*, so it is played once and read from both teams:
  half 0 measures team 0, half 1 measures team 1. That halves the BC arm's cost
  and keeps both arms row-for-row alike for the paired statistics.

Only the Subject team's seats (`is_subject`) are read by metrics — the opponents
are identical in both arms, so any Δ is the Subject's own behavior. See CONTEXT.md
§"Behavioral Drift Benchmark".
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import pandas as pd

from tichu_eval.drift_recorder import record_round
from tichu_eval.tournament import _chunk_bounds

_HALF_SEATS = {0: (0, 2), 1: (1, 3)}


@dataclass
class DriftLog:
    """The benchmark's raw recording: one row per Decision, one per Round, each
    tagged with its `arm` ("subject" / "bc"), `deal` and Seat-Swap `half`."""

    decisions: pd.DataFrame
    rounds: pd.DataFrame


def run_aa_null(bc_builder, positions_a, positions_b, *, workers: int = 1,
                progress=None) -> DriftLog:
    """The **A/A Null Run**: the BC arm alone, played on two *disjoint* deal sets
    and paired by deal index — `positions_a` as the "bc" arm, `positions_b` as the
    "subject" arm. With deterministic agents an A/A on shared deals is the same
    games twice (every Δ exactly 0, every CI [0, 0]) and tests nothing; on
    disjoint deals the pairing is meaningless but the bootstrap stays valid, so
    the run checks CI calibration, the BH false-positive rate and denominators.
    Passing it gates reading any real Subject. See CONTEXT.md §"A/A Null Run"."""
    if len(positions_a) != len(positions_b):
        raise ValueError("the A/A arms need equally many deals to pair by index")
    a = run_drift_arms(None, bc_builder, positions_a, workers=workers, progress=progress)
    b = run_drift_arms(None, bc_builder, positions_b, workers=workers, progress=progress)
    return DriftLog(
        decisions=pd.concat([a.decisions, b.decisions.assign(arm="subject")], ignore_index=True),
        rounds=pd.concat([a.rounds, b.rounds.assign(arm="subject")], ignore_index=True),
    )


def run_drift_arms(subject_builder, bc_builder, positions, *, workers: int = 1,
                   progress=None) -> DriftLog:
    """Play both arms over `positions` and return the combined log.

    Builders are zero-arg callables (the Tournament contract): each is built once
    per process and the instance reused across seats and Rounds — the Tournament
    does the same, so an Agent must be stateless between `act` calls, and loading
    an ML Checkpoint per Round would dominate the run. With `workers > 1` the Pool
    is split into contiguous chunks over spawned processes — each rebuilds its own
    agents (torch models do not pickle/fork cleanly), so builders must be
    importable by qualname (module-level `partial`s, never closures). Chunks are
    stitched back in Position order, so the log equals the serial run's.
    `progress(n)`, if given, is called as each chunk of n Positions finishes.
    `subject_builder=None` plays the BC arm alone (the A/A Null Run's building block)."""
    if workers <= 1:
        decisions, rounds = _play_chunk(_build(subject_builder), bc_builder(), positions, 0)
        if progress is not None:
            progress(len(positions))
        return _to_log(decisions, rounds)
    _build(subject_builder); bc_builder()   # fail fast on the main process, not in a worker
    bounds = _chunk_bounds(len(positions), workers, max_chunk=64)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_worker_init,
                  initargs=(subject_builder, bc_builder, positions)) as pool:
        pending = [
            pool.apply_async(_worker_chunk, (start, end),
                             callback=None if progress is None
                             else (lambda _r, n=end - start: progress(n)))
            for start, end in bounds
        ]
        decisions: list[dict] = []
        rounds: list[dict] = []
        for ar in pending:   # submission order == Position order
            d, r = ar.get()
            decisions.extend(d)
            rounds.extend(r)
    return _to_log(decisions, rounds)


def _build(builder):
    return None if builder is None else builder()


def _play_chunk(subject, bc, positions, first_deal: int):
    """Both arms over a contiguous slice of the Pool; `subject=None` skips the
    Subject arm."""
    decisions: list[dict] = []
    rounds: list[dict] = []
    for i, pos in enumerate(positions):
        deal = first_deal + i
        if subject is not None:
            for half, seats in _HALF_SEATS.items():
                agents = tuple(subject if s in seats else bc for s in range(4))
                log = record_round(agents, pos, deal=deal, half=half, subject_seats=seats)
                _extend(decisions, rounds, log, arm="subject")
        bc_log = record_round((bc,) * 4, pos, deal=deal, half=0, subject_seats=_HALF_SEATS[0])
        _extend(decisions, rounds, bc_log, arm="bc")
        _extend(decisions, rounds, _relabel(bc_log, half=1), arm="bc")
    return decisions, rounds


def _to_log(decisions, rounds) -> DriftLog:
    return DriftLog(decisions=pd.DataFrame(decisions), rounds=pd.DataFrame(rounds))


# --- Worker-process state (one set per spawned process) ----------------------
_WORKER: dict = {}


def _worker_init(subject_builder, bc_builder, positions) -> None:
    # One torch thread per worker, so W processes don't each spin up W intra-op
    # threads (the Tournament's setting). Guarded: baseline agents have no torch.
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover
        pass
    _WORKER.update(subject=_build(subject_builder), bc=bc_builder(), positions=positions)


def _worker_chunk(start: int, end: int):
    return _play_chunk(_WORKER["subject"], _WORKER["bc"], _WORKER["positions"][start:end], start)


def _relabel(log, *, half: int):
    """The same BC game read from the other team: flip `half` and `is_subject`."""
    seats = _HALF_SEATS[half]
    return type(log)(
        decisions=[{**r, "half": half, "is_subject": r["seat"] in seats} for r in log.decisions],
        round={**log.round, "half": half, "subject_team": seats[0] % 2},
    )


def _extend(decisions: list, rounds: list, log, *, arm: str) -> None:
    decisions.extend({**r, "arm": arm} for r in log.decisions)
    rounds.append({**log.round, "arm": arm})


# --- Persistence ---------------------------------------------------------------
# Tuple-valued Round columns (seat orders, per-seat counts) come back from Parquet
# as arrays; metrics use tuple semantics (`in`, `.index`), so they are restored.
_TUPLE_COLUMNS = ("out_order", "tricks_won", "grand_callers", "tichu_callers")


def save_drift_log(log: DriftLog, directory) -> None:
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    log.decisions.to_parquet(directory / "decisions.parquet", index=False)
    log.rounds.to_parquet(directory / "rounds.parquet", index=False)


def load_drift_log(directory) -> DriftLog:
    from pathlib import Path

    directory = Path(directory)
    rounds = pd.read_parquet(directory / "rounds.parquet")
    for col in _TUPLE_COLUMNS:
        if col in rounds:
            # None stays None (a replayed human Round has no Trick attribution).
            rounds[col] = rounds[col].map(lambda v: None if v is None else tuple(int(x) for x in v))
    return DriftLog(decisions=pd.read_parquet(directory / "decisions.parquet"), rounds=rounds)
