"""All-vs-all tournament orchestrator.

For every unordered pair of named agents, plays each deal in the pool twice
with seat-swap (A in team-0 seats, then B in team-0 seats) to cancel the
team-assignment advantage of the Mahjong holder being at a fixed seat.

The output is a paired-difference matrix: mean A-minus-B score delta per pair,
with a percentile bootstrap 95% CI computed from the per-deal deltas.
"""

import itertools
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from tichu_engine.state import GameState
from tichu_eval.play import play_round
from tichu_eval.play_full import play_full_round
from tichu_ml.agent import Agent


log = logging.getLogger(__name__)

# Target Position count per parallel task when a progress bar is attached. Small
# enough that the bar advances many times per pair, large enough that per-chunk
# pickling/IPC stays negligible (each chunk returns only a short list of floats).
_PROGRESS_CHUNK = 64


@dataclass
class MatrixResult:
    rows: list[dict]
    _mean: dict[tuple[str, str], float] = field(default_factory=dict)
    _ci: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    _n: dict[tuple[str, str], int] = field(default_factory=dict)
    _call_bonus_mean: dict[tuple[str, str], float] = field(default_factory=dict)

    def mean(self, a: str, b: str) -> float:
        return self._mean[(a, b)]

    def ci(self, a: str, b: str) -> tuple[float, float]:
        return self._ci[(a, b)]

    def n(self, a: str, b: str) -> int:
        return self._n[(a, b)]

    def call_bonus_mean(self, a: str, b: str) -> float:
        """Mean A-minus-B Call-bonus delta per Round (Full-strength only; 0.0
        for the play-strength matrix, where calls are never made)."""
        return self._call_bonus_mean.get((a, b), 0.0)


def run_tournament(
    agents: dict[str, Agent],
    deals: list[GameState],
    *,
    bootstrap_iters: int = 1000,
    seed: int = 0,
) -> MatrixResult:
    """Play-strength all-vs-all matrix (no Schupfen, calls declined)."""
    def pair_fn(a: str, b: str):
        deltas = _play_pair(agents[a], agents[b], deals, label=f"{a} vs {b}")
        return deltas, None

    return _run_matrix(sorted(agents.keys()), pair_fn, bootstrap_iters, seed, len(deals))


def run_full_tournament(
    agent_builders: dict[str, Callable[[], Agent]],
    positions: list,
    *,
    bootstrap_iters: int = 1000,
    seed: int = 0,
    workers: int = 1,
    progress: Callable[[int], None] | None = None,
) -> MatrixResult:
    """Full-strength all-vs-all matrix: the complete stack (Grand-Tichu ->
    Schupfen -> Tichu -> Play), reporting the total delta plus the Call-bonus
    breakdown per pair. See ADR-0025.

    Agents are supplied as zero-argument **builder callables** (name -> builder)
    rather than live instances, so the same recipe can be rebuilt independently
    inside parallel workers (torch models do not pickle/fork cleanly). For the
    serial path the builders are simply invoked once up front.

    `workers` (default 1) sets the size of a process Pool that the Pool of
    Positions is chunked across. The play loop is embarrassingly parallel — every
    Position is independent — chunk results are stitched back in Position order,
    and the bootstrap is still drawn on the main process in pair order. So the
    *orchestration* is deterministic, and for **torch-free deterministic agents**
    (RuleAgent) a parallel run is bit-identical to the serial run for the same
    `seed`. Two caveats, both pre-existing and below the bootstrap CI:
      * **ML agents** are not bit-reproducible across the process boundary —
        identical inputs yield bit-different torch CPU logits in another process,
        occasionally flipping a near-tie greedy argmax. Parallel and serial then
        differ by torch float noise (~0.1% of the mean), not by anything the
        parallelization introduces.
      * **Stateful agents** (e.g. RandomAgent, which threads one RNG through the
        whole run) are reproducible per run-config but NOT stream-equal to serial
        under `workers > 1`, because each worker rebuilds its agents afresh.
    """
    names = sorted(agent_builders)

    if workers <= 1:
        agents = {name: agent_builders[name]() for name in names}

        def pair_fn(a: str, b: str):
            return _play_pair_full(
                agents[a], agents[b], positions, label=f"{a} vs {b}", progress=progress
            )

        return _run_matrix(names, pair_fn, bootstrap_iters, seed, len(positions))

    return _run_full_tournament_parallel(
        names, agent_builders, positions, bootstrap_iters, seed, workers, progress
    )


def _run_full_tournament_parallel(
    names, agent_builders, positions, bootstrap_iters, seed, workers, progress=None,
) -> MatrixResult:
    """Parallel full-strength matrix. Each worker rebuilds every agent once (from
    the builders) and caches the Pool of Positions; per-pair work is dispatched as
    a handful of contiguous Position-index chunks (NOT one task per Round — that
    would drown in pickling/IPC). Chunk results are stitched back in Position
    order so the delta arrays — and therefore the bootstrap — match the serial
    run exactly."""
    # Fail fast: build every agent once here so a broken builder (unknown
    # factory, missing checkpoint) raises a clear error on the main process
    # rather than from inside the Pool initializer — where an exception silently
    # deadlocks the run as multiprocessing endlessly respawns the dead worker.
    # The instances are discarded; workers rebuild their own (torch models do not
    # pickle/fork cleanly).
    for name in names:
        agent_builders[name]()

    # With a progress bar attached, cap chunk size so the bar advances steadily
    # through each pair instead of jumping when the few worker-sized chunks all
    # finish together at the end. Without a bar there is nothing to smooth, so the
    # coarse worker-sized chunking (less IPC) stands.
    bounds = _chunk_bounds(
        len(positions), workers,
        max_chunk=_PROGRESS_CHUNK if progress is not None else None,
    )
    ctx = mp.get_context("spawn")  # spawn on every platform; workers must rebuild.
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(agent_builders, positions),
    ) as pool:
        def pair_fn(a: str, b: str):
            log.debug("  [%s vs %s] dispatching %d chunks x2 (seat-swap)", a, b, len(bounds))
            # `callback` fires on the Pool's result-handler thread as each chunk
            # lands (completion order), advancing the bar by that chunk's Position
            # count. Result assembly below still reads `.get()` in submission order,
            # so the determinism guarantee is untouched — the bar is a side channel.
            pending = [
                pool.apply_async(
                    _worker_play_chunk, (a, b, start, end),
                    callback=(
                        None if progress is None
                        else (lambda _res, n=end - start: progress(n))
                    ),
                )
                for start, end in bounds
            ]
            totals: list[float] = []
            bonuses: list[float] = []
            for ar in pending:  # chunk order == Position order == serial order.
                chunk_totals, chunk_bonuses = ar.get()
                totals.extend(chunk_totals)
                bonuses.extend(chunk_bonuses)
            return (
                np.array(totals, dtype=np.float64),
                np.array(bonuses, dtype=np.float64),
            )

        return _run_matrix(names, pair_fn, bootstrap_iters, seed, len(positions))


def _chunk_bounds(
    n: int, workers: int, *, max_chunk: int | None = None
) -> list[tuple[int, int]]:
    """Split [0, n) into contiguous, near-equal, non-empty half-open ranges.
    Concatenating their results in order reproduces 0..n-1.

    Without `max_chunk` there are exactly min(workers, n) chunks (one per worker).
    With `max_chunk` set, the count is raised to whatever it takes to keep every
    chunk no larger than `max_chunk` — at least `workers` chunks (so all cores
    stay busy), at most `n` (no empty chunks). More, smaller chunks make the
    progress bar advance steadily through a pair rather than jumping at its end."""
    if n == 0:
        return []
    k = min(workers, n)
    if max_chunk is not None:
        k = min(n, max(k, -(-n // max_chunk)))  # ceil(n / max_chunk), >= workers
    base, extra = divmod(n, k)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


# --- Worker-process state (one set per spawned process) ----------------------
# Populated once by `_worker_init`; read by `_worker_play_chunk`. Module-level so
# both are importable under spawn, where the child re-imports this module.
_WORKER_AGENTS: dict[str, Agent] = {}
_WORKER_POSITIONS: list = []


def _worker_init(agent_builders: dict[str, Callable[[], Agent]], positions: list) -> None:
    global _WORKER_AGENTS, _WORKER_POSITIONS
    # Pin torch (if present) to one thread per worker so W processes don't each
    # spin up W intra-op threads and oversubscribe the cores. Guarded: the
    # baseline agents have no torch dependency.
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover - torch always present in the real eval
        pass
    _WORKER_AGENTS = {name: build() for name, build in agent_builders.items()}
    _WORKER_POSITIONS = positions


def _worker_play_chunk(
    a_name: str, b_name: str, start: int, end: int
) -> tuple[list[float], list[float]]:
    """Play one pair over the Position slice [start, end) using this worker's
    rebuilt agents and cached Pool. Returns (total deltas, call-bonus deltas) in
    Position order, two entries per Position (seat-swap)."""
    agent_a = _WORKER_AGENTS[a_name]
    agent_b = _WORKER_AGENTS[b_name]
    totals: list[float] = []
    bonuses: list[float] = []
    for pos in _WORKER_POSITIONS[start:end]:
        t0, b0, t1, b1 = _play_one_position(agent_a, agent_b, pos)
        totals.extend((t0, t1))
        bonuses.extend((b0, b1))
    return totals, bonuses


def _run_matrix(names, pair_fn, bootstrap_iters: int, seed: int, n_units: int) -> MatrixResult:
    if len(names) < 2:
        raise ValueError(f"need at least 2 agents, got {len(names)}")
    rng = np.random.default_rng(seed)
    result = MatrixResult(rows=[])

    pairs = list(itertools.combinations(names, 2))
    for pair_i, (a, b) in enumerate(pairs, start=1):
        log.info(
            "pair %d/%d: %s vs %s -- %d positions x2 (seat-swap)",
            pair_i, len(pairs), a, b, n_units,
        )
        deltas, cb_deltas = pair_fn(a, b)
        mean_ab, lo, hi = _bootstrap_ci(deltas, bootstrap_iters, rng)
        n_obs = len(deltas)
        cb_mean = float(cb_deltas.mean()) if cb_deltas is not None and len(cb_deltas) else 0.0
        log.info(
            "  done: %s vs %s mean delta=%+.1f (call-bonus %+.1f, n=%d)",
            a, b, mean_ab, cb_mean, n_obs,
        )
        result._mean[(a, b)] = mean_ab
        result._mean[(b, a)] = -mean_ab
        result._ci[(a, b)] = (lo, hi)
        result._ci[(b, a)] = (-hi, -lo)
        result._n[(a, b)] = n_obs
        result._n[(b, a)] = n_obs
        if cb_deltas is not None:
            result._call_bonus_mean[(a, b)] = cb_mean
            result._call_bonus_mean[(b, a)] = -cb_mean
        result.rows.append({
            "agent_a": a, "agent_b": b,
            "mean": mean_ab, "ci_lower": lo, "ci_upper": hi, "n": n_obs,
            "call_bonus_mean": cb_mean,
        })
        result.rows.append({
            "agent_a": b, "agent_b": a,
            "mean": -mean_ab, "ci_lower": -hi, "ci_upper": -lo, "n": n_obs,
            "call_bonus_mean": -cb_mean,
        })

    for name in names:
        result._mean[(name, name)] = 0.0
        result._ci[(name, name)] = (0.0, 0.0)
        result._n[(name, name)] = 0
        result._call_bonus_mean[(name, name)] = 0.0
        result.rows.append({
            "agent_a": name, "agent_b": name,
            "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0,
            "call_bonus_mean": 0.0,
        })
    return result


def _play_pair(
    agent_a: Agent, agent_b: Agent, deals: list[GameState], *, label: str = "",
) -> np.ndarray:
    """Play every deal twice with seat-swap. Returns 2N score deltas A-minus-B.

    Emits ~10 evenly-spaced INFO progress ticks across the deal loop so a long
    run is observable (tournament mode has no other per-deal output).
    """
    n = len(deals)
    tick = max(1, n // 10)
    deltas: list[float] = []
    for i, deal in enumerate(deals, start=1):
        # Arrangement 1: A on team 0 (seats 0, 2), B on team 1 (seats 1, 3).
        s0, s1 = play_round((agent_a, agent_b, agent_a, agent_b), deal)
        deltas.append(float(s0 - s1))
        # Arrangement 2: rotated — A on team 1, B on team 0.
        s0, s1 = play_round((agent_b, agent_a, agent_b, agent_a), deal)
        deltas.append(float(s1 - s0))
        if i % tick == 0 or i == n:
            log.info("  [%s] %d/%d deals", label, i, n)
    return np.array(deltas, dtype=np.float64)


def _play_pair_full(
    agent_a: Agent, agent_b: Agent, positions: list, *, label: str = "",
    progress: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-strength seat-swap over the Pool. Returns (total deltas, call-bonus
    deltas), both A-minus-B, with 2N entries each. `progress`, if given, is called
    with 1 per completed Position so a caller can drive a bar (see
    `run_full_tournament`)."""
    n = len(positions)
    tick = max(1, n // 10)
    totals: list[float] = []
    bonuses: list[float] = []
    for i, pos in enumerate(positions, start=1):
        t0, b0, t1, b1 = _play_one_position(agent_a, agent_b, pos)
        totals.extend((t0, t1))
        bonuses.extend((b0, b1))
        if progress is not None:
            progress(1)
        if i % tick == 0 or i == n:
            log.debug("  [%s] %d/%d positions", label, i, n)
    return np.array(totals, dtype=np.float64), np.array(bonuses, dtype=np.float64)


def _play_one_position(agent_a: Agent, agent_b: Agent, pos) -> tuple[float, float, float, float]:
    """One Full-strength Starting Position played twice with seat-swap.

    Returns ``(total_1, bonus_1, total_2, bonus_2)`` — the A-minus-B total and
    Call-bonus deltas for arrangement 1 (A on team 0) and arrangement 2 (A on
    team 1). Pure given deterministic agents: the result depends only on `pos`
    and the agents, never on any previously played position. That purity is what
    lets the Pool replay arbitrary position chunks and still reproduce the serial
    matrix bit-for-bit (see `run_full_tournament(..., workers=...)`)."""
    # Arrangement 1: A on team 0 (seats 0, 2), B on team 1 (seats 1, 3).
    r1 = play_full_round((agent_a, agent_b, agent_a, agent_b), pos.state, pos.grand_prefixes)
    # Arrangement 2: rotated — A on team 1, B on team 0.
    r2 = play_full_round((agent_b, agent_a, agent_b, agent_a), pos.state, pos.grand_prefixes)
    return (
        float(r1.total[0] - r1.total[1]),
        float(r1.call_bonus[0] - r1.call_bonus[1]),
        float(r2.total[1] - r2.total[0]),
        float(r2.call_bonus[1] - r2.call_bonus[0]),
    )


def _bootstrap_ci(
    deltas: np.ndarray, iters: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    if len(deltas) == 0:
        return (0.0, 0.0, 0.0)
    mean = float(deltas.mean())
    if iters <= 0:
        return (mean, mean, mean)
    idx = rng.integers(0, len(deltas), size=(iters, len(deltas)))
    boot_means = deltas[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return (mean, lo, hi)
