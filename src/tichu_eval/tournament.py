"""All-vs-all tournament orchestrator.

For every unordered pair of named agents, plays each deal in the pool twice
with seat-swap (A in team-0 seats, then B in team-0 seats) to cancel the
team-assignment advantage of the Mahjong holder being at a fixed seat.

The output is a paired-difference matrix: mean A-minus-B score delta per pair,
with a percentile bootstrap 95% CI computed from the per-deal deltas.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np

from tichu_engine.state import GameState
from tichu_eval.play import play_round
from tichu_ml.agent import Agent


@dataclass
class MatrixResult:
    rows: list[dict]
    _mean: dict[tuple[str, str], float] = field(default_factory=dict)
    _ci: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    _n: dict[tuple[str, str], int] = field(default_factory=dict)

    def mean(self, a: str, b: str) -> float:
        return self._mean[(a, b)]

    def ci(self, a: str, b: str) -> tuple[float, float]:
        return self._ci[(a, b)]

    def n(self, a: str, b: str) -> int:
        return self._n[(a, b)]


def run_tournament(
    agents: dict[str, Agent],
    deals: list[GameState],
    *,
    bootstrap_iters: int = 1000,
    seed: int = 0,
) -> MatrixResult:
    if len(agents) < 2:
        raise ValueError(f"need at least 2 agents, got {len(agents)}")
    rng = np.random.default_rng(seed)
    names = sorted(agents.keys())
    result = MatrixResult(rows=[])

    for a, b in itertools.combinations(names, 2):
        deltas = _play_pair(agents[a], agents[b], deals)
        mean_ab, lo, hi = _bootstrap_ci(deltas, bootstrap_iters, rng)
        n_obs = len(deltas)
        result._mean[(a, b)] = mean_ab
        result._mean[(b, a)] = -mean_ab
        result._ci[(a, b)] = (lo, hi)
        result._ci[(b, a)] = (-hi, -lo)
        result._n[(a, b)] = n_obs
        result._n[(b, a)] = n_obs
        result.rows.append({
            "agent_a": a, "agent_b": b,
            "mean": mean_ab, "ci_lower": lo, "ci_upper": hi, "n": n_obs,
        })
        result.rows.append({
            "agent_a": b, "agent_b": a,
            "mean": -mean_ab, "ci_lower": -hi, "ci_upper": -lo, "n": n_obs,
        })

    for name in names:
        result._mean[(name, name)] = 0.0
        result._ci[(name, name)] = (0.0, 0.0)
        result._n[(name, name)] = 0
        result.rows.append({
            "agent_a": name, "agent_b": name,
            "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0,
        })
    return result


def _play_pair(agent_a: Agent, agent_b: Agent, deals: list[GameState]) -> np.ndarray:
    """Play every deal twice with seat-swap. Returns 2N score deltas A-minus-B."""
    deltas: list[float] = []
    for deal in deals:
        # Arrangement 1: A on team 0 (seats 0, 2), B on team 1 (seats 1, 3).
        s0, s1 = play_round((agent_a, agent_b, agent_a, agent_b), deal)
        deltas.append(float(s0 - s1))
        # Arrangement 2: rotated — A on team 1, B on team 0.
        s0, s1 = play_round((agent_b, agent_a, agent_b, agent_a), deal)
        deltas.append(float(s1 - s0))
    return np.array(deltas, dtype=np.float64)


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
