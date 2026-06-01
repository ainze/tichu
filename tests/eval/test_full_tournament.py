"""Full-strength tournament: all-vs-all over a Full-strength Pool, reporting the
total score-delta matrix plus the Call-bonus breakdown. See ADR-0025.
"""

from functools import partial

import pytest

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.tournament import MatrixResult, _chunk_bounds, run_full_tournament
from tichu_ml.random_agent import RandomAgent
from tichu_ml.rule_agent import RuleAgent


def _agents(seed: int = 0):
    """Builder callables (name -> zero-arg factory), as run_full_tournament now
    expects. `partial` pins RandomAgent's seed; RuleAgent is its own zero-arg
    builder."""
    return {"random": partial(RandomAgent, seed=seed), "rule": RuleAgent}


class GrandCaller(RuleAgent):
    """Plays like RuleAgent but always calls Grand-Tichu."""

    def should_call(self, private_state, kind: str) -> bool:
        return kind == "grand"


def test_full_tournament_rule_beats_random():
    pool = generate_full_position_pool(seed=0, n=40)
    result = run_full_tournament(_agents(), pool, bootstrap_iters=100, seed=0)
    assert isinstance(result, MatrixResult)
    assert result.mean("rule", "random") > 0


def test_full_tournament_seat_swap_doubles_n():
    pool = generate_full_position_pool(seed=0, n=4)
    result = run_full_tournament(_agents(), pool, bootstrap_iters=20, seed=0)
    assert result.n("rule", "random") == 2 * len(pool)


def test_call_bonus_is_zero_for_non_calling_baselines():
    pool = generate_full_position_pool(seed=0, n=10)
    result = run_full_tournament(_agents(), pool, bootstrap_iters=20, seed=0)
    assert result.call_bonus_mean("rule", "random") == 0.0


def test_call_bonus_breakdown_reflects_a_caller():
    pool = generate_full_position_pool(seed=1, n=20)
    agents = {"caller": GrandCaller, "rule": RuleAgent}
    result = run_full_tournament(agents, pool, bootstrap_iters=50, seed=0)
    # The caller calls Grand-Tichu every Round; the rule agent never calls.
    assert result.call_bonus_mean("caller", "rule") != 0.0


def test_parallel_matches_serial_for_deterministic_agents():
    """The determinism guarantee: a parallel run is bit-identical to the serial
    run for the same seed, provided the agents are deterministic per Position
    (RuleAgent and the greedy ML agents — the real eval). Stateful agents like
    RandomAgent thread one RNG through the whole run and are deliberately out of
    scope here (see the module docstring / ADR note)."""
    pool = generate_full_position_pool(seed=2, n=30)
    builders = {"caller": GrandCaller, "rule": RuleAgent}

    serial = run_full_tournament(builders, pool, bootstrap_iters=50, seed=0, workers=1)
    parallel = run_full_tournament(builders, pool, bootstrap_iters=50, seed=0, workers=4)

    assert parallel.rows == serial.rows


def test_parallel_fails_fast_on_broken_builder():
    """A builder that raises must surface its error on the main process, not
    deadlock the Pool from inside its initializer (which would hang forever as
    multiprocessing respawns the dead worker)."""
    pool = generate_full_position_pool(seed=0, n=8)

    def broken():
        raise ValueError("boom: cannot build this agent")

    builders = {"rule": RuleAgent, "broken": broken}
    with pytest.raises(ValueError, match="boom"):
        run_full_tournament(builders, pool, bootstrap_iters=10, seed=0, workers=2)


def test_chunk_bounds_partitions_positions_contiguously():
    """Position chunks must tile [0, n) with no gaps/overlaps and never produce
    more (or empty) chunks than there are Positions — that contiguous order is
    what keeps the stitched delta array equal to the serial run."""
    assert _chunk_bounds(0, 4) == []
    assert _chunk_bounds(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert _chunk_bounds(3, 8) == [(0, 1), (1, 2), (2, 3)]  # capped at n chunks
    for n, w in [(7, 3), (100, 7), (1, 5), (12, 12)]:
        bounds = _chunk_bounds(n, w)
        assert bounds[0][0] == 0 and bounds[-1][1] == n
        assert all(lo < hi for lo, hi in bounds)  # no empty chunk
        assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))
