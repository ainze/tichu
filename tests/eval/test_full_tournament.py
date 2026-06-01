"""Full-strength tournament: all-vs-all over a Full-strength Pool, reporting the
total score-delta matrix plus the Call-bonus breakdown. See ADR-0025.
"""

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.tournament import MatrixResult, run_full_tournament
from tichu_ml.random_agent import RandomAgent
from tichu_ml.rule_agent import RuleAgent


def _agents(seed: int = 0):
    return {"random": RandomAgent(seed=seed), "rule": RuleAgent()}


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
    agents = {"caller": GrandCaller(), "rule": RuleAgent()}
    result = run_full_tournament(agents, pool, bootstrap_iters=50, seed=0)
    # The caller calls Grand-Tichu every Round; the rule agent never calls.
    assert result.call_bonus_mean("caller", "rule") != 0.0
