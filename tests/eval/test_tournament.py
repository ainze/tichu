"""All-vs-all tournament: paired-difference matrix with bootstrap CIs.

The cornerstone smoke check is that `RuleAgent` beats `RandomAgent`
by a positive average margin — this catches a broken orchestrator,
broken legality enforcement, and broken score accounting in one test.
"""

from tichu_eval.starting_position_pool import generate_starting_position_pool
from tichu_eval.tournament import MatrixResult, run_tournament
from tichu_ml.random_agent import RandomAgent
from tichu_ml.rule_agent import RuleAgent


def _make_agents(seed: int = 0):
    return {
        "random": RandomAgent(seed=seed),
        "rule": RuleAgent(),
    }


def test_rule_beats_random_on_100_deals():
    deals = generate_starting_position_pool(seed=0, n=100)
    result = run_tournament(_make_agents(), deals, bootstrap_iters=200, seed=0)
    assert isinstance(result, MatrixResult)
    rule_vs_random = result.mean("rule", "random")
    assert rule_vs_random > 0, f"expected positive margin, got {rule_vs_random}"


def test_paired_observation_count_doubles_with_seat_swap():
    deals = generate_starting_position_pool(seed=0, n=4)
    result = run_tournament(_make_agents(), deals, bootstrap_iters=50, seed=0)
    n = result.n("rule", "random")
    assert n == 2 * len(deals)


def test_diagonal_is_zero_with_zero_width_ci():
    deals = generate_starting_position_pool(seed=0, n=4)
    result = run_tournament(_make_agents(), deals, bootstrap_iters=50, seed=0)
    assert result.mean("rule", "rule") == 0.0
    lo, hi = result.ci("rule", "rule")
    assert lo == 0.0 and hi == 0.0


def test_mean_is_antisymmetric():
    deals = generate_starting_position_pool(seed=0, n=4)
    result = run_tournament(_make_agents(), deals, bootstrap_iters=50, seed=0)
    assert result.mean("rule", "random") == -result.mean("random", "rule")


def test_bootstrap_ci_is_deterministic_for_same_seed():
    deals = generate_starting_position_pool(seed=0, n=8)
    a = run_tournament(_make_agents(), deals, bootstrap_iters=100, seed=42)
    b = run_tournament(_make_agents(), deals, bootstrap_iters=100, seed=42)
    assert a.ci("rule", "random") == b.ci("rule", "random")


def test_rows_have_expected_columns():
    deals = generate_starting_position_pool(seed=0, n=4)
    result = run_tournament(_make_agents(), deals, bootstrap_iters=20, seed=0)
    assert result.rows, "expected at least one row"
    expected = {"agent_a", "agent_b", "mean", "ci_lower", "ci_upper", "n",
                "win_rate", "tie_rate"}
    assert expected <= set(result.rows[0].keys())


def test_rule_out_wins_random_per_round():
    deals = generate_starting_position_pool(seed=0, n=100)
    result = run_tournament(_make_agents(), deals, bootstrap_iters=200, seed=0)
    # The stronger agent should win the majority of Rounds, not just on average margin.
    assert result.win_rate("rule", "random") > result.win_rate("random", "rule")
    assert result.win_rate("rule", "random") > 0.5


def test_win_tie_loss_rates_sum_to_one():
    deals = generate_starting_position_pool(seed=0, n=20)
    result = run_tournament(_make_agents(), deals, bootstrap_iters=50, seed=0)
    total = (result.win_rate("rule", "random")
             + result.tie_rate("rule", "random")
             + result.win_rate("random", "rule"))
    assert abs(total - 1.0) < 1e-9


def test_tie_rate_is_symmetric_and_diagonal_is_self_tie():
    deals = generate_starting_position_pool(seed=0, n=8)
    result = run_tournament(_make_agents(), deals, bootstrap_iters=50, seed=0)
    assert result.tie_rate("rule", "random") == result.tie_rate("random", "rule")
    # An agent vs itself ties every Round.
    assert result.tie_rate("rule", "rule") == 1.0
    assert result.win_rate("rule", "rule") == 0.0
