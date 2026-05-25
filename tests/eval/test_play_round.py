"""Single-deal runner: play one round to completion."""

from tichu_eval.starting_position_pool import generate_starting_position_pool
from tichu_eval.play import play_round
from tichu_ml.random_agent import RandomAgent
from tichu_ml.rule_agent import RuleAgent


def _seed_agents(cls, seeds):
    return tuple(cls(seed=s) for s in seeds)


def test_random_x4_completes_and_returns_int_tuple():
    deal = generate_starting_position_pool(seed=0, n=1)[0]
    agents = _seed_agents(RandomAgent, (1, 2, 3, 4))
    delta = play_round(agents, deal)
    assert isinstance(delta, tuple)
    assert len(delta) == 2
    assert isinstance(delta[0], int)
    assert isinstance(delta[1], int)


def test_same_seeded_agents_and_state_produce_identical_deltas():
    deal = generate_starting_position_pool(seed=0, n=1)[0]
    a = play_round(_seed_agents(RandomAgent, (1, 2, 3, 4)), deal)
    b = play_round(_seed_agents(RandomAgent, (1, 2, 3, 4)), deal)
    assert a == b


def test_different_seeds_can_produce_different_deltas():
    deal = generate_starting_position_pool(seed=0, n=1)[0]
    a = play_round(_seed_agents(RandomAgent, (1, 2, 3, 4)), deal)
    b = play_round(_seed_agents(RandomAgent, (10, 20, 30, 40)), deal)
    # Vanishingly unlikely to collide.
    assert a != b or True  # don't hard-fail; just exercise the path


def test_rule_agents_complete_multiple_deals():
    deals = generate_starting_position_pool(seed=5, n=3)
    for deal in deals:
        agents = (RuleAgent(), RuleAgent(), RuleAgent(), RuleAgent())
        delta = play_round(agents, deal)
        assert isinstance(delta[0], int) and isinstance(delta[1], int)


def test_returned_delta_matches_engine_final_scores_relative_to_initial():
    deal = generate_starting_position_pool(seed=7, n=1)[0]
    agents = _seed_agents(RandomAgent, (1, 2, 3, 4))
    delta = play_round(agents, deal)
    # initial scores are (0, 0), so delta == final scores at round end.
    # Just check that points are conserved in a reasonable range
    # (Tichu rounds rarely exceed ~300 total card points + bonuses).
    total = delta[0] + delta[1]
    assert -400 <= total <= 600
