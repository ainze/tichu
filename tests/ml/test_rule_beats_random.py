"""RuleAgent must beat RandomAgent by a positive average margin.

This is the headline behavioural test from issue #003: RuleAgent's heuristics
should be strong enough — even though they are deliberately minimal — that the
team of RuleAgents ends up ahead of the team of RandomAgents on average over a
large number of independent deals.

We run team-vs-team: seats 0 and 2 are RuleAgent, seats 1 and 3 are RandomAgent.
The team0 - team1 final-score difference is summed across deals.
"""

import pytest

from tichu_engine.engine import step
from tichu_engine.state import deal_initial_state
from tichu_ml.random_agent import RandomAgent
from tichu_ml.rule_agent import RuleAgent


def _play_one(seed: int) -> int:
    state = deal_initial_state(seed=seed)
    agents = [
        RuleAgent(),
        RandomAgent(seed=seed * 7 + 1),
        RuleAgent(),
        RandomAgent(seed=seed * 7 + 3),
    ]
    for _ in range(2000):
        actor = state.public.current_player
        private = state.private_view(actor)
        action = agents[actor].act(private)
        state, _, done, _ = step(state, action)
        if done:
            break
    return state.public.scores[0] - state.public.scores[1]


@pytest.mark.slow
def test_rule_agent_beats_random_over_many_deals():
    total = 0
    n = 1000
    for seed in range(n):
        total += _play_one(seed=seed)
    average = total / n
    assert average > 0, f"RuleAgent did not beat RandomAgent on average: avg margin = {average}"
