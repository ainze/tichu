"""Tracer-bullet test for the Agent contract.

The RandomAgent is the simplest possible Agent — it uniformly samples from the
legal-action set. This test exercises the full Agent <-> engine loop: deal a
game, ask each player's RandomAgent to act, step the engine, and assert that
every action returned is legal at the moment it is returned.
"""

from tichu_engine.engine import step
from tichu_engine.legality import legal_actions
from tichu_engine.state import deal_initial_state
from tichu_ml.agent import Agent
from tichu_ml.random_agent import RandomAgent


def test_random_agent_is_an_agent():
    assert isinstance(RandomAgent(seed=0), Agent)


def test_random_agent_plays_a_full_game_with_only_legal_actions():
    for seed in range(10):
        state = deal_initial_state(seed=seed)
        agents = [RandomAgent(seed=seed * 7 + i) for i in range(4)]
        for _ in range(2000):
            legal = legal_actions(state)
            actor = state.public.current_player
            private = state.private_view(actor)
            action = agents[actor].act(private)
            assert action in legal, (
                f"seed={seed}: RandomAgent returned illegal action {action!r}"
            )
            state, _, done, _ = step(state, action)
            if done:
                break
        else:
            raise AssertionError(f"seed={seed}: game did not terminate")
