"""Behavioural and contract tests for the rule-based baseline."""

from tichu_engine.engine import step
from tichu_engine.legality import legal_actions
from tichu_engine.state import deal_initial_state, deal_for_schupfen
from tichu_ml.agent import Agent
from tichu_ml.registry import build_agent
from tichu_ml.rule_agent import RuleAgent


def test_rule_agent_is_an_agent():
    assert isinstance(RuleAgent(), Agent)


def test_rule_agent_is_registered_as_rule():
    assert isinstance(build_agent("rule"), RuleAgent)


def test_rule_agent_plays_a_full_game_with_only_legal_actions():
    for seed in range(10):
        state = deal_initial_state(seed=seed)
        agents = [RuleAgent() for _ in range(4)]
        for _ in range(2000):
            legal = legal_actions(state)
            actor = state.public.current_player
            private = state.private_view(actor)
            action = agents[actor].act(private)
            assert action in legal, (
                f"seed={seed}: RuleAgent returned illegal action {action!r}"
            )
            state, _, done, _ = step(state, action)
            if done:
                break
        else:
            raise AssertionError(f"seed={seed}: game did not terminate")


def test_rule_agent_completes_a_game_with_schupfen():
    for seed in range(5):
        state = deal_for_schupfen(seed=seed)
        agents = [RuleAgent() for _ in range(4)]
        for _ in range(2000):
            legal = legal_actions(state)
            actor = state.public.current_player
            private = state.private_view(actor)
            action = agents[actor].act(private)
            assert action in legal, (
                f"seed={seed}: RuleAgent returned illegal schupfen/play action {action!r}"
            )
            state, _, done, _ = step(state, action)
            if done:
                break
        else:
            raise AssertionError(f"seed={seed}: game did not terminate")
