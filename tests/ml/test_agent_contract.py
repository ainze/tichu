"""Parametrized Agent contract test.

Every concrete `Agent` subclass registered in `AGENT_REGISTRY` must produce
only legal actions across a full random game. This is the safety net that
keeps new agents — learned or rule-based — from silently violating the
engine's invariants.
"""

import pytest

from tichu_engine.engine import step
from tichu_engine.legality import legal_actions
from tichu_engine.state import deal_initial_state, deal_for_schupfen
from tichu_ml import AGENT_REGISTRY, build_agent


def _bare_agents(name, n=4):
    """Four bare instances of `name`, or skip: checkpoint-backed agents (ml, search,
    the forced_* probes) land in the registry via side-effect imports from earlier-
    collected test modules and cannot be built without artifact paths — the contract
    only covers agents constructible with no arguments, independent of which other
    test modules happened to import first."""
    try:
        return [build_agent(name) for _ in range(n)]
    except TypeError as exc:
        pytest.skip(f"{name!r} is not bare-constructible: {exc}")


@pytest.mark.parametrize("name", sorted(AGENT_REGISTRY.keys()))
def test_registered_agent_only_produces_legal_actions(name):
    for seed in range(5):
        state = deal_initial_state(seed=seed)
        agents = _bare_agents(name)
        for _ in range(2000):
            legal = legal_actions(state)
            actor = state.public.current_player
            private = state.private_view(actor)
            action = agents[actor].act(private)
            assert action in legal, (
                f"agent={name} seed={seed}: returned illegal action {action!r}"
            )
            state, _, done, _ = step(state, action)
            if done:
                break
        else:
            raise AssertionError(f"agent={name} seed={seed}: game did not terminate")


@pytest.mark.parametrize("name", sorted(AGENT_REGISTRY.keys()))
def test_registered_agent_handles_schupfen(name):
    for seed in range(3):
        state = deal_for_schupfen(seed=seed)
        agents = _bare_agents(name)
        for _ in range(2000):
            legal = legal_actions(state)
            actor = state.public.current_player
            private = state.private_view(actor)
            action = agents[actor].act(private)
            assert action in legal, (
                f"agent={name} seed={seed} (schupfen): illegal action {action!r}"
            )
            state, _, done, _ = step(state, action)
            if done:
                break
        else:
            raise AssertionError(f"agent={name} seed={seed}: game did not terminate")
