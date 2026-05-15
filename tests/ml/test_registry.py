"""The agent registry — a name -> factory map used by the eval harness.

The harness (issue #007) builds tournament matrices by looking agents up by
name. This module defines that registration mechanism; the harness itself comes
later. Built-in baselines register themselves on import.
"""

import pytest

from tichu_ml.agent import Agent
from tichu_ml.registry import AGENT_REGISTRY, build_agent, register_agent


def test_random_agent_is_registered_by_default():
    assert "random" in AGENT_REGISTRY


def test_build_agent_returns_an_agent_instance():
    agent = build_agent("random")
    assert isinstance(agent, Agent)


def test_build_agent_passes_kwargs_through():
    # `seed` is constructor kwarg for RandomAgent; this confirms the registry
    # doesn't swallow construction parameters.
    a = build_agent("random", seed=123)
    b = build_agent("random", seed=123)
    # Same seed -> first call to act on identical state should be deterministic.
    from tichu_engine.state import deal_initial_state
    state = deal_initial_state(seed=0)
    private = state.private_view(state.public.current_player)
    assert a.act(private) == b.act(private)


def test_register_agent_decorator_adds_class_to_registry():
    @register_agent("toy")
    class _Toy(Agent):
        def act(self, private_state):  # pragma: no cover - not called
            raise NotImplementedError

    try:
        assert "toy" in AGENT_REGISTRY
        assert isinstance(build_agent("toy"), _Toy)
    finally:
        del AGENT_REGISTRY["toy"]


def test_register_agent_rejects_duplicate_names():
    @register_agent("dup")
    class _A(Agent):
        def act(self, private_state):  # pragma: no cover
            raise NotImplementedError

    try:
        with pytest.raises(ValueError, match="dup"):
            @register_agent("dup")
            class _B(Agent):
                def act(self, private_state):  # pragma: no cover
                    raise NotImplementedError
    finally:
        del AGENT_REGISTRY["dup"]


def test_build_agent_raises_on_unknown_name():
    with pytest.raises(KeyError, match="does-not-exist"):
        build_agent("does-not-exist")
