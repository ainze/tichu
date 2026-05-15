"""Tichu ML package.

Importing `tichu_ml` registers all built-in baseline agents in
`tichu_ml.registry.AGENT_REGISTRY`. Eval harnesses and tournaments look agents
up by name through `build_agent`.
"""

from tichu_ml.agent import Agent
from tichu_ml.registry import AGENT_REGISTRY, build_agent, register_agent

# Side-effect imports: register built-in baselines.
from tichu_ml import random_agent  # noqa: F401
from tichu_ml import rule_agent  # noqa: F401

__all__ = ["Agent", "AGENT_REGISTRY", "build_agent", "register_agent"]
