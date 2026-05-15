"""Uniform-random baseline agent.

`RandomAgent` is the floor against which every other agent is measured. It
samples one action uniformly from the legal-action set at every decision point.
A fixed seed makes its behaviour reproducible across runs.
"""

import random

from tichu_engine.legality import Action, legal_actions_for
from tichu_engine.state import PrivateState

from tichu_ml.agent import Agent
from tichu_ml.registry import register_agent


@register_agent("random")
class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, private_state: PrivateState) -> Action:
        options = list(legal_actions_for(private_state))
        return self._rng.choice(options)
