"""The Agent interface.

An `Agent` is an inference-only policy: given a `PrivateState`, it returns one
`Action`. This is the single integration point between the engine and any
concrete strategy — random, rule-based, or learned.

The interface intentionally carries no training state and no environment
reference. Implementations must not rely on anything beyond what is observable
from the player's `PrivateState`.
"""

from abc import ABC, abstractmethod

from tichu_engine.legality import Action
from tichu_engine.state import PrivateState


class Agent(ABC):
    @abstractmethod
    def act(self, private_state: PrivateState) -> Action:
        """Choose one action given the player's private view of the game."""

    def rank_actions(self, private_state: PrivateState) -> list[Action] | None:
        """Return all legal actions ranked best-to-worst, or `None` if the agent
        does not expose a ranking.

        Used by the held-out move-prediction evaluator to compute top-k accuracy.
        Baseline agents (random, rule) return `None`; learned agents that have a
        head-logit distribution override this to return actions ordered by
        predicted probability.
        """
        return None
