"""Tests for SearchAgent (ADR-0030).

SearchAgent runs PIMC on Play Decisions and delegates everything else (pending
wish/dragon/schupfen, and calls via should_call) to the wrapped master policy.
Tested via `from_policy` with a torch-free stub — the registered constructor that
builds a real MLAgent is exercised at the Tournament integration step.
"""

import random
from dataclasses import replace

from tichu_engine.legality import legal_actions_for
from tichu_engine.state import MahjongWishPending, PrivateState, deal_initial_state
from tichu_training.search.agent import SearchAgent


class RealishStub:
    """Torch-free master stand-in: uniform prior, deterministic legal env move."""

    def play_action_scores(self, pv):
        legal = list(legal_actions_for(pv))
        return [(a, 1.0 / len(legal)) for a in legal]

    def act(self, pv):
        return min(legal_actions_for(pv), key=repr)


class DelegationStub:
    def __init__(self):
        self.act_called = False
        self.call_kind = None

    def act(self, pv):
        self.act_called = True
        return "DELEGATED-ACT"

    def should_call(self, pv, kind):
        self.call_kind = kind
        return True


def test_play_decision_runs_search_and_returns_a_legal_action():
    state = deal_initial_state(seed=5)
    root = state.public.current_player
    pv = state.private_view(root)

    agent = SearchAgent.from_policy(RealishStub(), worlds=2, sims=10, seed=0)
    action = agent.act(pv)

    assert action in legal_actions_for(pv)


def test_pending_decision_delegates_to_the_policy():
    state = deal_initial_state(seed=5)
    root = state.public.current_player
    pub = replace(state.public, pending_decision=MahjongWishPending(player=root))
    pv = PrivateState(player=root, hand=state.hands[root], public=pub)

    stub = DelegationStub()
    agent = SearchAgent.from_policy(stub, worlds=2, sims=10, seed=0)

    assert agent.act(pv) == "DELEGATED-ACT"
    assert stub.act_called


def test_should_call_delegates_to_the_policy():
    state = deal_initial_state(seed=5)
    root = state.public.current_player
    pv = state.private_view(root)

    stub = DelegationStub()
    agent = SearchAgent.from_policy(stub, worlds=2, sims=10, seed=0)

    assert agent.should_call(pv, "tichu") is True
    assert stub.call_kind == "tichu"
