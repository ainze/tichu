"""Tests for ExplorationAgent (ADR-0030 exploration-critic pilot).

The wrapper perturbs ONLY Play Decisions with uniform ε-greedy over the full
legal set — so beating plays and bombs alike get visited at their structural
frequency, not the master's (suppressed) one. Pending wish/dragon/schupfen and
the Tichu/Grand calls delegate to the wrapped master unchanged, so the self-play
that feeds the critic keeps a realistic call/leadership distribution and only the
in-trick play coverage broadens. Tested torch-free with a deterministic stub.
"""

from dataclasses import replace

from tichu_engine.legality import Pass, legal_actions_for
from tichu_engine.state import MahjongWishPending, PrivateState, deal_initial_state

from tichu_training.search.exploration import ExplorationAgent


class MasterStub:
    """Deterministic master: always Passes if it can, else first legal by repr."""

    def __init__(self):
        self.calls = []

    def act(self, pv):
        legal = list(legal_actions_for(pv))
        for a in legal:
            if isinstance(a, Pass):
                return a
        return min(legal, key=repr)

    def should_call(self, pv, kind):
        self.calls.append(kind)
        return True


def test_epsilon_zero_returns_the_masters_play():
    state = deal_initial_state(seed=5)
    pv = state.private_view(state.public.current_player)
    master = MasterStub()
    agent = ExplorationAgent(master, epsilon=0.0, seed=0)

    assert agent.act(pv) == master.act(pv)


def test_epsilon_one_explores_beyond_the_masters_play():
    # With ε=1 every Play Decision is a uniform-random legal action; over many
    # draws on a state with several legal actions we must see something other than
    # the master's deterministic pick — that surfaced coverage is the whole pilot.
    state = deal_initial_state(seed=5)
    pv = state.private_view(state.public.current_player)
    master = MasterStub()
    masters_pick = master.act(pv)
    agent = ExplorationAgent(master, epsilon=1.0, seed=0)

    seen = {repr(agent.act(pv)) for _ in range(50)}
    legal = list(legal_actions_for(pv))
    assert len(legal) > 1  # guard: the test state must offer a choice
    assert all(a in {repr(x) for x in legal} for a in seen)  # never illegal
    assert seen - {repr(masters_pick)}  # explored past the master


def test_pending_decision_always_delegates_to_master():
    # Even at ε=1, non-Play Decisions (wish/dragon/schupfen) must stay master —
    # the pilot only broadens in-trick play, not the pending heads.
    state = deal_initial_state(seed=5)
    root = state.public.current_player
    pub = replace(state.public, pending_decision=MahjongWishPending(player=root))
    pv = PrivateState(player=root, hand=state.hands[root], public=pub)

    master = MasterStub()
    agent = ExplorationAgent(master, epsilon=1.0, seed=0)
    assert agent.act(pv) == master.act(pv)


def test_should_call_delegates_to_master():
    state = deal_initial_state(seed=5)
    pv = state.private_view(state.public.current_player)
    master = MasterStub()
    agent = ExplorationAgent(master, epsilon=1.0, seed=0)

    assert agent.should_call(pv, "grand") is True
    assert master.calls == ["grand"]
