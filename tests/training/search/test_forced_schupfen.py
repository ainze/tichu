"""ForcedSchupfenAgent — schupfen-coupling EV-probe (ADR-0033 follow-up).

Plays the master everywhere EXCEPT the Schupfen Decision, which it overrides with
either the RuleAgent heuristic ("3 lowest cards") or a random legal SchupfenPass.
Tournamented vs master, it tests whether master's *learned* schupfen bottlenecks
play EV (the play<->schupfen coupling hypothesis):
  * a heuristic/different schupfen beats master -> schupfen is a lever;
  * random << master but heuristic ~= master    -> schupfen matters, master is good;
  * random ~= master                            -> schupfen barely affects EV.
"""

from __future__ import annotations

from tichu_engine.legality import SchupfenPass, legal_actions_for
from tichu_engine.state import deal_for_schupfen, deal_initial_state
from tichu_ml.rule_agent import RuleAgent
from tichu_training.search.forced_schupfen import ForcedSchupfenAgent


class _FakePolicy:
    """Records whether the wrapped policy was asked to act (it must NOT be, for a
    Schupfen Decision) and returns a sentinel for any delegated decision."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, private_state):
        self.calls += 1
        return ("policy-move", private_state.player)

    def should_call(self, private_state, kind):
        return False


def test_rule_mode_overrides_schupfen_with_the_rule_heuristic():
    state = deal_for_schupfen(seed=0)
    pv = state.private_view(state.public.current_player)
    fake = _FakePolicy()
    agent = ForcedSchupfenAgent.from_policy(fake, mode="rule")

    action = agent.act(pv)

    assert isinstance(action, SchupfenPass)
    assert action == RuleAgent().act(pv)   # exactly the rule heuristic
    assert fake.calls == 0                  # the master was NOT consulted for schupfen
    assert agent.forced_count == 1


def test_random_mode_returns_a_legal_schupfen_deterministically():
    state = deal_for_schupfen(seed=1)
    pv = state.private_view(state.public.current_player)
    legal = {a for a in legal_actions_for(pv) if isinstance(a, SchupfenPass)}

    a1 = ForcedSchupfenAgent.from_policy(_FakePolicy(), mode="random", seed=7).act(pv)
    a2 = ForcedSchupfenAgent.from_policy(_FakePolicy(), mode="random", seed=7).act(pv)

    assert isinstance(a1, SchupfenPass) and a1 in legal
    assert a1 == a2                         # deterministic under a fixed seed


def test_non_schupfen_decisions_delegate_to_the_master_policy():
    state = deal_initial_state(seed=0)      # a Play lead (no pending decision)
    pv = state.private_view(state.public.current_player)
    fake = _FakePolicy()
    agent = ForcedSchupfenAgent.from_policy(fake, mode="rule")

    result = agent.act(pv)

    assert result == ("policy-move", pv.player)
    assert fake.calls == 1
    assert agent.forced_count == 0
