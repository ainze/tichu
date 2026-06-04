"""Tests for the single-world PUCT/MCTS core (ADR-0030).

`run_mcts` is engine-agnostic: it drives an injected `world` (legal_actions /
prior / transition / leaf_value) so the search mechanics — selection, expansion,
multi-level backup, root visit accounting — are tested deterministically without
torch or the rules engine. The Tichu wiring (real engine + master policy + leaf
rollout) is a separate `world` implementation tested at the agent level.
"""

import random

from tichu_training.search.mcts import pimc_decide, run_mcts


class FakeWorld:
    """A tiny fixed game tree (root chooses SAFE vs RISKY).

        SAFE  -> terminal 0.3
        RISKY -> child; child chooses BAD (terminal 0.0) or GOOD (terminal 1.0)

    The best line is RISKY then GOOD (= 1.0), so a working search must back the
    deep GOOD value up through the RISKY child and prefer RISKY at the root.
    """

    _LEGAL = {"root": ("SAFE", "RISKY"), "risky": ("BAD", "GOOD")}
    _TRANSITIONS = {
        ("root", "SAFE"): (None, 0.3, True),
        ("root", "RISKY"): ("risky", 0.0, False),
        ("risky", "BAD"): (None, 0.0, True),
        ("risky", "GOOD"): (None, 1.0, True),
    }

    def legal_actions(self, state):
        return self._LEGAL[state]

    def prior(self, state):
        actions = self._LEGAL[state]
        return {a: 1.0 / len(actions) for a in actions}

    def transition(self, state, action, rng):
        return self._TRANSITIONS[(state, action)]

    def leaf_value(self, state, rng):
        return 0.5  # neutral non-terminal estimate


def test_every_root_action_is_expanded_at_least_once():
    visits, _ = run_mcts(FakeWorld(), "root", sims=20, rng=random.Random(0))
    assert set(visits) == {"SAFE", "RISKY"}
    assert all(v >= 1 for v in visits.values())


def test_root_visit_counts_sum_to_sims():
    visits, _ = run_mcts(FakeWorld(), "root", sims=50, rng=random.Random(1))
    assert sum(visits.values()) == 50


def test_search_prefers_the_deep_winning_line():
    # RISKY -> GOOD = 1.0 beats SAFE = 0.3; multi-level backup must surface it.
    visits, values = run_mcts(FakeWorld(), "root", sims=400, rng=random.Random(2))
    assert visits["RISKY"] > visits["SAFE"]
    q_risky = values["RISKY"] / visits["RISKY"]
    assert q_risky > 0.3  # backed-up value exceeds the safe terminal


def test_pimc_aggregates_visits_across_worlds_and_picks_best():
    # K identical worlds: aggregate visits = K x single-world visits; the best
    # action (RISKY) still wins, and the total visit budget is worlds x sims.
    def make_world(rng):
        return FakeWorld(), "root"

    best, visits, _ = pimc_decide(make_world, worlds=5, sims=100, rng=random.Random(3))
    assert best == "RISKY"
    assert sum(visits.values()) == 5 * 100
