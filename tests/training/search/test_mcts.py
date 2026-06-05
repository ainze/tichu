"""Tests for the single-world PUCT/MCTS core (ADR-0030).

`run_mcts` is engine-agnostic: it drives an injected `world` (legal_actions /
prior / transition / leaf_value) so the search mechanics — selection, expansion,
multi-level backup, root visit accounting — are tested deterministically without
torch or the rules engine. The Tichu wiring (real engine + master policy + leaf
rollout) is a separate `world` implementation tested at the agent level.
"""

import random

from tichu_training.search.mcts import pimc_decide, run_mcts, visit_policy


def test_visit_policy_is_normalized_and_proportional():
    # The policy-improvement target (ADR-0031 Decision C): π ∝ aggregated root visits,
    # normalised over the legal actions, with 0-visit actions getting 0 mass.
    pi = visit_policy({"a": 30, "b": 10, "c": 0})
    assert abs(sum(pi.values()) - 1.0) < 1e-9
    assert abs(pi["a"] - 0.75) < 1e-9
    assert abs(pi["b"] - 0.25) < 1e-9
    assert pi["c"] == 0.0


def test_visit_policy_single_action_is_certain():
    assert visit_policy({"only": 7}) == {"only": 1.0}


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


class SkewedWorld:
    """Two root actions, both terminal value 0.0, with a near-degenerate prior
    (hi=0.99, lo=0.01). With equal action-values, PUCT after the forced first visit
    is governed purely by the prior, so 'lo' is starved — exactly the suppressed-move
    case root Dirichlet noise (ADR-0031 Decision G) must rescue."""

    def legal_actions(self, state):
        return ("hi", "lo")

    def prior(self, state):
        return {"hi": 0.99, "lo": 0.01}

    def transition(self, state, action, rng):
        return None, 0.0, True

    def leaf_value(self, state, rng):
        return 0.0


def test_root_dirichlet_noise_rescues_a_suppressed_action():
    # No noise: 'lo' gets only its single forced expansion, then the 0.99-prior 'hi'
    # wins every PUCT step. With ε=0.5 Dirichlet noise on the root prior, 'lo' is
    # floored well above 0.01 and gets visited many more times.
    bare, _ = run_mcts(SkewedWorld(), "root", sims=50, rng=random.Random(0))
    noised, _ = run_mcts(SkewedWorld(), "root", sims=50, rng=random.Random(0),
                         root_noise=(1.0, 0.5))
    assert bare["lo"] == 1
    assert noised["lo"] > 5


def test_root_noise_none_matches_default():
    # The exploration knob is strictly opt-in: omitting it and passing None are identical,
    # so the frozen-search path (ADR-0030) is untouched.
    default, _ = run_mcts(FakeWorld(), "root", sims=40, rng=random.Random(7))
    explicit_none, _ = run_mcts(FakeWorld(), "root", sims=40, rng=random.Random(7),
                                root_noise=None)
    assert default == explicit_none


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


def test_pimc_threads_root_noise_into_each_world():
    # The self-play loop runs PIMC with root Dirichlet noise; pimc_decide must pass it
    # through to every per-world run_mcts so the suppressed action is explored in all of
    # them (ADR-0031 Decision G), not just a single tree.
    def make_world(rng):
        return SkewedWorld(), "root"

    _, bare, _ = pimc_decide(make_world, worlds=2, sims=25, rng=random.Random(0))
    _, noised, _ = pimc_decide(make_world, worlds=2, sims=25, rng=random.Random(0),
                               root_noise=(1.0, 0.5))
    assert bare["lo"] == 2          # one forced expansion per world, then starved
    assert noised["lo"] > bare["lo"]


def test_pimc_aggregates_visits_across_worlds_and_picks_best():
    # K identical worlds: aggregate visits = K x single-world visits; the best
    # action (RISKY) still wins, and the total visit budget is worlds x sims.
    def make_world(rng):
        return FakeWorld(), "root"

    best, visits, _ = pimc_decide(make_world, worlds=5, sims=100, rng=random.Random(3))
    assert best == "RISKY"
    assert sum(visits.values()) == 5 * 100
