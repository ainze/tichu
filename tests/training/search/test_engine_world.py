"""Tests for EngineWorld — the Tichu `world` the MCTS searches (ADR-0030).

EngineWorld wires the real rules engine (transitions, terminal detection,
round_outcome) and a policy (PUCT prior + in-tree env moves) behind the abstract
`world` interface `run_mcts` consumes. Tested with the REAL engine but a cheap
deterministic stub policy, so no torch / checkpoint is needed.
"""

import random

from tichu_engine.legality import legal_actions, legal_actions_for
from tichu_engine.state import deal_initial_state
from tichu_training.search.determinize import sample_determinized_world
from tichu_training.search.engine_world import (
    EngineWorld,
    fast_rollout_leaf,
    sample_from_scores,
)
from tichu_training.search.mcts import pimc_decide


def test_sample_from_scores_is_proportional_and_always_listed():
    # τ=1 sampling over a policy's play_action_scores: frequencies track the probs and
    # the result is always one of the offered actions. This is the in-tree opponent
    # model the learning loop uses instead of greedy argmax (ADR-0031 Decision F).
    rng = random.Random(0)
    scores = [("A", 0.7), ("B", 0.2), ("C", 0.1)]
    counts = {"A": 0, "B": 0, "C": 0}
    for _ in range(4000):
        counts[sample_from_scores(scores, rng)] += 1
    assert sum(counts.values()) == 4000
    assert 0.65 < counts["A"] / 4000 < 0.75
    assert 0.15 < counts["B"] / 4000 < 0.25
    assert counts["C"] > 0


class StubPolicy:
    """Deterministic, torch-free stand-in for the master MLAgent: uniform prior,
    and a fixed legal move for env seats."""

    def play_action_scores(self, private_state):
        legal = list(legal_actions_for(private_state))
        return [(a, 1.0 / len(legal)) for a in legal]

    def act(self, private_state):
        return min(legal_actions_for(private_state), key=repr)


def _root_observation(seed):
    state = deal_initial_state(seed=seed)
    root = state.public.current_player
    return state.private_view(root), root


def test_transition_lands_on_a_root_decision_or_terminal():
    own_view, root = _root_observation(seed=7)
    world_state = sample_determinized_world(own_view, None, random.Random(0))
    world = EngineWorld(root, StubPolicy(), fast_rollout_leaf)

    action = next(iter(world.legal_actions(world_state)))
    child, value, done = world.transition(world_state, action, random.Random(0))

    if done:
        assert child is None and isinstance(value, float)
    else:
        # The env has been advanced back to the root's own Play decision.
        assert child.public.current_player == root
        assert child.public.pending_decision is None


def test_engine_world_opponent_sample_makes_transitions_stochastic():
    # Default (greedy) world: opponents argmax, so a fixed root action + world yields
    # one deterministic child regardless of rng. opponent_sample=True draws opponents
    # from the policy, so the same root action explores several continuation lines.
    own_view, root = _root_observation(seed=7)
    ws = sample_determinized_world(own_view, None, random.Random(0))
    greedy = EngineWorld(root, StubPolicy(), fast_rollout_leaf)
    sampler = EngineWorld(root, StubPolicy(), fast_rollout_leaf, opponent_sample=True)
    action = next(iter(greedy.legal_actions(ws)))

    def distinct_children(world):
        seen = set()
        for s in range(30):
            child, _, done = world.transition(ws, action, random.Random(s))
            seen.add("DONE" if done else repr(child))
        return seen

    assert len(distinct_children(greedy)) == 1
    assert len(distinct_children(sampler)) > 1


def test_leaf_value_returns_a_normalized_outcome():
    own_view, root = _root_observation(seed=11)
    world_state = sample_determinized_world(own_view, None, random.Random(1))

    value = fast_rollout_leaf(world_state, root, random.Random(1))

    assert isinstance(value, float)
    assert -3.0 <= value <= 3.0  # round_outcome / 100, comfortably bounded


def test_pimc_over_the_real_engine_returns_a_legal_root_action():
    own_view, root = _root_observation(seed=3)

    def make_world(rng):
        ws = sample_determinized_world(own_view, None, rng)
        return EngineWorld(root, StubPolicy(), fast_rollout_leaf), ws

    best, visits, _ = pimc_decide(make_world, worlds=3, sims=15, rng=random.Random(0))

    legal = set(legal_actions(sample_determinized_world(own_view, None, random.Random(0))))
    assert best in legal
    assert sum(visits.values()) == 3 * 15
