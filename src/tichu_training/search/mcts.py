"""Single-world PUCT/MCTS core for Phase-2 PIMC (ADR-0030).

Engine-agnostic on purpose: `run_mcts` drives an injected ``world`` so the search
mechanics are testable without torch or the rules engine. A ``world`` exposes:

    legal_actions(state) -> tuple          # the root seat's choices at a node
    prior(state)         -> dict[action, float]   # PUCT prior (master policy)
    transition(state, action, rng) -> (child_state | None, terminal_value, done)
        # apply the root action and advance the env (other seats) to the next
        # root decision; if the round ends, done=True and terminal_value is the
        # team-relative round_outcome
    leaf_value(state, rng) -> float        # value estimate at a freshly expanded node

The tree branches ONLY on the root seat's own Decisions (single-rooted); the other
three seats live inside ``transition`` as environment dynamics. Values are always
from the root team's perspective, so backup is a plain sum — no per-seat sign flip.
"""

import math
import random
from collections import defaultdict


class Node:
    __slots__ = ("state", "actions", "prior", "N", "W", "nsum", "children")

    def __init__(self, state, actions, prior) -> None:
        self.state = state
        self.actions = actions
        self.prior = prior
        self.N = {a: 0 for a in actions}
        self.W = {a: 0.0 for a in actions}
        self.nsum = 0
        self.children: dict = {}  # action -> Node | ("TERMINAL", value)


def run_mcts(world, root_state, sims: int, rng: random.Random, c_puct: float = 1.4):
    """Run ``sims`` PUCT simulations from ``root_state`` in one world.

    Returns ``(visits, values)`` — per root-action visit counts ``N`` and summed
    backed-up values ``W`` (so the mean action-value is ``W[a] / N[a]``)."""
    root = Node(root_state, tuple(world.legal_actions(root_state)), world.prior(root_state))
    for _ in range(sims):
        _simulate(world, root, rng, c_puct)
    return dict(root.N), dict(root.W)


def pimc_decide(make_world, worlds: int, sims: int, rng: random.Random, c_puct: float = 1.4):
    """Root-parallel PIMC: sample ``worlds`` Determinized Worlds, run ``sims`` MCTS
    in each, sum root visit counts across worlds, pick the argmax action.

    ``make_world(rng) -> (world, root_state)`` produces one Determinized World and
    its root decision state. Returns ``(best_action, agg_visits, agg_values)``."""
    agg_visits: dict = defaultdict(int)
    agg_values: dict = defaultdict(float)
    for _ in range(worlds):
        world, root_state = make_world(rng)
        visits, values = run_mcts(world, root_state, sims, rng, c_puct)
        for a in visits:
            agg_visits[a] += visits[a]
            agg_values[a] += values[a]
    best = max(agg_visits, key=lambda a: agg_visits[a])
    return best, dict(agg_visits), dict(agg_values)


def _simulate(world, node: Node, rng: random.Random, c_puct: float) -> float:
    a = _select(node, c_puct, rng)
    child = node.children.get(a)
    if child is None:
        child_state, terminal_value, done = world.transition(node.state, a, rng)
        if done:
            node.children[a] = ("TERMINAL", terminal_value)
            value = terminal_value
        else:
            node.children[a] = Node(
                child_state,
                tuple(world.legal_actions(child_state)),
                world.prior(child_state),
            )
            value = world.leaf_value(child_state, rng)  # expand: one leaf estimate
    elif isinstance(child, tuple):  # ("TERMINAL", value)
        value = child[1]
    else:
        value = _simulate(world, child, rng, c_puct)
    node.N[a] += 1
    node.W[a] += value
    node.nsum += 1
    return value


def _select(node: Node, c_puct: float, rng: random.Random):
    """PUCT selection. Every action is tried once before the formula governs."""
    unvisited = [a for a in node.actions if node.N[a] == 0]
    if unvisited:
        return rng.choice(unvisited)
    sqrt_total = math.sqrt(node.nsum)
    return max(
        node.actions,
        key=lambda a: node.W[a] / node.N[a]
        + c_puct * node.prior[a] * sqrt_total / (1 + node.N[a]),
    )
