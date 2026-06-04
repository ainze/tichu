"""EngineWorld — the Tichu `world` that PUCT/MCTS searches (ADR-0030).

Wraps the real rules engine and a policy behind the abstract `world` interface
(`legal_actions` / `prior` / `transition` / `leaf_value`). Single-rooted: only the
root seat's Play Decisions branch; the other three seats are advanced *inside*
`transition` by the policy's `act` (the `master` opponent model). Leaf evaluation
is pluggable — v1 ships `fast_rollout_leaf` (a cheap random playout); the
critic-bootstrap leaf is a drop-in replacement once a Value Baseline is persisted.

The policy is duck-typed to match `MLAgent`:
    play_action_scores(private_state) -> list[(ConcreteAction, prob)]   # PUCT prior
    act(private_state)                -> ConcreteAction                 # env move
"""

import random

from tichu_engine.engine import step
from tichu_engine.legality import legal_actions
from tichu_engine.state import GameState


def round_outcome(state: GameState, root: int) -> float:
    """Team-relative round result for the root seat, normalized to ~[-1, 1]."""
    t = root % 2
    s = state.public.scores
    return (s[t] - s[1 - t]) / 100.0


def fast_rollout_leaf(state: GameState, root: int, rng: random.Random) -> float:
    """v1 leaf: a cheap random-policy playout to round-terminal. No torch — keeps
    the deep part of every simulation off the GPU (the prototype showed rollouts
    dominate cost). The critic-bootstrap leaf is the documented upgrade."""
    while True:
        action = rng.choice(tuple(legal_actions(state)))
        state, _, done, _ = step(state, action)
        if done:
            return round_outcome(state, root)


class EngineWorld:
    __slots__ = ("root", "_policy", "_leaf_fn")

    def __init__(self, root: int, policy, leaf_fn) -> None:
        self.root = root
        self._policy = policy
        self._leaf_fn = leaf_fn

    def legal_actions(self, state: GameState):
        return tuple(legal_actions(state))

    def prior(self, state: GameState) -> dict:
        legal = self.legal_actions(state)
        scores = dict(self._policy.play_action_scores(state.private_view(self.root)))
        floor = 1e-6  # unmapped legal actions still need a (tiny) prior for PUCT
        return {a: scores.get(a, floor) for a in legal}

    def transition(self, state: GameState, action, rng: random.Random):
        """Apply the root action, then advance the env to the next root decision."""
        state, _, done, _ = step(state, action)
        if done:
            return None, round_outcome(state, self.root), True
        state, done = self._advance(state, rng)
        if done:
            return None, round_outcome(state, self.root), True
        return state, 0.0, False

    def leaf_value(self, state: GameState, rng: random.Random) -> float:
        return self._leaf_fn(state, self.root, rng)

    def _advance(self, state: GameState, rng: random.Random):
        """Step the other seats (and any root pending Decision — Play-only search
        delegates wish/dragon to the policy) until the root's next Play Decision or
        round end."""
        while not (state.public.current_player == self.root
                   and state.public.pending_decision is None):
            actor = state.public.current_player
            action = self._policy.act(state.private_view(actor))
            state, _, done, _ = step(state, action)
            if done:
                return state, True
        return state, False
