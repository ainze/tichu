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


def sample_from_scores(scores, rng: random.Random):
    """Sample one action from ``[(action, prob)]`` proportional to prob (τ=1).

    The in-tree opponent model for the search+learning loop (ADR-0031 Decision F):
    instead of greedy argmax over the policy, draw from its play distribution so the
    visit target reflects play that is good against the opponent's *distribution*,
    not one deterministic line. Falls back to the last action on float round-off, and
    to a uniform pick if every prob is zero (the unmapped-legal-actions case)."""
    total = sum(p for _, p in scores)
    if total <= 0.0:
        return rng.choice([a for a, _ in scores])
    r = rng.random() * total
    upto = 0.0
    for action, p in scores:
        upto += p
        if r <= upto:
            return action
    return scores[-1][0]


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
    __slots__ = ("root", "_policy", "_leaf_fn", "_opponent_sample")

    def __init__(self, root: int, policy, leaf_fn, *, opponent_sample: bool = False) -> None:
        self.root = root
        self._policy = policy
        self._leaf_fn = leaf_fn
        # opponent_sample=False (default) advances in-tree seats by greedy policy.act
        # — the frozen-experiment behaviour (ADR-0030). True samples Play Decisions
        # from the policy (τ=1) for the learning loop (ADR-0031 Decision F).
        self._opponent_sample = bool(opponent_sample)

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
            view = state.private_view(actor)
            # Sample only normal Play Decisions; pending wish/dragon/schupfen have no
            # play distribution, so they always delegate to the policy's own choice.
            if self._opponent_sample and state.public.pending_decision is None:
                action = sample_from_scores(self._policy.play_action_scores(view), rng)
            else:
                action = self._policy.act(view)
            state, _, done, _ = step(state, action)
            if done:
                return state, True
        return state, False
