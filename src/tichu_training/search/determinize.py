"""Determinization Sampler (ADR-0030).

Turns an Observation — an acting Player's ``PrivateState`` plus the Belief
Model's ``(3, 56)`` per-card occupancy marginals (relative-seat order:
next / partner / previous) — into a Determinized World: a perfect-information
``GameState`` whose three opponent Hands are a constraint-respecting,
belief-weighted assignment of the unseen cards.

Torch-free: the Belief Model forward happens upstream; this module consumes the
already-computed marginals as a plain array, so it imports cheaply and tests
deterministically.
"""

import random

import numpy as np

from tichu_engine.deck import fresh_deck
from tichu_engine.state import GameState, PrivateState
from tichu_training.card_slots import card_slot


def sample_determinized_world(
    own_view: PrivateState,
    belief: np.ndarray | None,
    rng: random.Random,
) -> GameState:
    """Sample one Determinized World from an Observation.

    ``belief`` is a ``(3, 56)`` array of per-card occupancy probabilities in
    relative-seat order (next / partner / previous); ``None`` selects the
    belief-off ablation (uniform over feasible opponents).
    """
    root = own_view.player
    seen = set(own_view.hand) | set(own_view.public.played_cards_this_round)
    unseen = [c for c in fresh_deck() if c not in seen]
    opp_seats = [(root + 1) % 4, (root + 2) % 4, (root + 3) % 4]

    capacity = {s: own_view.public.hand_sizes[s] for s in opp_seats}
    if sum(capacity.values()) != len(unseen):
        raise ValueError(
            f"opponent capacities {capacity} sum to {sum(capacity.values())}, "
            f"but {len(unseen)} cards are unseen — inconsistent Observation"
        )
    buckets: dict[int, set] = {s: set() for s in opp_seats}
    for card in unseen:
        eligible = [s for s in opp_seats if capacity[s] > 0]
        seat = _draw_holder(card, eligible, opp_seats, belief, rng)
        buckets[seat].add(card)
        capacity[seat] -= 1

    hands = tuple(
        own_view.hand if s == root else frozenset(buckets[s])
        for s in range(4)
    )
    return GameState(hands=hands, public=own_view.public)


def _draw_holder(card, eligible, opp_seats, belief, rng):
    """Pick one capacity-eligible opponent seat for ``card``, weighted by the
    Belief marginals. Falls back to uniform when belief is absent or assigns the
    card zero mass across every eligible opponent."""
    if belief is None:
        return rng.choice(eligible)
    slot = card_slot(card)
    weights = [float(belief[opp_seats.index(s), slot]) for s in eligible]
    if sum(weights) <= 0.0:
        return rng.choice(eligible)
    return rng.choices(eligible, weights=weights, k=1)[0]
