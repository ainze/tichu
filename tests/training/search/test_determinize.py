"""Tests for the Determinization Sampler (ADR-0030).

The sampler turns an Observation (an acting Player's PrivateState + the Belief
Model's per-card marginals) into a Determinized World — a perfect-information
GameState whose three opponent Hands are a constraint-respecting, belief-weighted
assignment of the unseen cards. These tests pin the sampler's *contract* (the
partition + capacity invariants), not its internal sampling mechanics.
"""

import random

import numpy as np
import pytest

from tichu_engine.deck import fresh_deck
from tichu_engine.state import PrivateState, PublicState, Trick, deal_initial_state
from tichu_training.card_slots import card_slot
from tichu_training.search.determinize import sample_determinized_world


def _unseen_cards(own_view):
    """Cards held by the three opponents = deck − own Hand − played-this-round."""
    seen = set(own_view.hand) | set(own_view.public.played_cards_this_round)
    return set(fresh_deck()) - seen


def _opponent_seats(root):
    return [(root + 1) % 4, (root + 2) % 4, (root + 3) % 4]


def test_sampled_world_is_a_valid_partition_of_the_unseen_cards():
    # Tracer bullet: a fresh deal — root holds 14, the other 42 are unseen.
    state = deal_initial_state(seed=7)
    root = state.public.current_player
    own_view = state.private_view(root)
    belief = np.full((3, 56), 1.0 / 3.0, dtype=np.float32)

    world = sample_determinized_world(own_view, belief, random.Random(0))

    # Root's own Hand is preserved exactly.
    assert world.hands[root] == own_view.hand

    opp_seats = _opponent_seats(root)
    opp_hands = [set(world.hands[s]) for s in opp_seats]
    unseen = _unseen_cards(own_view)

    # Opponent Hands partition the unseen set: pairwise disjoint, union == unseen.
    assert opp_hands[0] | opp_hands[1] | opp_hands[2] == unseen
    assert opp_hands[0].isdisjoint(opp_hands[1])
    assert opp_hands[0].isdisjoint(opp_hands[2])
    assert opp_hands[1].isdisjoint(opp_hands[2])

    # No opponent is dealt a card the root holds or one already played.
    for h in opp_hands:
        assert h.isdisjoint(own_view.hand)
        assert h.isdisjoint(own_view.public.played_cards_this_round)


def test_opponent_hand_sizes_match_public_hand_sizes():
    state = deal_initial_state(seed=3)
    root = state.public.current_player
    own_view = state.private_view(root)
    belief = np.full((3, 56), 1.0 / 3.0, dtype=np.float32)

    world = sample_determinized_world(own_view, belief, random.Random(1))

    for s in _opponent_seats(root):
        assert len(world.hands[s]) == own_view.public.hand_sizes[s]


def test_belief_marginals_place_cards_at_the_mapped_opponent_seat():
    # Belief row i is the i-th relative opponent (next / partner / previous) =
    # absolute seat (root + 1 + i) % 4. A near-certain marginal must land the
    # card at exactly that seat — pinning both "belief is used" and the axis map.
    state = deal_initial_state(seed=5)
    root = state.public.current_player
    own_view = state.private_view(root)
    cards = sorted(_unseen_cards(own_view), key=card_slot)[:3]

    belief = np.full((3, 56), 1.0 / 3.0, dtype=np.float32)
    for i, c in enumerate(cards):
        belief[:, card_slot(c)] = 0.0
        belief[i, card_slot(c)] = 1.0

    world = sample_determinized_world(own_view, belief, random.Random(2))

    opp_seats = _opponent_seats(root)
    for i, c in enumerate(cards):
        assert c in world.hands[opp_seats[i]]


def test_capacity_constraints_override_adversarial_belief():
    # A synthetic Observation: own=14, played=30, so 12 cards are unseen, split
    # across opponents with capacities (1, 1, 10). Belief insists every unseen
    # card belongs to the capacity-1 opponent — sizes must still hold.
    deck = list(fresh_deck())
    own = frozenset(deck[:14])
    unseen = deck[14:26]
    played = frozenset(deck[26:])
    public = PublicState(
        current_player=0,
        hand_sizes=(14, 1, 1, 10),
        scores=(0, 0),
        trick=Trick.empty(),
        played_cards_this_round=played,
    )
    own_view = PrivateState(player=0, hand=own, public=public)

    belief = np.zeros((3, 56), dtype=np.float32)
    for c in unseen:
        belief[0, card_slot(c)] = 1.0  # all want relative-opponent 0 = seat 1

    world = sample_determinized_world(own_view, belief, random.Random(4))

    assert len(world.hands[1]) == 1
    assert len(world.hands[2]) == 1
    assert len(world.hands[3]) == 10
    assert set(world.hands[1]) | set(world.hands[2]) | set(world.hands[3]) == set(unseen)


def test_belief_off_samples_uniformly_over_feasible_opponents():
    # The belief-off ablation arm: belief=None must still produce valid worlds,
    # and assign uniformly. The first unseen card is always placed first, when
    # all three opponents have full capacity, so it lands ~1/3 at each.
    state = deal_initial_state(seed=11)
    root = state.public.current_player
    own_view = state.private_view(root)
    opp_seats = _opponent_seats(root)
    first_card = min(_unseen_cards(own_view), key=card_slot)

    counts = {s: 0 for s in opp_seats}
    n = 1500
    for seed in range(n):
        world = sample_determinized_world(own_view, None, random.Random(seed))
        for s in opp_seats:
            assert len(world.hands[s]) == own_view.public.hand_sizes[s]
            if first_card in world.hands[s]:
                counts[s] += 1

    for s in opp_seats:
        assert abs(counts[s] - n / 3) < 0.1 * n  # ~8 sigma band; never flaky


def test_same_rng_seed_yields_identical_world():
    state = deal_initial_state(seed=2)
    root = state.public.current_player
    own_view = state.private_view(root)
    belief = np.full((3, 56), 1.0 / 3.0, dtype=np.float32)

    w1 = sample_determinized_world(own_view, belief, random.Random(99))
    w2 = sample_determinized_world(own_view, belief, random.Random(99))
    assert w1.hands == w2.hands

    w3 = sample_determinized_world(own_view, belief, random.Random(100))
    assert w1.hands != w3.hands  # different seed → different world (overwhelmingly)


def test_opponent_already_out_receives_no_cards():
    # Seat 2 has gone out (hand_size 0); the 12 unseen cards split between the
    # two opponents that still hold cards. The out opponent gets nothing.
    deck = list(fresh_deck())
    own = frozenset(deck[:14])
    unseen = deck[14:26]
    played = frozenset(deck[26:])
    public = PublicState(
        current_player=0,
        hand_sizes=(14, 6, 0, 6),
        scores=(0, 0),
        trick=Trick.empty(),
        played_cards_this_round=played,
    )
    own_view = PrivateState(player=0, hand=own, public=public)
    belief = np.full((3, 56), 1.0 / 3.0, dtype=np.float32)

    world = sample_determinized_world(own_view, belief, random.Random(0))

    assert world.hands[2] == frozenset()
    assert len(world.hands[1]) == 6
    assert len(world.hands[3]) == 6
    assert set(world.hands[1]) | set(world.hands[3]) == set(unseen)


def test_inconsistent_hand_sizes_raise_rather_than_corrupt_silently():
    # Opponent capacities (7+1+10=18) exceed the 12 unseen cards — an impossible
    # Observation. The sampler must fail loudly, not return undersized Hands the
    # search would then trust.
    deck = list(fresh_deck())
    own = frozenset(deck[:14])
    played = frozenset(deck[26:])  # 30 played → 12 unseen
    public = PublicState(
        current_player=0,
        hand_sizes=(14, 7, 1, 10),  # opponents sum to 18 ≠ 12
        scores=(0, 0),
        trick=Trick.empty(),
        played_cards_this_round=played,
    )
    own_view = PrivateState(player=0, hand=own, public=public)
    belief = np.full((3, 56), 1.0 / 3.0, dtype=np.float32)

    with pytest.raises(ValueError):
        sample_determinized_world(own_view, belief, random.Random(0))
