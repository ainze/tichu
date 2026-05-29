"""Property-based tests over the combination enumerator.

For arbitrary subsets of the full deck, these properties must hold:
- Every card in every enumerated combination is in the input hand.
- Enumeration is deterministic.
- Enumeration is invariant to the iteration order of the input hand.
- Same-type same-length combinations are mutually orderable (no TypeError).
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tichu_engine.cards import Card, MAHJONG, PHOENIX, Suit
from tichu_engine.combinations import (
    FourOfAKindBomb,
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    StraightFlushBomb,
    Triple,
)
from tichu_engine.deck import fresh_deck
from tichu_engine.enumeration import (
    _no_straight_possible,
    enumerate_four_of_a_kind_bombs,
    enumerate_full_houses,
    enumerate_pair_steps,
    enumerate_pairs,
    enumerate_singles,
    enumerate_straight_flush_bombs,
    enumerate_straights,
    enumerate_triples,
)
from tichu_engine.legality import _cards_in


_DECK = list(fresh_deck())


# Hands up to size 14 (a real Tichu hand). Smaller sizes are well-covered too.
_hands = st.lists(
    st.sampled_from(_DECK), min_size=0, max_size=14, unique=True
).map(frozenset)


def _all_enumerations(hand):
    return (
        set(enumerate_singles(hand))
        | set(enumerate_pairs(hand))
        | set(enumerate_triples(hand))
        | set(enumerate_full_houses(hand))
        | set(enumerate_straights(hand))
        | set(enumerate_pair_steps(hand))
        | set(enumerate_four_of_a_kind_bombs(hand))
        | set(enumerate_straight_flush_bombs(hand))
    )


@given(_hands)
@settings(max_examples=50, deadline=None)
def test_every_combination_uses_only_cards_from_hand(hand):
    for combo in _all_enumerations(hand):
        cards = set(_cards_in(combo))
        assert cards <= hand, f"{combo!r} contains cards not in hand"


@given(_hands)
@settings(max_examples=30, deadline=None)
def test_enumeration_is_deterministic(hand):
    assert enumerate_pairs(hand) == enumerate_pairs(hand)
    assert enumerate_triples(hand) == enumerate_triples(hand)
    assert enumerate_straights(hand) == enumerate_straights(hand)
    assert enumerate_pair_steps(hand) == enumerate_pair_steps(hand)
    assert enumerate_full_houses(hand) == enumerate_full_houses(hand)
    assert enumerate_four_of_a_kind_bombs(hand) == enumerate_four_of_a_kind_bombs(hand)
    assert enumerate_straight_flush_bombs(hand) == enumerate_straight_flush_bombs(hand)


@given(_hands)
@settings(max_examples=30, deadline=None)
def test_enumeration_is_permutation_invariant(hand):
    # Build a "reversed" hand (frozenset has no order, but we materialize via list).
    a = enumerate_pairs(hand)
    b = enumerate_pairs(frozenset(list(hand)[::-1]))
    assert a == b


@given(_hands)
@settings(max_examples=20, deadline=None)
def test_same_type_combinations_are_mutually_comparable(hand):
    # Pairs of same type/length should never raise on comparison.
    pairs = list(enumerate_pairs(hand))[:20]  # cap to keep this cheap
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            # Either ordering should succeed and produce a bool.
            _ = a > b
            _ = a < b


@given(_hands)
@settings(max_examples=20, deadline=None)
def test_straights_of_same_length_are_mutually_comparable(hand):
    straights = list(enumerate_straights(hand))[:20]
    by_length: dict[int, list] = {}
    for s in straights:
        by_length.setdefault(s.length, []).append(s)
    for group in by_length.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                _ = group[i] > group[j]
                _ = group[j] > group[i]


@given(_hands)
@settings(max_examples=20, deadline=None)
def test_pairs_in_a_pair_step_have_consecutive_ranks(hand):
    for ps in enumerate_pair_steps(hand):
        ranks = [p.rank for p in ps.pairs]
        for i in range(1, len(ranks)):
            assert ranks[i] == ranks[i - 1] + 1


@given(_hands)
@settings(max_examples=20, deadline=None)
def test_full_house_triple_and_pair_have_different_ranks(hand):
    for fh in enumerate_full_houses(hand):
        assert fh.triple.rank != fh.pair.rank


def _predicate_inputs(hand):
    ranks_present = frozenset(c.rank for c in hand if isinstance(c, Card))
    return ranks_present, PHOENIX in hand, MAHJONG in hand


@given(_hands)
@settings(max_examples=400, deadline=None)
def test_no_straight_possible_predicate_is_exact(hand):
    """The early-out guard must short-circuit ONLY when the full loop
    would emit nothing. A false positive (predicate True but a straight
    exists) would silently drop legal straights from the legal-action
    set — the one failure mode this optimization could introduce.
    """
    predicted_empty = _no_straight_possible(*_predicate_inputs(hand))
    actual_empty = len(enumerate_straights(hand)) == 0
    assert predicted_empty == actual_empty, (
        f"predicate said empty={predicted_empty} but actual empty="
        f"{actual_empty} for hand {sorted(map(repr, hand))}"
    )


def _h(*cards):
    return frozenset(cards)


_J = Suit.JADE
_S = Suit.SWORD


@pytest.mark.parametrize(
    "hand, expect_no_straight",
    [
        # No specials, no run of 5 -> impossible.
        (_h(Card(_J, 2), Card(_J, 4), Card(_J, 6), Card(_J, 8), Card(_J, 10)), True),
        # Five consecutive naturals -> possible.
        (_h(Card(_J, 2), Card(_J, 3), Card(_J, 4), Card(_J, 5), Card(_J, 6)), False),
        # Single gap bridged by Phoenix (2,3,_,5,6 + Ph) -> possible.
        (_h(Card(_J, 2), Card(_J, 3), Card(_J, 5), Card(_J, 6), PHOENIX), False),
        # Two gaps, only one Phoenix -> impossible (2,_,4,_,6 + Ph).
        (_h(Card(_J, 2), Card(_J, 4), Card(_J, 6), PHOENIX), True),
        # Mahjong-led: ranks 2,3,4,5 + Mahjong fills rank-1 window 1..5 -> possible.
        (_h(MAHJONG, Card(_J, 2), Card(_J, 3), Card(_J, 4), Card(_J, 5)), False),
        # Mahjong + Phoenix bridging a gap in the rank-1 window
        # (1=Ma, 2, _, 4, 5 with Ph filling rank 3) -> possible.
        (_h(MAHJONG, Card(_J, 2), Card(_J, 4), Card(_J, 5), PHOENIX), False),
        # Top-end window 10..14 -> possible.
        (_h(Card(_J, 10), Card(_J, 11), Card(_J, 12), Card(_J, 13), Card(_J, 14)), False),
        # Phoenix alone, scattered ranks, no fillable window -> impossible.
        (_h(Card(_J, 2), Card(_J, 5), Card(_J, 9), Card(_J, 14), PHOENIX), True),
        # Empty hand -> impossible.
        (_h(), True),
    ],
)
def test_no_straight_possible_adversarial_cases(hand, expect_no_straight):
    assert _no_straight_possible(*_predicate_inputs(hand)) is expect_no_straight
    # Cross-check the predicate against the real enumerator.
    assert (len(enumerate_straights(hand)) == 0) is expect_no_straight
