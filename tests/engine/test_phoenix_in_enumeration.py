"""Phoenix-aware enumeration.

These tests cover what enumerators must produce when the hand contains Phoenix.
Singles also exercise Dragon/Mahjong/Dog so that enumeration handles the full
set of special cards a player may hold.
"""

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_engine.combinations import (
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    Triple,
)
from tichu_engine.enumeration import (
    enumerate_full_houses,
    enumerate_pair_steps,
    enumerate_pairs,
    enumerate_singles,
    enumerate_straights,
    enumerate_triples,
)


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


# ---- Singles ----

def test_enumerate_singles_includes_phoenix():
    hand = frozenset({_c(Suit.JADE, 7), PHOENIX})
    result = enumerate_singles(hand)
    assert Single(PHOENIX) in result
    assert Single(_c(Suit.JADE, 7)) in result
    assert len(result) == 2


def test_enumerate_singles_includes_all_special_cards():
    hand = frozenset({DRAGON, MAHJONG, DOG, PHOENIX})
    result = enumerate_singles(hand)
    assert result == frozenset({Single(DRAGON), Single(MAHJONG), Single(DOG), Single(PHOENIX)})


# ---- Pairs ----

def test_enumerate_pairs_phoenix_alone_no_pair():
    # Phoenix alone with no natural card to pair with — no pairs.
    hand = frozenset({PHOENIX})
    assert enumerate_pairs(hand) == frozenset()


def test_enumerate_pairs_phoenix_plus_one_card_makes_one_pair():
    hand = frozenset({PHOENIX, _c(Suit.JADE, 7)})
    result = enumerate_pairs(hand)
    assert result == frozenset({Pair(PHOENIX, _c(Suit.JADE, 7))})


def test_enumerate_pairs_phoenix_with_natural_pair_adds_two_phoenix_pairs():
    # Two 7s + Phoenix -> natural pair (J,S) + Phoenix-pair with J + Phoenix-pair with S = 3 total.
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7), PHOENIX})
    result = enumerate_pairs(hand)
    assert len(result) == 3
    assert Pair(_c(Suit.JADE, 7), _c(Suit.SWORD, 7)) in result
    assert Pair(PHOENIX, _c(Suit.JADE, 7)) in result
    assert Pair(PHOENIX, _c(Suit.SWORD, 7)) in result


def test_enumerate_pairs_phoenix_pairs_with_each_distinct_rank():
    # Phoenix + 7 + 9 -> Phoenix-pair-with-7 + Phoenix-pair-with-9.
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 9), PHOENIX})
    result = enumerate_pairs(hand)
    assert result == frozenset({
        Pair(PHOENIX, _c(Suit.JADE, 7)),
        Pair(PHOENIX, _c(Suit.SWORD, 9)),
    })


# ---- Triples ----

def test_enumerate_triples_phoenix_with_two_of_a_kind_makes_one_triple():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7), PHOENIX})
    result = enumerate_triples(hand)
    assert result == frozenset({Triple(PHOENIX, _c(Suit.JADE, 7), _c(Suit.SWORD, 7))})


def test_enumerate_triples_phoenix_with_three_of_a_kind_adds_phoenix_options():
    # Three 7s + Phoenix -> 1 natural triple + C(3,2)=3 phoenix triples = 4.
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), PHOENIX})
    result = enumerate_triples(hand)
    assert len(result) == 4


def test_enumerate_triples_phoenix_with_single_card_per_rank_makes_no_triples():
    # Phoenix can only substitute for one card; without a pair to extend, no triples.
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 9), PHOENIX})
    assert enumerate_triples(hand) == frozenset()


# ---- Full houses ----

def test_enumerate_full_houses_phoenix_extends_pair_to_triple():
    # Pair of 7s + pair of 9s + Phoenix -> Phoenix can complete either pair to a triple,
    # using the other natural pair: 2 full houses (triple=7+phoenix / triple=9+phoenix).
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7),
        _c(Suit.PAGODA, 9), _c(Suit.STAR, 9),
        PHOENIX,
    })
    result = enumerate_full_houses(hand)
    assert len(result) == 2


def test_enumerate_full_houses_phoenix_extends_single_to_pair():
    # Triple of 7s + single 9 + Phoenix -> Phoenix pairs with the 9: 1 full house.
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7),
        _c(Suit.STAR, 9), PHOENIX,
    })
    result = enumerate_full_houses(hand)
    assert len(result) == 1


def test_enumerate_full_houses_no_double_phoenix():
    # Pair of 7s + single 9 + Phoenix -> Phoenix could extend 7-pair to triple OR
    # pair with 9. Both at once would require two Phoenixes; only one direction is legal.
    # Triple=7+phoenix (using both 7s) + pair=9+phoenix is impossible (one phoenix).
    # Triple=7+phoenix + pair=??? — no natural pair exists. Pair=9+phoenix + triple=??? — no triple.
    # So zero legal full houses.
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7),
        _c(Suit.PAGODA, 9), PHOENIX,
    })
    assert enumerate_full_houses(hand) == frozenset()


# ---- Pair-steps ----

def test_enumerate_pair_steps_phoenix_fills_one_missing_pair():
    # Pair of 7s + single 8 + Phoenix -> Phoenix pairs with 8 for a length-2 step.
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7),
        _c(Suit.PAGODA, 8), PHOENIX,
    })
    result = enumerate_pair_steps(hand)
    assert len(result) == 1
    ps = next(iter(result))
    assert ps.rank == 7 and ps.length == 2


def test_enumerate_pair_steps_phoenix_in_middle_of_three_run():
    # Pair of 7s + single 8 + pair of 9s + Phoenix
    # -> length-3 step at 7 (with Phoenix bridging 8) is one option.
    # Also two length-2 steps possible: 7-8 (phoenix-8) and 8-9 (phoenix-8).
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7),
        _c(Suit.PAGODA, 8),
        _c(Suit.JADE, 9), _c(Suit.STAR, 9),
        PHOENIX,
    })
    result = enumerate_pair_steps(hand)
    # Natural step has no length-2 at 7 (8 has only one card) — wait, 7,8 needs pair at 8.
    # Without phoenix: no consecutive runs because pair at 8 is missing.
    # With phoenix: length-2 (7,8), length-2 (8,9), length-3 (7,8,9). 3 total.
    assert len(result) == 3


# ---- Straights ----

def test_enumerate_straights_phoenix_fills_inner_gap():
    # 5,6,_,8,9 + Phoenix -> one straight (5..9) with phoenix as 7.
    hand = frozenset({
        _c(Suit.JADE, 5), _c(Suit.SWORD, 6),
        _c(Suit.STAR, 8), _c(Suit.JADE, 9),
        PHOENIX,
    })
    result = enumerate_straights(hand)
    assert len(result) == 1
    s = next(iter(result))
    assert s.rank == 5 and s.length == 5
    assert PHOENIX in s.cards


def test_enumerate_straights_phoenix_extends_high_end():
    # Hand: 5,6,7,8,9 + Phoenix.
    # Length-5 windows:
    #   4..8: phoenix=4 + natural 5,6,7,8 -> 1
    #   5..9: natural (no phoenix) -> 1
    #   5..9: phoenix shadows each of 5,6,7,8,9 (natural card stays unused) -> 5
    #   6..10: phoenix=10 + natural 6,7,8,9 -> 1
    # Length-6:
    #   4..9: phoenix=4 + natural 5..9 -> 1
    #   5..10: phoenix=10 + natural 5..9 -> 1
    # Length-7: 4..10 needs phoenix at both 4 and 10 — impossible.
    # Total: 10.
    hand = frozenset({
        _c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7),
        _c(Suit.STAR, 8), _c(Suit.JADE, 9),
        PHOENIX,
    })
    result = enumerate_straights(hand)
    assert len(result) == 10


def test_enumerate_straights_phoenix_extends_low_end():
    # Hand: 6,7,8,9,10 + Phoenix. By symmetry with the high-end test: 10 total.
    hand = frozenset({
        _c(Suit.JADE, 6), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 8),
        _c(Suit.STAR, 9), _c(Suit.JADE, 10),
        PHOENIX,
    })
    result = enumerate_straights(hand)
    assert len(result) == 10
