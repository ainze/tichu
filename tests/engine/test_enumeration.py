from tichu_engine.cards import Card, Suit
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
from tichu_engine.enumeration import (
    enumerate_four_of_a_kind_bombs,
    enumerate_full_houses,
    enumerate_pair_steps,
    enumerate_pairs,
    enumerate_singles,
    enumerate_straight_flush_bombs,
    enumerate_straights,
    enumerate_triples,
)


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def test_enumerate_singles_from_empty_hand_is_empty():
    assert enumerate_singles(frozenset()) == frozenset()


def test_enumerate_singles_returns_one_single_per_card():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 9), _c(Suit.PAGODA, 12)})
    result = enumerate_singles(hand)
    assert result == frozenset({Single(_c(Suit.JADE, 7)), Single(_c(Suit.SWORD, 9)), Single(_c(Suit.PAGODA, 12))})


def test_enumerate_singles_is_deterministic_set():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7)})
    # Two singles of rank 7 in different suits — both are legal as different singles.
    result = enumerate_singles(hand)
    assert len(result) == 2


def test_enumerate_pairs_empty_hand_is_empty():
    assert enumerate_pairs(frozenset()) == frozenset()


def test_enumerate_pairs_no_matching_ranks_is_empty():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 9), _c(Suit.PAGODA, 12)})
    assert enumerate_pairs(hand) == frozenset()


def test_enumerate_pairs_finds_single_pair():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 12)})
    result = enumerate_pairs(hand)
    assert result == frozenset({Pair(_c(Suit.JADE, 7), _c(Suit.SWORD, 7))})


def test_enumerate_pairs_with_three_same_rank_finds_three_pairs():
    # Three 7s -> C(3,2) = 3 distinct pairs.
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7)})
    result = enumerate_pairs(hand)
    assert len(result) == 3


def test_enumerate_triples_empty_hand_is_empty():
    assert enumerate_triples(frozenset()) == frozenset()


def test_enumerate_triples_two_of_a_kind_is_empty():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7)})
    assert enumerate_triples(hand) == frozenset()


def test_enumerate_triples_three_of_a_kind_finds_one():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7)})
    result = enumerate_triples(hand)
    assert result == frozenset(
        {Triple(_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7))}
    )


def test_enumerate_triples_four_of_a_kind_finds_four_triples():
    # Four 7s -> C(4,3) = 4 distinct triples.
    hand = frozenset(
        {_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7)}
    )
    assert len(enumerate_triples(hand)) == 4


def test_enumerate_straights_short_hand_is_empty():
    hand = frozenset({_c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8)})
    assert enumerate_straights(hand) == frozenset()


def test_enumerate_straights_finds_one_minimal_straight():
    hand = frozenset({
        _c(Suit.JADE, 5),
        _c(Suit.SWORD, 6),
        _c(Suit.PAGODA, 7),
        _c(Suit.STAR, 8),
        _c(Suit.JADE, 9),
    })
    result = enumerate_straights(hand)
    assert len(result) == 1
    s = next(iter(result))
    assert s.length == 5 and s.rank == 5


def test_enumerate_straights_finds_all_starting_positions():
    # Six consecutive ranks -> a 5-straight starting at 5, a 5-straight starting at 6,
    # and a 6-straight starting at 5.  3 total.
    hand = frozenset({
        _c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8),
        _c(Suit.JADE, 9), _c(Suit.SWORD, 10),
    })
    result = enumerate_straights(hand)
    assert len(result) == 3


def test_enumerate_full_houses_empty_when_no_triple():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 9), _c(Suit.STAR, 9)})
    assert enumerate_full_houses(hand) == frozenset()


def test_enumerate_full_houses_empty_when_no_pair():
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 9)
    })
    assert enumerate_full_houses(hand) == frozenset()


def test_enumerate_full_houses_minimal_case():
    # One triple (rank 7) + one pair (rank 9) -> 1 full house.
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7),
        _c(Suit.JADE, 9), _c(Suit.SWORD, 9),
    })
    result = enumerate_full_houses(hand)
    assert len(result) == 1


def test_enumerate_full_houses_counts_choices():
    # Four 7s and three 9s -> C(4,3) * C(3,2) = 4 * 3 = 12 full houses (triple=7, pair=9).
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7),
        _c(Suit.JADE, 9), _c(Suit.SWORD, 9), _c(Suit.PAGODA, 9),
    })
    result = enumerate_full_houses(hand)
    # Plus reverse: triple=9 (C(3,3)=1) * pair=7 (C(4,2)=6) = 6 more = 18 total.
    assert len(result) == 12 + 6


def test_enumerate_pair_steps_short_hand_empty():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 7)})
    assert enumerate_pair_steps(hand) == frozenset()


def test_enumerate_pair_steps_finds_minimal_step():
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7),
        _c(Suit.PAGODA, 8), _c(Suit.STAR, 8),
    })
    result = enumerate_pair_steps(hand)
    assert len(result) == 1


def test_enumerate_pair_steps_finds_longer_runs():
    # Three consecutive pair-rank groups -> three length-2 steps + one length-3 step = 3.
    # length-2 starting at 7, length-2 starting at 8, length-3 starting at 7.
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7),
        _c(Suit.JADE, 8), _c(Suit.SWORD, 8),
        _c(Suit.JADE, 9), _c(Suit.SWORD, 9),
    })
    result = enumerate_pair_steps(hand)
    assert len(result) == 3


def test_enumerate_four_of_a_kind_bombs_empty_when_no_quadruplet():
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7),
        _c(Suit.JADE, 9), _c(Suit.SWORD, 9),
    })
    assert enumerate_four_of_a_kind_bombs(hand) == frozenset()


def test_enumerate_four_of_a_kind_bombs_finds_one():
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7),
    })
    result = enumerate_four_of_a_kind_bombs(hand)
    assert len(result) == 1
    assert next(iter(result)).rank == 7


def test_enumerate_straight_flush_bombs_short_hand_empty():
    hand = frozenset({
        _c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7), _c(Suit.JADE, 8),
    })
    assert enumerate_straight_flush_bombs(hand) == frozenset()


def test_enumerate_straight_flush_bombs_mixed_suit_empty():
    # Consecutive ranks but multiple suits — not a single-suit straight flush.
    hand = frozenset({
        _c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.JADE, 7), _c(Suit.JADE, 8), _c(Suit.JADE, 9),
    })
    assert enumerate_straight_flush_bombs(hand) == frozenset()


def test_enumerate_straight_flush_bombs_finds_minimal():
    hand = frozenset({
        _c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7), _c(Suit.JADE, 8), _c(Suit.JADE, 9),
    })
    result = enumerate_straight_flush_bombs(hand)
    assert len(result) == 1


def test_enumerate_straight_flush_bombs_finds_all_lengths_and_starts():
    # Six consecutive cards in one suit -> length-5 starting at 5, length-5 at 6, length-6 at 5.
    hand = frozenset({
        _c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7),
        _c(Suit.JADE, 8), _c(Suit.JADE, 9), _c(Suit.JADE, 10),
    })
    result = enumerate_straight_flush_bombs(hand)
    assert len(result) == 3


def test_enumerate_pair_steps_gap_breaks_run():
    hand = frozenset({
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7),
        _c(Suit.JADE, 9), _c(Suit.SWORD, 9),
    })
    assert enumerate_pair_steps(hand) == frozenset()


def test_enumerate_straights_handles_rank_with_duplicate_suits():
    # Two 5s, then 6,7,8,9 -> two distinct 5-straights (one per 5).
    hand = frozenset({
        _c(Suit.JADE, 5), _c(Suit.SWORD, 5),
        _c(Suit.PAGODA, 6), _c(Suit.STAR, 7),
        _c(Suit.JADE, 8), _c(Suit.SWORD, 9),
    })
    result = enumerate_straights(hand)
    assert len(result) == 2
