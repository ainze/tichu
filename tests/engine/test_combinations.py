from tichu_engine.cards import Card, Suit
import pytest

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


def test_single_wraps_one_card():
    c = Card(suit=Suit.JADE, rank=7)
    s = Single(c)
    assert s.rank == 7


def test_higher_single_beats_lower_single():
    seven = Single(Card(suit=Suit.JADE, rank=7))
    nine = Single(Card(suit=Suit.SWORD, rank=9))
    assert nine > seven
    assert seven < nine


def test_singles_of_equal_rank_are_not_strictly_ordered():
    a = Single(Card(suit=Suit.JADE, rank=7))
    b = Single(Card(suit=Suit.SWORD, rank=7))
    assert not (a > b)
    assert not (b > a)


def test_pair_requires_two_cards_of_the_same_rank():
    seven_jade = Card(suit=Suit.JADE, rank=7)
    seven_sword = Card(suit=Suit.SWORD, rank=7)
    p = Pair(seven_jade, seven_sword)
    assert p.rank == 7


def test_pair_with_mismatched_ranks_raises():
    seven = Card(suit=Suit.JADE, rank=7)
    eight = Card(suit=Suit.JADE, rank=8)
    with pytest.raises(ValueError):
        Pair(seven, eight)


def test_pair_with_two_identical_cards_raises():
    # A pair needs two *distinct* cards of the same rank.
    seven = Card(suit=Suit.JADE, rank=7)
    with pytest.raises(ValueError):
        Pair(seven, seven)


def test_higher_pair_beats_lower_pair():
    sevens = Pair(Card(suit=Suit.JADE, rank=7), Card(suit=Suit.SWORD, rank=7))
    nines = Pair(Card(suit=Suit.JADE, rank=9), Card(suit=Suit.STAR, rank=9))
    assert nines > sevens
    assert sevens < nines


def test_triple_requires_three_cards_of_same_rank():
    t = Triple(
        Card(suit=Suit.JADE, rank=7),
        Card(suit=Suit.SWORD, rank=7),
        Card(suit=Suit.PAGODA, rank=7),
    )
    assert t.rank == 7


def test_triple_with_mismatched_ranks_raises():
    with pytest.raises(ValueError):
        Triple(
            Card(suit=Suit.JADE, rank=7),
            Card(suit=Suit.SWORD, rank=7),
            Card(suit=Suit.PAGODA, rank=8),
        )


def test_triple_with_duplicate_cards_raises():
    seven_jade = Card(suit=Suit.JADE, rank=7)
    with pytest.raises(ValueError):
        Triple(seven_jade, seven_jade, Card(suit=Suit.SWORD, rank=7))


def test_higher_triple_beats_lower_triple():
    sevens = Triple(
        Card(suit=Suit.JADE, rank=7), Card(suit=Suit.SWORD, rank=7), Card(suit=Suit.PAGODA, rank=7)
    )
    nines = Triple(
        Card(suit=Suit.JADE, rank=9), Card(suit=Suit.SWORD, rank=9), Card(suit=Suit.STAR, rank=9)
    )
    assert nines > sevens


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def test_straight_requires_at_least_five_cards():
    with pytest.raises(ValueError):
        Straight((_c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8)))


def test_straight_requires_consecutive_ranks():
    with pytest.raises(ValueError):
        Straight((
            _c(Suit.JADE, 5),
            _c(Suit.SWORD, 6),
            _c(Suit.PAGODA, 7),
            _c(Suit.STAR, 8),
            _c(Suit.JADE, 10),  # gap
        ))


def test_straight_rank_is_lowest_card():
    s = Straight((
        _c(Suit.JADE, 5),
        _c(Suit.SWORD, 6),
        _c(Suit.PAGODA, 7),
        _c(Suit.STAR, 8),
        _c(Suit.JADE, 9),
    ))
    assert s.rank == 5
    assert s.length == 5


def test_straight_card_order_is_canonical_by_rank():
    # Passing cards in any order produces the same Straight.
    cards_a = (
        _c(Suit.JADE, 5),
        _c(Suit.SWORD, 6),
        _c(Suit.PAGODA, 7),
        _c(Suit.STAR, 8),
        _c(Suit.JADE, 9),
    )
    cards_b = tuple(reversed(cards_a))
    assert Straight(cards_a) == Straight(cards_b)


def test_higher_straight_beats_lower_straight_of_same_length():
    low = Straight((
        _c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8), _c(Suit.JADE, 9),
    ))
    high = Straight((
        _c(Suit.JADE, 7), _c(Suit.SWORD, 8), _c(Suit.PAGODA, 9), _c(Suit.STAR, 10), _c(Suit.JADE, 11),
    ))
    assert high > low


def test_straights_of_different_lengths_do_not_compare():
    five = Straight((
        _c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8), _c(Suit.JADE, 9),
    ))
    six = Straight((
        _c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8), _c(Suit.JADE, 9), _c(Suit.SWORD, 10),
    ))
    with pytest.raises(TypeError):
        _ = six > five


def _triple(rank: int) -> Triple:
    return Triple(_c(Suit.JADE, rank), _c(Suit.SWORD, rank), _c(Suit.PAGODA, rank))


def _pair(rank: int) -> Pair:
    return Pair(_c(Suit.JADE, rank), _c(Suit.SWORD, rank))


def test_full_house_rank_is_the_triple_rank():
    fh = FullHouse(triple=_triple(9), pair=_pair(5))
    assert fh.rank == 9


def test_full_house_triple_and_pair_must_have_different_ranks():
    with pytest.raises(ValueError):
        FullHouse(triple=_triple(7), pair=Pair(_c(Suit.STAR, 7), _c(Suit.JADE, 7)))


def test_higher_full_house_beats_lower_full_house():
    # Triple rank determines comparison; pair rank is irrelevant.
    low = FullHouse(triple=_triple(7), pair=_pair(12))
    high = FullHouse(triple=_triple(9), pair=_pair(2))
    assert high > low


def test_pair_step_requires_at_least_two_pairs():
    with pytest.raises(ValueError):
        PairStep((_pair(7),))


def test_pair_step_requires_consecutive_ranks():
    with pytest.raises(ValueError):
        PairStep((_pair(7), _pair(9)))  # 8 missing


def test_pair_step_rank_is_lowest_pair_rank():
    ps = PairStep((_pair(7), _pair(8), _pair(9)))
    assert ps.rank == 7
    assert ps.length == 3


def test_higher_pair_step_beats_lower_pair_step_of_same_length():
    low = PairStep((_pair(5), _pair(6)))
    high = PairStep((_pair(9), _pair(10)))
    assert high > low


def test_pair_steps_of_different_lengths_do_not_compare():
    short = PairStep((_pair(5), _pair(6)))
    long = PairStep((_pair(5), _pair(6), _pair(7)))
    with pytest.raises(TypeError):
        _ = long > short


def test_pair_step_pairs_are_stored_in_rank_order():
    ps = PairStep((_pair(9), _pair(7), _pair(8)))
    assert [p.rank for p in ps.pairs] == [7, 8, 9]


def test_four_of_a_kind_bomb_requires_four_same_rank():
    b = FourOfAKindBomb(
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7)
    )
    assert b.rank == 7


def test_four_of_a_kind_bomb_rejects_mismatched_ranks():
    with pytest.raises(ValueError):
        FourOfAKindBomb(
            _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8)
        )


def test_higher_four_bomb_beats_lower_four_bomb():
    low = FourOfAKindBomb(
        _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7)
    )
    high = FourOfAKindBomb(
        _c(Suit.JADE, 9), _c(Suit.SWORD, 9), _c(Suit.PAGODA, 9), _c(Suit.STAR, 9)
    )
    assert high > low


def test_straight_flush_bomb_requires_same_suit():
    with pytest.raises(ValueError):
        StraightFlushBomb((
            _c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7),
            _c(Suit.JADE, 8), _c(Suit.SWORD, 9),  # wrong suit
        ))


def test_straight_flush_bomb_requires_consecutive_ranks():
    with pytest.raises(ValueError):
        StraightFlushBomb((
            _c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7),
            _c(Suit.JADE, 8), _c(Suit.JADE, 10),
        ))


def test_straight_flush_bomb_requires_at_least_five_cards():
    with pytest.raises(ValueError):
        StraightFlushBomb((
            _c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7), _c(Suit.JADE, 8),
        ))


def test_straight_flush_bomb_rank_is_lowest_card():
    b = StraightFlushBomb((
        _c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7),
        _c(Suit.JADE, 8), _c(Suit.JADE, 9),
    ))
    assert b.rank == 5 and b.length == 5


def test_longer_straight_flush_beats_shorter_straight_flush():
    short = StraightFlushBomb((
        _c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7),
        _c(Suit.JADE, 8), _c(Suit.JADE, 9),
    ))
    long = StraightFlushBomb((
        _c(Suit.SWORD, 5), _c(Suit.SWORD, 6), _c(Suit.SWORD, 7),
        _c(Suit.SWORD, 8), _c(Suit.SWORD, 9), _c(Suit.SWORD, 10),
    ))
    assert long > short


def test_same_length_straight_flush_compares_by_lowest_rank():
    low = StraightFlushBomb((
        _c(Suit.JADE, 5), _c(Suit.JADE, 6), _c(Suit.JADE, 7),
        _c(Suit.JADE, 8), _c(Suit.JADE, 9),
    ))
    high = StraightFlushBomb((
        _c(Suit.SWORD, 7), _c(Suit.SWORD, 8), _c(Suit.SWORD, 9),
        _c(Suit.SWORD, 10), _c(Suit.SWORD, 11),
    ))
    assert high > low


def test_different_combination_types_do_not_compare():
    # Comparing a Single to a Pair is a category error — the engine must not
    # silently return False. It must raise so callers cannot accidentally treat
    # mismatched types as orderable.
    s = Single(Card(suit=Suit.JADE, rank=9))
    p = Pair(Card(suit=Suit.JADE, rank=7), Card(suit=Suit.SWORD, rank=7))
    with pytest.raises(TypeError):
        _ = s > p
    with pytest.raises(TypeError):
        _ = p > s
