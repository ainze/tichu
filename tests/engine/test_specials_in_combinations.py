import pytest

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit
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


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


# ---- Dragon as a single ----

def test_dragon_single_has_rank_25():
    assert Single(DRAGON).rank == 25


def test_dragon_single_beats_ace_single():
    ace = Single(_c(Suit.JADE, 14))
    assert Single(DRAGON) > ace


# ---- Mahjong as a single ----

def test_mahjong_single_has_rank_1():
    assert Single(MAHJONG).rank == 1


def test_mahjong_single_is_beaten_by_any_2():
    two = Single(_c(Suit.JADE, 2))
    assert two > Single(MAHJONG)


def test_mahjong_single_is_below_phoenix_lead():
    # Phoenix as a lead single ranks 1.5 — just above Mahjong, just below 2.
    assert Single(PHOENIX) > Single(MAHJONG)


# ---- Dog as a single ----

def test_dog_single_can_be_constructed():
    # Dog is a legal single — it just doesn't follow rank ordering.
    Single(DOG)


# ---- Phoenix following ----

def test_phoenix_following_a_seven_beats_seven_and_loses_to_eight():
    seven = Single(_c(Suit.JADE, 7))
    eight = Single(_c(Suit.SWORD, 8))
    phoenix_on_seven = Single.phoenix_following(top_rank=7)
    assert phoenix_on_seven > seven
    assert eight > phoenix_on_seven


def test_phoenix_following_an_ace_beats_ace_but_loses_to_dragon():
    ace = Single(_c(Suit.JADE, 14))
    phoenix_on_ace = Single.phoenix_following(top_rank=14)
    assert phoenix_on_ace > ace
    # Dragon is the only single that beats a Phoenix-following.
    assert Single(DRAGON) > phoenix_on_ace


def test_phoenix_following_requires_phoenix():
    # Cannot construct a "following" single from a non-Phoenix card.
    with pytest.raises(ValueError):
        Single(_c(Suit.JADE, 7), as_rank=7.5)


# ---- Phoenix in a pair ----

def test_phoenix_pair_takes_rank_from_the_other_card():
    p = Pair(PHOENIX, _c(Suit.JADE, 7))
    assert p.rank == 7


def test_phoenix_pair_beats_lower_normal_pair():
    sevens_with_phoenix = Pair(PHOENIX, _c(Suit.JADE, 7))
    fives = Pair(_c(Suit.JADE, 5), _c(Suit.SWORD, 5))
    assert sevens_with_phoenix > fives


def test_phoenix_pair_order_independent():
    a = Pair(PHOENIX, _c(Suit.JADE, 7))
    b = Pair(_c(Suit.JADE, 7), PHOENIX)
    assert a == b


def test_pair_cannot_use_dragon_mahjong_or_dog_as_substitute():
    # Only Phoenix substitutes in combinations.
    seven = _c(Suit.JADE, 7)
    for special in (DRAGON, MAHJONG, DOG):
        with pytest.raises(ValueError):
            Pair(special, seven)


# ---- Phoenix in a triple ----

def test_phoenix_triple_takes_rank_from_pair():
    t = Triple(PHOENIX, _c(Suit.JADE, 9), _c(Suit.SWORD, 9))
    assert t.rank == 9


def test_phoenix_triple_with_three_distinct_ranks_raises():
    with pytest.raises(ValueError):
        Triple(PHOENIX, _c(Suit.JADE, 9), _c(Suit.SWORD, 8))


def test_phoenix_triple_beats_lower_normal_triple():
    nine_with_phoenix = Triple(PHOENIX, _c(Suit.JADE, 9), _c(Suit.SWORD, 9))
    sevens = Triple(_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7))
    assert nine_with_phoenix > sevens


# ---- Phoenix in a full house ----

def test_full_house_with_phoenix_in_triple():
    fh = FullHouse(
        triple=Triple(PHOENIX, _c(Suit.JADE, 7), _c(Suit.SWORD, 7)),
        pair=Pair(_c(Suit.PAGODA, 9), _c(Suit.STAR, 9)),
    )
    assert fh.rank == 7


def test_full_house_with_phoenix_in_pair():
    fh = FullHouse(
        triple=Triple(_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7)),
        pair=Pair(PHOENIX, _c(Suit.STAR, 9)),
    )
    assert fh.rank == 7


def test_full_house_cannot_use_phoenix_in_both_triple_and_pair():
    # Only one Phoenix exists in the deck; using it twice in one combination is
    # impossible and must be rejected even if a caller tries.
    with pytest.raises(ValueError):
        FullHouse(
            triple=Triple(PHOENIX, _c(Suit.JADE, 7), _c(Suit.SWORD, 7)),
            pair=Pair(PHOENIX, _c(Suit.PAGODA, 9)),
        )


# ---- Phoenix in a straight ----

def test_phoenix_fills_a_gap_in_a_straight():
    s = Straight(
        cards=(_c(Suit.JADE, 5), _c(Suit.SWORD, 6), PHOENIX, _c(Suit.STAR, 8), _c(Suit.JADE, 9)),
        phoenix_as_rank=7,
    )
    assert s.rank == 5 and s.length == 5


def test_phoenix_extends_the_high_end_of_a_straight():
    s = Straight(
        cards=(_c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8), PHOENIX),
        phoenix_as_rank=9,
    )
    assert s.rank == 5


def test_phoenix_extends_the_low_end_of_a_straight():
    s = Straight(
        cards=(PHOENIX, _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8), _c(Suit.JADE, 9)),
        phoenix_as_rank=5,
    )
    assert s.rank == 5


def test_phoenix_in_straight_requires_substitution_rank():
    with pytest.raises(ValueError):
        Straight(cards=(PHOENIX, _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8), _c(Suit.JADE, 9)))


def test_phoenix_substitution_must_form_a_valid_consecutive_sequence():
    # Phoenix declared as 7, but actual cards have a gap at 8, not at 7.
    with pytest.raises(ValueError):
        Straight(
            cards=(_c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), PHOENIX, _c(Suit.JADE, 10)),
            phoenix_as_rank=7,
        )


def test_phoenix_substitution_cannot_duplicate_an_existing_rank():
    # Cards already contain a 7; Phoenix declared as 7 would create a duplicate.
    with pytest.raises(ValueError):
        Straight(
            cards=(_c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), PHOENIX, _c(Suit.STAR, 9)),
            phoenix_as_rank=7,
        )


def test_phoenix_as_rank_irrelevant_without_phoenix_in_cards():
    # If cards don't contain Phoenix, phoenix_as_rank is forbidden — it's misleading.
    with pytest.raises(ValueError):
        Straight(
            cards=(_c(Suit.JADE, 5), _c(Suit.SWORD, 6), _c(Suit.PAGODA, 7), _c(Suit.STAR, 8), _c(Suit.JADE, 9)),
            phoenix_as_rank=10,
        )


# ---- Phoenix in a pair-step ----

def test_pair_step_with_phoenix_in_one_pair():
    ps = PairStep((
        Pair(_c(Suit.JADE, 7), _c(Suit.SWORD, 7)),
        Pair(PHOENIX, _c(Suit.PAGODA, 8)),
    ))
    assert ps.rank == 7 and ps.length == 2


def test_pair_step_with_phoenix_beats_lower_normal_pair_step():
    high = PairStep((
        Pair(PHOENIX, _c(Suit.JADE, 9)),
        Pair(_c(Suit.SWORD, 10), _c(Suit.PAGODA, 10)),
    ))
    low = PairStep((
        Pair(_c(Suit.JADE, 5), _c(Suit.SWORD, 5)),
        Pair(_c(Suit.PAGODA, 6), _c(Suit.STAR, 6)),
    ))
    assert high > low


# ---- Phoenix forbidden in bombs ----

def test_phoenix_cannot_be_in_four_of_a_kind_bomb():
    with pytest.raises(ValueError):
        FourOfAKindBomb(PHOENIX, _c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7))


def test_phoenix_cannot_be_in_straight_flush_bomb():
    with pytest.raises(ValueError):
        StraightFlushBomb((
            PHOENIX, _c(Suit.JADE, 6), _c(Suit.JADE, 7), _c(Suit.JADE, 8), _c(Suit.JADE, 9),
        ))


def test_dog_single_does_not_compare_to_normal_singles():
    # Dog cannot be played in a trick — its legality is governed externally
    # (it can only be led, and immediately passes to partner). It must therefore
    # neither beat nor be beaten by any normal single.
    dog = Single(DOG)
    five = Single(_c(Suit.JADE, 5))
    assert not (dog > five)
    assert not (five > dog)
    assert not (dog > Single(DRAGON))
    assert not (Single(DRAGON) > dog)
