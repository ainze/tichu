from tichu_engine.cards import Card, Suit, DRAGON, PHOENIX, MAHJONG, DOG


def test_card_has_suit_and_rank():
    c = Card(suit=Suit.JADE, rank=7)
    assert c.suit is Suit.JADE
    assert c.rank == 7


def test_higher_rank_beats_lower_rank_same_suit():
    seven = Card(suit=Suit.JADE, rank=7)
    nine = Card(suit=Suit.JADE, rank=9)
    assert nine > seven
    assert seven < nine


def test_higher_rank_beats_lower_rank_different_suit():
    # In Tichu, suit does not affect single-card comparison (except for bombs)
    seven_jade = Card(suit=Suit.JADE, rank=7)
    nine_sword = Card(suit=Suit.SWORD, rank=9)
    assert nine_sword > seven_jade


def test_four_special_cards_are_distinct():
    specials = [DRAGON, PHOENIX, MAHJONG, DOG]
    assert len(set(specials)) == 4


def test_specials_are_distinct_from_normal_cards():
    # No special should equal any (suit, rank) card.
    for special in (DRAGON, PHOENIX, MAHJONG, DOG):
        for suit in Suit:
            for rank in range(2, 15):
                assert special != Card(suit=suit, rank=rank)


def test_specials_are_hashable_and_usable_in_sets():
    assert {DRAGON, DRAGON} == {DRAGON}
    assert {DRAGON, PHOENIX} != {DRAGON}


def test_card_is_immutable():
    import dataclasses
    c = Card(suit=Suit.JADE, rank=7)
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        c.rank = 9  # type: ignore[misc]


def test_card_is_hashable():
    # Required for set membership in deck / hand operations.
    assert {Card(suit=Suit.JADE, rank=7), Card(suit=Suit.JADE, rank=7)} == {Card(suit=Suit.JADE, rank=7)}


def test_equal_rank_different_suit_is_not_strictly_ordered():
    # Two non-special cards of same rank: neither beats the other.
    seven_jade = Card(suit=Suit.JADE, rank=7)
    seven_sword = Card(suit=Suit.SWORD, rank=7)
    assert not (seven_jade > seven_sword)
    assert not (seven_sword > seven_jade)
