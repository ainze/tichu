"""Five or more consecutive cards of one suit are a straight-flush bomb and
nothing else. They cannot be played as a plain Straight (which a higher
unsuited straight could beat)."""

import pytest

from tichu_engine.cards import MAHJONG, PHOENIX, Card, Suit
from tichu_engine.combinations import Straight, StraightFlushBomb
from tichu_engine.enumeration import enumerate_straights
from tichu_engine.legality import _enumerate_all


def _run(suit: Suit, ranks) -> tuple[Card, ...]:
    return tuple(Card(suit=suit, rank=r) for r in ranks)


def _one_suit(s: Straight) -> bool:
    return all(isinstance(c, Card) for c in s.cards) and len({c.suit for c in s.cards}) == 1


def test_suited_run_is_not_enumerated_as_a_straight():
    hand = frozenset(_run(Suit.JADE, range(5, 10)))
    assert enumerate_straights(hand) == frozenset()


def test_suited_run_is_only_legal_as_a_bomb():
    cards = _run(Suit.JADE, range(5, 10))
    combos = _enumerate_all(frozenset(cards))
    assert StraightFlushBomb(cards) in combos
    assert not any(isinstance(c, Straight) for c in combos)


def test_mixed_suit_straights_survive_next_to_a_suited_run():
    # Green 5-9 plus a red 7: the straights using the red 7 are mixed-suit
    # and stay legal; the all-green one does not.
    hand = frozenset(_run(Suit.JADE, range(5, 10)) + (Card(suit=Suit.STAR, rank=7),))
    straights = enumerate_straights(hand)
    assert len(straights) == 1
    assert not any(_one_suit(s) for s in straights)


def test_longer_suited_run_drops_every_one_suit_window():
    hand = frozenset(_run(Suit.SWORD, range(2, 9)) + (Card(suit=Suit.PAGODA, rank=9),))
    straights = enumerate_straights(hand)
    assert straights  # e.g. 5-9 using the pagoda 9
    assert not any(_one_suit(s) for s in straights)


def test_phoenix_or_mahjong_breaks_the_flush():
    hand = frozenset(_run(Suit.JADE, (5, 6, 8, 9)) + (PHOENIX,))
    assert any(PHOENIX in s.cards for s in enumerate_straights(hand))
    hand = frozenset(_run(Suit.JADE, range(2, 6)) + (MAHJONG,))
    assert any(MAHJONG in s.cards for s in enumerate_straights(hand))


def test_straight_constructor_rejects_one_suit():
    with pytest.raises(ValueError):
        Straight(_run(Suit.JADE, range(5, 10)))
