from tichu_engine.cards import Card, Suit, DRAGON, PHOENIX, MAHJONG, DOG
from tichu_engine.deck import fresh_deck


def test_fresh_deck_has_56_cards():
    assert len(fresh_deck()) == 56


def test_fresh_deck_cards_are_all_unique():
    deck = fresh_deck()
    assert len(set(deck)) == 56


def test_fresh_deck_contains_all_four_specials():
    deck = set(fresh_deck())
    assert DRAGON in deck
    assert PHOENIX in deck
    assert MAHJONG in deck
    assert DOG in deck


def test_fresh_deck_contains_all_52_suited_cards():
    deck = set(fresh_deck())
    for suit in Suit:
        for rank in range(2, 15):  # 2..14 inclusive (Ace = 14)
            assert Card(suit=suit, rank=rank) in deck, f"missing {suit} {rank}"


def test_fresh_deck_is_deterministic():
    # Same call must return cards in the same order across invocations.
    assert fresh_deck() == fresh_deck()
