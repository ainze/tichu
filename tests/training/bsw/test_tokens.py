"""Card-token parsing.

BSW logs encode cards with one-letter suit + one-character rank:
  Suits: B (Blau/blue), G (Grün/green), R (Rot/red), S (Schwarz/black)
  Ranks: 2..9 numeric, 10 as the literal '10', B=Bube (J), D=Dame (Q),
         K=König (K), A=Ass (A)
  Specials: Ma (Mahjong), Hu (Hund/Dog), Dr (Drache/Dragon), Ph (Phoenix)

Mapping to our engine's `Suit` enum is internal-and-stable: it doesn't matter
which colour we map to which `Suit` value as long as we are consistent.
"""

import pytest

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_training.bsw.tokens import parse_card_token


def test_parses_a_simple_numeric_card():
    card = parse_card_token("R2")
    assert isinstance(card, Card)
    assert card.rank == 2


def test_parses_a_ten():
    card = parse_card_token("B10")
    assert isinstance(card, Card)
    assert card.rank == 10


def test_parses_face_ranks():
    for token, expected in [("SB", 11), ("BD", 12), ("GK", 13), ("RA", 14)]:
        card = parse_card_token(token)
        assert isinstance(card, Card)
        assert card.rank == expected, f"{token} -> expected rank {expected}, got {card.rank}"


def test_each_suit_letter_maps_to_a_distinct_suit():
    suits = {parse_card_token(letter + "2").suit for letter in "BGRS"}
    assert len(suits) == 4
    assert suits <= set(Suit)


def test_parses_special_cards():
    assert parse_card_token("Ma") is MAHJONG
    assert parse_card_token("Hu") is DOG
    assert parse_card_token("Dr") is DRAGON
    assert parse_card_token("Ph") is PHOENIX


def test_rejects_unknown_token():
    with pytest.raises(ValueError):
        parse_card_token("ZZ")
    with pytest.raises(ValueError):
        parse_card_token("")
