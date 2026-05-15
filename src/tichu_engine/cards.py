from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    JADE = "jade"
    SWORD = "sword"
    PAGODA = "pagoda"
    STAR = "star"


@dataclass(frozen=True, order=False)
class Card:
    suit: Suit
    rank: int

    def __lt__(self, other: "Card") -> bool:
        return self.rank < other.rank

    def __gt__(self, other: "Card") -> bool:
        return self.rank > other.rank


@dataclass(frozen=True)
class SpecialCard:
    name: str


DRAGON = SpecialCard("dragon")
PHOENIX = SpecialCard("phoenix")
MAHJONG = SpecialCard("mahjong")
DOG = SpecialCard("dog")
