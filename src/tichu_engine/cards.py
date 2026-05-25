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

    def __reduce__(self):
        # Engine and featurizer use `is` comparisons against the module-level
        # singletons (DRAGON / PHOENIX / MAHJONG / DOG). Without this, pickle
        # would round-trip to fresh instances and identity-based checks would
        # silently fail across process boundaries (e.g. ProcessPoolExecutor
        # workers). Pickle back to the singleton lookup so identity is
        # preserved.
        return (_special_by_name, (self.name,))


DRAGON = SpecialCard("dragon")
PHOENIX = SpecialCard("phoenix")
MAHJONG = SpecialCard("mahjong")
DOG = SpecialCard("dog")

_SPECIALS_BY_NAME: dict[str, SpecialCard] = {
    "dragon": DRAGON, "phoenix": PHOENIX, "mahjong": MAHJONG, "dog": DOG,
}


def _special_by_name(name: str) -> SpecialCard:
    return _SPECIALS_BY_NAME[name]
