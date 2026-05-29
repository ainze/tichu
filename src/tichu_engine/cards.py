from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    JADE = "jade"
    SWORD = "sword"
    PAGODA = "pagoda"
    STAR = "star"


# Perfect-hash bases for Card. (Suit, rank) maps to a unique int in
# [0, 64), so no collisions and `hash` returns the canonical value
# directly. See the 2026-05-29 enumeration spike notes for the
# motivation: dataclass-generated `__hash__` was tuple-building +
# recursively hashing the Suit Enum on every call (~143 ns), and Card
# hashing dominated the per-call cost of every combination class via
# its tuple-of-cards hash. A flat int return is ~30 ns.
_SUIT_HASH_BASE: dict["Suit", int] = {
    Suit.JADE: 0,
    Suit.SWORD: 16,
    Suit.PAGODA: 32,
    Suit.STAR: 48,
}


@dataclass(frozen=True, order=False, eq=True)
class Card:
    suit: Suit
    rank: int

    def __post_init__(self) -> None:
        # Cache the perfect hash on the instance. With `frozen=True`,
        # all fields are immutable after construction, so the hash
        # never goes stale. `object.__setattr__` bypasses the frozen
        # guard for this single internal write. Card is hashed
        # heavily by frozenset(hand) operations and recursively by
        # every combination class's tuple-of-cards hash; precomputing
        # turns those into pure attribute reads.
        object.__setattr__(
            self, "_hash", _SUIT_HASH_BASE[self.suit] | self.rank,
        )

    def __lt__(self, other: "Card") -> bool:
        return self.rank < other.rank

    def __gt__(self, other: "Card") -> bool:
        return self.rank > other.rank

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]


# SpecialCards are intended to be singletons (DRAGON / PHOENIX /
# MAHJONG / DOG); the module's `__reduce__` enforces this across
# pickle. A perfect hash by name keeps eq/hash invariants holding even
# if a non-singleton instance slips in (e.g. test code that constructs
# `SpecialCard("dragon")` directly).
_SPECIAL_HASH_BY_NAME: dict[str, int] = {
    "dragon": 200,
    "phoenix": 201,
    "mahjong": 202,
    "dog": 203,
}


@dataclass(frozen=True, eq=True)
class SpecialCard:
    name: str

    def __hash__(self) -> int:
        return _SPECIAL_HASH_BY_NAME.get(self.name, hash(self.name))

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
