from dataclasses import dataclass
from typing import Union

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, SpecialCard


CardOrSpecial = Union[Card, SpecialCard]


# When a special card is played as a Single, its effective rank for ordering:
#   Mahjong  = 1   (lowest single)
#   Phoenix  = 1.5 (just above Mahjong when leading; see also Single.phoenix_following)
#   Dragon   = 25  (highest single)
#   Dog      = N/A — Dog is non-comparable, see _DOG_RANK below.
_DOG_RANK = float("nan")  # sentinel: comparisons with NaN are always False

_SPECIAL_SINGLE_RANK: dict[SpecialCard, float] = {
    MAHJONG: 1,
    PHOENIX: 1.5,
    DRAGON: 25,
    DOG: _DOG_RANK,
}


class _SameTypeRankCompare:
    """Mixin: two combinations of the same concrete type compare by their `.rank` attribute."""

    rank: int

    def __lt__(self, other: "object") -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.rank < other.rank  # type: ignore[attr-defined]

    def __gt__(self, other: "object") -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.rank > other.rank  # type: ignore[attr-defined]


class _SameTypeSameLengthRankCompare:
    """Mixin: combinations of the same type compare by rank only when their `length` is equal."""

    rank: int
    length: int

    def __lt__(self, other: "object") -> bool:
        if type(self) is not type(other) or self.length != other.length:  # type: ignore[attr-defined]
            return NotImplemented
        return self.rank < other.rank  # type: ignore[attr-defined]

    def __gt__(self, other: "object") -> bool:
        if type(self) is not type(other) or self.length != other.length:  # type: ignore[attr-defined]
            return NotImplemented
        return self.rank > other.rank  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Single(_SameTypeRankCompare):
    card: CardOrSpecial
    # Only set when the card is the Phoenix being played onto a known top card —
    # in which case the Phoenix's effective rank is top_rank + 0.5.
    as_rank: float | None = None

    def __post_init__(self) -> None:
        if self.as_rank is not None and self.card is not PHOENIX:
            raise ValueError("as_rank is only valid when the card is Phoenix")
        # Cache the dataclass-equivalent hash on the instance. Combination
        # classes are immutable (`frozen=True`) so the hash never goes
        # stale. Subsequent `hash(x)` calls become a single attribute
        # read. See 2026-05-29 card-hash-caching notes for context.
        object.__setattr__(self, "_hash", hash((self.card, self.as_rank)))

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    @property
    def rank(self) -> float:
        if self.as_rank is not None:
            return self.as_rank
        if isinstance(self.card, SpecialCard):
            return _SPECIAL_SINGLE_RANK[self.card]
        return self.card.rank

    @classmethod
    def phoenix_following(cls, top_rank: float) -> "Single":
        """Phoenix played onto a single of the given rank: beats by 0.5."""
        return cls(PHOENIX, as_rank=top_rank + 0.5)


def _validate_same_rank_group(
    cards: tuple[CardOrSpecial, ...], expected_size: int, kind: str
) -> None:
    if len(cards) != expected_size:
        raise ValueError(f"{kind} requires {expected_size} cards, got {len(cards)}")
    if len(set(cards)) != expected_size:
        raise ValueError(f"{kind} requires {expected_size} distinct cards, got duplicates")
    # Only Phoenix may substitute in a rank-group combination. Dragon/Mahjong/Dog cannot.
    for c in cards:
        if isinstance(c, SpecialCard) and c is not PHOENIX:
            raise ValueError(f"{kind} cannot contain special card {c.name}")
    normal_ranks = {c.rank for c in cards if isinstance(c, Card)}
    if len(normal_ranks) > 1:
        raise ValueError(f"{kind} requires all non-Phoenix cards to share a rank, got {sorted(normal_ranks)}")
    if not normal_ranks:
        # A combination made entirely of Phoenix(es) has no anchoring rank.
        raise ValueError(f"{kind} requires at least one non-Phoenix card to anchor the rank")


def _canonical_order(cards: tuple[CardOrSpecial, ...]) -> tuple[CardOrSpecial, ...]:
    """Sort so that combinations are equal regardless of construction order.
    Normal cards come first (ordered by suit), then Phoenix (or any other special) last."""
    def key(c: CardOrSpecial) -> tuple[int, str]:
        if isinstance(c, Card):
            return (0, c.suit.value)
        return (1, c.name)
    return tuple(sorted(cards, key=key))


def _anchored_rank(cards: tuple[CardOrSpecial, ...]) -> int:
    """Return the rank of the first non-Phoenix card. Caller must have validated."""
    for c in cards:
        if isinstance(c, Card):
            return c.rank
    raise AssertionError("unreachable: validation requires at least one non-Phoenix card")


@dataclass(frozen=True)
class Pair(_SameTypeRankCompare):
    a: CardOrSpecial
    b: CardOrSpecial

    def __post_init__(self) -> None:
        _validate_same_rank_group((self.a, self.b), 2, "Pair")
        a, b = _canonical_order((self.a, self.b))
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "_hash", hash((a, b)))

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    @property
    def rank(self) -> int:
        return _anchored_rank((self.a, self.b))


@dataclass(frozen=True)
class Triple(_SameTypeRankCompare):
    a: CardOrSpecial
    b: CardOrSpecial
    c: CardOrSpecial

    def __post_init__(self) -> None:
        _validate_same_rank_group((self.a, self.b, self.c), 3, "Triple")
        a, b, c = _canonical_order((self.a, self.b, self.c))
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "_hash", hash((a, b, c)))

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    @property
    def rank(self) -> int:
        return _anchored_rank((self.a, self.b, self.c))


@dataclass(frozen=True)
class FourOfAKindBomb(_SameTypeRankCompare):
    a: Card
    b: Card
    c: Card
    d: Card

    def __post_init__(self) -> None:
        for x in (self.a, self.b, self.c, self.d):
            if isinstance(x, SpecialCard):
                raise ValueError("FourOfAKindBomb cannot contain any special card (including Phoenix)")
        _validate_same_rank_group((self.a, self.b, self.c, self.d), 4, "FourOfAKindBomb")
        a, b, c, d = _canonical_order((self.a, self.b, self.c, self.d))
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "_hash", hash((a, b, c, d)))

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    @property
    def rank(self) -> int:
        return self.a.rank


@dataclass(frozen=True)
class StraightFlushBomb:
    cards: tuple[Card, ...]

    def __post_init__(self) -> None:
        if len(self.cards) < 5:
            raise ValueError(f"StraightFlushBomb requires at least 5 cards, got {len(self.cards)}")
        if len(set(self.cards)) != len(self.cards):
            raise ValueError("StraightFlushBomb cards must be distinct")
        for c in self.cards:
            if isinstance(c, SpecialCard):
                raise ValueError("StraightFlushBomb cannot contain any special card (including Phoenix)")
        suits = {c.suit for c in self.cards}
        if len(suits) != 1:
            raise ValueError(f"StraightFlushBomb requires one suit, got {suits}")
        sorted_cards = tuple(sorted(self.cards, key=lambda c: c.rank))
        ranks = [c.rank for c in sorted_cards]
        for i in range(1, len(ranks)):
            if ranks[i] != ranks[i - 1] + 1:
                raise ValueError(f"StraightFlushBomb requires consecutive ranks, got {ranks}")
        object.__setattr__(self, "cards", sorted_cards)
        object.__setattr__(self, "_hash", hash((sorted_cards,)))

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    @property
    def rank(self) -> int:
        return self.cards[0].rank

    @property
    def length(self) -> int:
        return len(self.cards)

    def __lt__(self, other: "object") -> bool:
        if type(self) is not type(other):
            return NotImplemented
        if self.length != other.length:  # type: ignore[attr-defined]
            return self.length < other.length  # type: ignore[attr-defined]
        return self.rank < other.rank  # type: ignore[attr-defined]

    def __gt__(self, other: "object") -> bool:
        if type(self) is not type(other):
            return NotImplemented
        if self.length != other.length:  # type: ignore[attr-defined]
            return self.length > other.length  # type: ignore[attr-defined]
        return self.rank > other.rank  # type: ignore[attr-defined]


@dataclass(frozen=True)
class FullHouse(_SameTypeRankCompare):
    triple: "Triple"
    pair: "Pair"

    def __post_init__(self) -> None:
        if self.triple.rank == self.pair.rank:
            raise ValueError(
                f"FullHouse triple and pair must differ in rank, got both at {self.triple.rank}"
            )
        triple_has_phoenix = any(c is PHOENIX for c in (self.triple.a, self.triple.b, self.triple.c))
        pair_has_phoenix = any(c is PHOENIX for c in (self.pair.a, self.pair.b))
        if triple_has_phoenix and pair_has_phoenix:
            raise ValueError("FullHouse cannot use Phoenix in both the triple and the pair")
        object.__setattr__(self, "_hash", hash((self.triple, self.pair)))

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    @property
    def rank(self) -> int:
        return self.triple.rank


@dataclass(frozen=True)
class PairStep(_SameTypeSameLengthRankCompare):
    pairs: tuple[Pair, ...]

    def __post_init__(self) -> None:
        if len(self.pairs) < 2:
            raise ValueError(f"PairStep requires at least 2 pairs, got {len(self.pairs)}")
        sorted_pairs = tuple(sorted(self.pairs, key=lambda p: p.rank))
        ranks = [p.rank for p in sorted_pairs]
        for i in range(1, len(ranks)):
            if ranks[i] != ranks[i - 1] + 1:
                raise ValueError(f"PairStep requires consecutive pair ranks, got {ranks}")
        object.__setattr__(self, "pairs", sorted_pairs)
        object.__setattr__(self, "_hash", hash((sorted_pairs,)))

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    @property
    def rank(self) -> int:
        return self.pairs[0].rank

    @property
    def length(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class Straight(_SameTypeSameLengthRankCompare):
    cards: tuple[CardOrSpecial, ...]
    phoenix_as_rank: int | None = None

    def __post_init__(self) -> None:
        if len(self.cards) < 5:
            raise ValueError(f"Straight requires at least 5 cards, got {len(self.cards)}")
        if len(set(self.cards)) != len(self.cards):
            raise ValueError("Straight cards must be distinct")
        for c in self.cards:
            if isinstance(c, SpecialCard) and c is not PHOENIX and c is not MAHJONG:
                raise ValueError(f"Straight cannot contain special card {c.name}")
        has_phoenix = PHOENIX in self.cards
        has_mahjong = MAHJONG in self.cards
        if has_phoenix and self.phoenix_as_rank is None:
            raise ValueError("phoenix_as_rank is required when Phoenix is in the straight")
        if not has_phoenix and self.phoenix_as_rank is not None:
            raise ValueError("phoenix_as_rank is only valid when Phoenix is in the straight")
        # Validate that the resulting rank sequence is consecutive and has no duplicates.
        normal_ranks = [c.rank for c in self.cards if isinstance(c, Card)]
        all_ranks = sorted(normal_ranks)
        if has_phoenix:
            if self.phoenix_as_rank in all_ranks:
                raise ValueError(
                    f"phoenix_as_rank={self.phoenix_as_rank} duplicates an existing card rank"
                )
            all_ranks = sorted(all_ranks + [self.phoenix_as_rank])
        if has_mahjong:
            # Mahjong has effective rank 1 (lowest in any straight).
            if 1 in all_ranks:
                raise ValueError("Straight cannot contain both Mahjong and a rank-1 card")
            all_ranks = sorted(all_ranks + [1])
        for i in range(1, len(all_ranks)):
            if all_ranks[i] != all_ranks[i - 1] + 1:
                raise ValueError(f"Straight requires consecutive ranks, got {all_ranks}")
        # Canonicalize: normal cards in rank order, Phoenix/Mahjong at their slots.
        sorted_cards = self._canonical_card_order(all_ranks)
        object.__setattr__(self, "cards", sorted_cards)
        object.__setattr__(
            self, "_hash", hash((sorted_cards, self.phoenix_as_rank)),
        )

    def __hash__(self) -> int:
        return self._hash  # type: ignore[attr-defined]

    def _canonical_card_order(self, all_ranks: list[int]) -> tuple[CardOrSpecial, ...]:
        cards_by_rank: dict[int, CardOrSpecial] = {}
        for c in self.cards:
            if isinstance(c, Card):
                cards_by_rank[c.rank] = c
        if self.phoenix_as_rank is not None:
            cards_by_rank[self.phoenix_as_rank] = PHOENIX
        if MAHJONG in self.cards:
            cards_by_rank[1] = MAHJONG
        return tuple(cards_by_rank[r] for r in all_ranks)

    @property
    def rank(self) -> int:
        first = self.cards[0]
        if isinstance(first, Card):
            return first.rank
        if first is MAHJONG:
            return 1
        # First slot is Phoenix at the low end.
        assert self.phoenix_as_rank is not None
        return self.phoenix_as_rank

    @property
    def length(self) -> int:
        return len(self.cards)

    @classmethod
    def _from_canonical_cards(
        cls,
        cards: tuple,
        phoenix_as_rank: int | None,
    ) -> "Straight":
        """Skip `__post_init__` validation. Trusted construction path for
        callers (notably `enumerate_straights`) that have already
        verified inputs and pre-arranged `cards` in canonical order.

        Canonical order: ascending effective rank (Mahjong at rank 1 if
        present, Phoenix at `phoenix_as_rank` if present, normal cards
        at their natural rank).

        Misuse → equality/hash collisions with regularly-constructed
        Straights silently break. Do not call this from anywhere that
        hasn't been audited against the validation rules in
        `__post_init__`.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "cards", cards)
        object.__setattr__(obj, "phoenix_as_rank", phoenix_as_rank)
        # Must match the regular `__post_init__` hash so canonical-path
        # and trusted-path Straights collide in sets/frozensets.
        object.__setattr__(obj, "_hash", hash((cards, phoenix_as_rank)))
        return obj
