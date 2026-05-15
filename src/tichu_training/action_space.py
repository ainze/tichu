"""Canonical, intent-level action space for Tichu (v1).

This module enumerates every distinct *intent* an agent can express, in a
fixed order, and exposes a stable index → action mapping plus the inverse.

Concrete card-suit picks (e.g. which two 3s make up a "pair of 3s") are NOT
part of the action — the wrapper resolves them heuristically before submitting
to the engine. This keeps the action space ~1.8k entries instead of ~10k.

The version string is frozen at v1. Any change to the ordering or membership
requires bumping `ACTION_SPACE_VERSION`.
"""

from dataclasses import dataclass
import logging
from typing import Union


ACTION_SPACE_VERSION: str = "v1"


# ---------------------------------------------------------------------------
# Action sum type.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaySingle:
    """Play a single card.

    Exactly one of `(suit, rank)`, `special`, or `phoenix_as_rank` identifies it:
      - 52 natural cards: (suit, rank) set, special=None, phoenix_as_rank=None.
      - 4 special-card singles (mahjong, dog, dragon, phoenix): special set.
      - 13 phoenix-following singles (phoenix played onto a known top rank):
            phoenix_as_rank in 2..14, others None.
    """
    suit: str | None = None
    rank: int | None = None
    special: str | None = None
    phoenix_as_rank: int | None = None


@dataclass(frozen=True)
class PlayPair:
    rank: int
    with_phoenix: bool


@dataclass(frozen=True)
class PlayTriple:
    rank: int
    with_phoenix: bool


@dataclass(frozen=True)
class PlayFullHouse:
    triple_rank: int
    pair_rank: int
    phoenix_position: str  # "none" | "triple" | "pair"


@dataclass(frozen=True)
class PlayPairStep:
    start_rank: int
    length: int  # number of pairs; the played count is 2 * length
    phoenix_position: int | None  # which pair index 0..length-1 contains the phoenix


@dataclass(frozen=True)
class PlayStraight:
    start_rank: int  # 1 if mahjong-led, else 2..10
    length: int      # 5..14
    phoenix_position: int | None  # offset 0..length-1; for start_rank==1, cannot be 0


@dataclass(frozen=True)
class PlayFourBomb:
    rank: int


@dataclass(frozen=True)
class PlayStraightFlushBomb:
    suit: str
    start_rank: int
    length: int


@dataclass(frozen=True)
class Pass:
    pass


@dataclass(frozen=True)
class CallTichu:
    pass


@dataclass(frozen=True)
class CallGrandTichu:
    pass


@dataclass(frozen=True)
class SchupfenDirection:
    direction: str  # "next" | "partner" | "previous"


@dataclass(frozen=True)
class WishRank:
    rank: int | None  # 2..14 or None for "no wish"


@dataclass(frozen=True)
class DragonGive:
    side: str  # "left" | "right"


Action = Union[
    PlaySingle, PlayPair, PlayTriple, PlayFullHouse, PlayPairStep,
    PlayStraight, PlayFourBomb, PlayStraightFlushBomb,
    Pass, CallTichu, CallGrandTichu, SchupfenDirection, WishRank, DragonGive,
]


# ---------------------------------------------------------------------------
# Enumeration.
# ---------------------------------------------------------------------------


_SUITS = ("jade", "sword", "pagoda", "star")  # deterministic order
_SPECIALS = ("mahjong", "dog", "dragon", "phoenix")


def _enumerate_singles() -> list[Action]:
    out: list[Action] = []
    for rank in range(2, 15):
        for suit in _SUITS:
            out.append(PlaySingle(suit=suit, rank=rank))
    for special in _SPECIALS:
        out.append(PlaySingle(special=special))
    for r in range(2, 15):
        out.append(PlaySingle(phoenix_as_rank=r))
    return out


def _enumerate_pairs() -> list[Action]:
    return [PlayPair(rank=r, with_phoenix=ph) for r in range(2, 15) for ph in (False, True)]


def _enumerate_triples() -> list[Action]:
    return [PlayTriple(rank=r, with_phoenix=ph) for r in range(2, 15) for ph in (False, True)]


def _enumerate_full_houses() -> list[Action]:
    out: list[Action] = []
    for tr in range(2, 15):
        for pr in range(2, 15):
            if tr == pr:
                continue
            for pos in ("none", "triple", "pair"):
                out.append(PlayFullHouse(triple_rank=tr, pair_rank=pr, phoenix_position=pos))
    return out


def _enumerate_pair_steps() -> list[Action]:
    out: list[Action] = []
    for length in range(2, 14):  # 2..13 pair-steps; 13-pair-step spans ranks 2..14
        for start in range(2, 14 - length + 2):
            if start + length - 1 > 14:
                continue
            out.append(PlayPairStep(start_rank=start, length=length, phoenix_position=None))
            for pos in range(length):
                out.append(PlayPairStep(start_rank=start, length=length, phoenix_position=pos))
    return out


def _enumerate_straights() -> list[Action]:
    out: list[Action] = []
    for start in range(1, 11):       # 1..10 (1 = mahjong-led)
        for length in range(5, 15):
            if start + length - 1 > 14:
                continue
            # Natural (no phoenix).
            out.append(PlayStraight(start_rank=start, length=length, phoenix_position=None))
            # Phoenix substitution: any slot except slot 0 when mahjong-led.
            for pos in range(length):
                if start == 1 and pos == 0:
                    continue
                out.append(PlayStraight(start_rank=start, length=length, phoenix_position=pos))
    return out


def _enumerate_four_bombs() -> list[Action]:
    return [PlayFourBomb(rank=r) for r in range(2, 15)]


def _enumerate_straight_flush_bombs() -> list[Action]:
    out: list[Action] = []
    for suit in _SUITS:
        for start in range(2, 11):
            for length in range(5, 15):
                if start + length - 1 > 14:
                    continue
                out.append(PlayStraightFlushBomb(suit=suit, start_rank=start, length=length))
    return out


def _enumerate_non_play_actions() -> list[Action]:
    out: list[Action] = []
    out.append(Pass())
    out.append(CallTichu())
    out.append(CallGrandTichu())
    for direction in ("next", "partner", "previous"):
        out.append(SchupfenDirection(direction=direction))
    out.append(WishRank(rank=None))
    for r in range(2, 15):
        out.append(WishRank(rank=r))
    for side in ("left", "right"):
        out.append(DragonGive(side=side))
    return out


def _build_canonical() -> tuple[Action, ...]:
    sections: list[tuple[str, list[Action]]] = [
        ("singles", _enumerate_singles()),
        ("pairs", _enumerate_pairs()),
        ("triples", _enumerate_triples()),
        ("full_houses", _enumerate_full_houses()),
        ("pair_steps", _enumerate_pair_steps()),
        ("straights", _enumerate_straights()),
        ("four_bombs", _enumerate_four_bombs()),
        ("sf_bombs", _enumerate_straight_flush_bombs()),
        ("non_play", _enumerate_non_play_actions()),
    ]
    all_actions: list[Action] = []
    for _, items in sections:
        all_actions.extend(items)
    _SECTION_SIZES.update({name: len(items) for name, items in sections})
    return tuple(all_actions)


_SECTION_SIZES: dict[str, int] = {}
CANONICAL_ACTIONS: tuple[Action, ...] = _build_canonical()
ACTION_SPACE_SIZE: int = len(CANONICAL_ACTIONS)

_ACTION_TO_INDEX: dict[Action, int] = {a: i for i, a in enumerate(CANONICAL_ACTIONS)}


def encode(action: Action) -> int:
    """Return the canonical index for an action. KeyError if not in the space."""
    return _ACTION_TO_INDEX[action]


def decode(index: int) -> Action:
    """Return the action at the given canonical index. IndexError if out of range."""
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise IndexError(f"action index {index} out of range [0, {ACTION_SPACE_SIZE})")
    return CANONICAL_ACTIONS[index]


# Import-time log of the section breakdown and total.
_log = logging.getLogger(__name__)
_log.info(
    "action space %s: %d entries (%s)",
    ACTION_SPACE_VERSION,
    ACTION_SPACE_SIZE,
    ", ".join(f"{name}={n}" for name, n in _SECTION_SIZES.items()),
)
