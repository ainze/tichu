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
# Intent sum type.
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


Intent = Union[
    PlaySingle, PlayPair, PlayTriple, PlayFullHouse, PlayPairStep,
    PlayStraight, PlayFourBomb, PlayStraightFlushBomb,
    Pass, CallTichu, CallGrandTichu, SchupfenDirection, WishRank, DragonGive,
]


# ---------------------------------------------------------------------------
# Enumeration.
# ---------------------------------------------------------------------------


_SUITS = ("jade", "sword", "pagoda", "star")  # deterministic order
_SPECIALS = ("mahjong", "dog", "dragon", "phoenix")


def _enumerate_singles() -> list[Intent]:
    out: list[Intent] = []
    for rank in range(2, 15):
        for suit in _SUITS:
            out.append(PlaySingle(suit=suit, rank=rank))
    for special in _SPECIALS:
        out.append(PlaySingle(special=special))
    for r in range(2, 15):
        out.append(PlaySingle(phoenix_as_rank=r))
    return out


def _enumerate_pairs() -> list[Intent]:
    return [PlayPair(rank=r, with_phoenix=ph) for r in range(2, 15) for ph in (False, True)]


def _enumerate_triples() -> list[Intent]:
    return [PlayTriple(rank=r, with_phoenix=ph) for r in range(2, 15) for ph in (False, True)]


def _enumerate_full_houses() -> list[Intent]:
    out: list[Intent] = []
    for tr in range(2, 15):
        for pr in range(2, 15):
            if tr == pr:
                continue
            for pos in ("none", "triple", "pair"):
                out.append(PlayFullHouse(triple_rank=tr, pair_rank=pr, phoenix_position=pos))
    return out


def _enumerate_pair_steps() -> list[Intent]:
    out: list[Intent] = []
    for length in range(2, 14):  # 2..13 pair-steps; 13-pair-step spans ranks 2..14
        for start in range(2, 14 - length + 2):
            if start + length - 1 > 14:
                continue
            out.append(PlayPairStep(start_rank=start, length=length, phoenix_position=None))
            for pos in range(length):
                out.append(PlayPairStep(start_rank=start, length=length, phoenix_position=pos))
    return out


def _enumerate_straights() -> list[Intent]:
    out: list[Intent] = []
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


def _enumerate_four_bombs() -> list[Intent]:
    return [PlayFourBomb(rank=r) for r in range(2, 15)]


def _enumerate_straight_flush_bombs() -> list[Intent]:
    out: list[Intent] = []
    for suit in _SUITS:
        for start in range(2, 11):
            for length in range(5, 15):
                if start + length - 1 > 14:
                    continue
                out.append(PlayStraightFlushBomb(suit=suit, start_rank=start, length=length))
    return out


def _enumerate_non_play_actions() -> list[Intent]:
    out: list[Intent] = []
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


def _build_canonical() -> tuple[Intent, ...]:
    sections: list[tuple[str, list[Intent]]] = [
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
    all_actions: list[Intent] = []
    for _, items in sections:
        all_actions.extend(items)
    _SECTION_SIZES.update({name: len(items) for name, items in sections})
    return tuple(all_actions)


_SECTION_SIZES: dict[str, int] = {}
CANONICAL_ACTIONS: tuple[Intent, ...] = _build_canonical()
ACTION_SPACE_SIZE: int = len(CANONICAL_ACTIONS)

_ACTION_TO_INDEX: dict[Intent, int] = {a: i for i, a in enumerate(CANONICAL_ACTIONS)}


def encode(action: Intent) -> int:
    """Return the canonical index for an action. KeyError if not in the space."""
    return _ACTION_TO_INDEX[action]


def decode(index: int) -> Intent:
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


# ---------------------------------------------------------------------------
# Concrete → Intent inverse resolver (BC training-time use).
#
# The forward Resolver (Intent → ConcreteAction) is heuristic and lives in
# tichu_inference/ml_agent.py. The inverse direction here is mechanical:
# given a ConcreteAction the engine produced (or a parsed BSW action), map
# it back to the canonical Intent index for the relevant BC head.
#
# Heads have different output dims (see HEAD_LOGIT_DIMS):
#   play              → 1809 (full Action Space)
#   wish              → 14   (None + 13 ranks 2..14)
#   dragon_assignment → 2    (left, right relative to winner)
#
# See [ADR-0011](../../docs/adr/0011-bc-training-replay-on-the-fly.md).
# ---------------------------------------------------------------------------


# Local indices for the small heads. Order pins what BC targets in [0,K) mean.
_WISH_INDEX_BY_RANK: dict[int | None, int] = {None: 0, **{r: r - 1 for r in range(2, 15)}}
_DRAGON_INDEX_BY_SIDE: dict[str, int] = {"left": 0, "right": 1}


# Suit-enum-to-name and small helpers, used by the play-Intent resolver below.
# Kept here (not in featurizer) so the action_space module remains the single
# source of truth for the concrete→Intent mapping per ADR-0011.

def _intent_combo_cards(combo: object) -> tuple:
    """All constituent cards in an engine Combination, flattened. Mirrors
    ``tichu_training.featurizer._combo_cards`` so action_space stays the
    canonical home for concrete→Intent logic."""
    # Imports are deferred to avoid a circular module load: featurizer
    # imports from action_space at module top-level.
    from tichu_engine.cards import Suit  # noqa: F401  (kept for type-check eyes)
    from tichu_engine.combinations import (
        FourOfAKindBomb, FullHouse, Pair, PairStep, Single, Straight,
        StraightFlushBomb, Triple,
    )
    if isinstance(combo, Single):
        return (combo.card,)
    if isinstance(combo, Pair):
        return (combo.a, combo.b)
    if isinstance(combo, Triple):
        return (combo.a, combo.b, combo.c)
    if isinstance(combo, FullHouse):
        return _intent_combo_cards(combo.triple) + _intent_combo_cards(combo.pair)
    if isinstance(combo, PairStep):
        out: list = []
        for p in combo.pairs:
            out.extend(_intent_combo_cards(p))
        return tuple(out)
    if isinstance(combo, (Straight, StraightFlushBomb)):
        return tuple(combo.cards)
    if isinstance(combo, FourOfAKindBomb):
        return (combo.a, combo.b, combo.c, combo.d)
    return ()


def _intent_has_phoenix(cards) -> bool:
    from tichu_engine.cards import PHOENIX
    return any(c is PHOENIX for c in cards)


def _intent_suit_name(suit_enum) -> str:
    from tichu_engine.cards import Suit
    return {Suit.JADE: "jade", Suit.SWORD: "sword",
            Suit.PAGODA: "pagoda", Suit.STAR: "star"}[suit_enum]


def play_intent_index(combo_or_pass) -> int:
    """Map an engine play action (Combination, Pass, BombInterrupt) to its
    play-head index in [0, 1809).

    Raises ``ValueError`` for shapes the v1 Action Space cannot represent
    — uniformly, so callers can ``except ValueError: continue`` to skip
    unrepresentable actions. Concretely:

    * FullHouse with Phoenix in both triple and pair (legal in engine,
      not enumerated in v1 action space).
    * Phoenix-following Mahjong (engine ``as_rank=1.5`` → no
      ``PlaySingle(phoenix_as_rank=1)`` in the v1 catalogue).
    * Straight starting at rank 1 with Phoenix substituting at slot 0
      (Mahjong owns slot 0 in a mahjong-led straight; the v1 catalogue
      excludes ``PlayStraight(start_rank=1, phoenix_position=0)``).
    * Any other intent the v1 catalogue does not list.
    """
    # Deferred imports to avoid the cycle with featurizer/engine.
    from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX
    from tichu_engine.combinations import (
        FourOfAKindBomb, FullHouse, Pair, PairStep, Single, Straight,
        StraightFlushBomb, Triple,
    )
    from tichu_engine.legality import BombInterrupt, Pass as _EnginePass

    # Pass → the single Pass intent in the action space.
    if isinstance(combo_or_pass, _EnginePass):
        return _encode_or_value_error(Pass())

    # Bomb interrupts carry the underlying bomb; unwrap and recurse.
    if isinstance(combo_or_pass, BombInterrupt):
        return play_intent_index(combo_or_pass.bomb)

    combo = combo_or_pass
    if isinstance(combo, Single):
        card = combo.card
        if isinstance(card, Card):
            return _encode_or_value_error(
                PlaySingle(suit=_intent_suit_name(card.suit), rank=card.rank),
            )
        if card is PHOENIX:
            if combo.as_rank is not None:
                return _encode_or_value_error(
                    PlaySingle(phoenix_as_rank=int(combo.as_rank)),
                )
            return _encode_or_value_error(PlaySingle(special="phoenix"))
        if card is MAHJONG:
            return _encode_or_value_error(PlaySingle(special="mahjong"))
        if card is DOG:
            return _encode_or_value_error(PlaySingle(special="dog"))
        if card is DRAGON:
            return _encode_or_value_error(PlaySingle(special="dragon"))
        raise ValueError(f"Single with unrecognised card: {card!r}")
    if isinstance(combo, Pair):
        return _encode_or_value_error(PlayPair(
            rank=combo.rank, with_phoenix=_intent_has_phoenix(_intent_combo_cards(combo)),
        ))
    if isinstance(combo, Triple):
        return _encode_or_value_error(PlayTriple(
            rank=combo.rank, with_phoenix=_intent_has_phoenix(_intent_combo_cards(combo)),
        ))
    if isinstance(combo, FullHouse):
        triple_has_phx = _intent_has_phoenix(_intent_combo_cards(combo.triple))
        pair_has_phx = _intent_has_phoenix(_intent_combo_cards(combo.pair))
        if triple_has_phx and pair_has_phx:
            raise ValueError("FullHouse with Phoenix in both triple and pair is not in the v1 Action Space")
        pos = "triple" if triple_has_phx else ("pair" if pair_has_phx else "none")
        return _encode_or_value_error(PlayFullHouse(
            triple_rank=combo.triple.rank, pair_rank=combo.pair.rank, phoenix_position=pos,
        ))
    if isinstance(combo, PairStep):
        pairs = combo.pairs
        phx_pos = next(
            (i for i, p in enumerate(pairs) if _intent_has_phoenix(_intent_combo_cards(p))),
            None,
        )
        return _encode_or_value_error(PlayPairStep(
            start_rank=pairs[0].rank, length=len(pairs), phoenix_position=phx_pos,
        ))
    if isinstance(combo, Straight):
        cards = combo.cards
        has_mj = any(c is MAHJONG for c in cards)
        start = 1 if has_mj else combo.rank
        phx_pos = combo.phoenix_as_rank - start if combo.phoenix_as_rank is not None else None
        if phx_pos is not None and not 0 <= phx_pos < len(cards):
            phx_pos = None
        return _encode_or_value_error(PlayStraight(
            start_rank=start, length=len(cards), phoenix_position=phx_pos,
        ))
    if isinstance(combo, FourOfAKindBomb):
        return _encode_or_value_error(PlayFourBomb(rank=combo.rank))
    if isinstance(combo, StraightFlushBomb):
        return _encode_or_value_error(PlayStraightFlushBomb(
            suit=_intent_suit_name(combo.cards[0].suit),
            start_rank=combo.cards[0].rank, length=combo.length,
        ))
    raise ValueError(f"unknown engine action type: {type(combo_or_pass).__name__}")


def _encode_or_value_error(intent) -> int:
    """Wrap ``encode`` so callers see a uniform ``ValueError`` for any intent
    not enumerated in v1. The catalogue is the canonical source of truth;
    if a shape isn't in it, it isn't representable, regardless of why."""
    try:
        return encode(intent)
    except KeyError as exc:
        raise ValueError(
            f"intent {intent!r} is not in the v1 Action Space"
        ) from exc


def wish_intent_index(mahjong_wish) -> int:
    """Map a `MahjongWish(rank: int | None)` to its wish-head index in [0, 14).

    Index 0 = no-wish; indices 1..13 = ranks 2..14.
    """
    return _WISH_INDEX_BY_RANK[mahjong_wish.rank]


def dragon_intent_index(dragon_give, winner_seat: int) -> int:
    """Map a `DragonGive(target: int)` to its dragon_assignment head index
    in {0, 1}.

    Side convention (per replay.py): left opponent of winner W = (W+3) % 4;
    right opponent = (W+1) % 4. Index 0 = left, 1 = right.
    """
    target = dragon_give.target
    if target == (winner_seat + 3) % 4:
        return _DRAGON_INDEX_BY_SIDE["left"]
    if target == (winner_seat + 1) % 4:
        return _DRAGON_INDEX_BY_SIDE["right"]
    raise ValueError(
        f"DragonGive(target={target}) is not an opponent of winner seat {winner_seat}"
    )


def legal_mask(
    decision_type: str,
    game_state,
    player: int,
    *,
    cached_actions=None,
):
    """Legal-action mask for a BC head, per ADR-0011 §3.

    Returns a 1-D bool ndarray of length K = HEAD_LOGIT_DIMS[decision_type]:
      play              → 1809   (over the full Action Space)
      wish              → 14     (None + 13 ranks 2..14)
      dragon_assignment → 2      (left, right relative to winner)

    For `play`, the mask is the union of:
      - When `player` is the current player: every action in
        ``legal_actions(state)``, mapped through ``play_intent_index``.
      - When `player` is NOT the current player: every bomb interrupt in
        ``legal_bomb_interrupts(state, player)``. This is the small set
        the BSW logs record when a non-current player drops a bomb to
        seize the lead mid-trick.

    For `wish` and `dragon_assignment`, every output slot is legal at the
    moment the engine is sitting in the corresponding pending decision —
    a Mahjong-wisher may declare any rank (or decline); a Dragon-trick
    winner may give to either opponent.

    `cached_actions`: optional pre-computed `legal_actions(game_state)`
    frozenset. When supplied AND the actor is the current player, the
    cached set is used instead of re-enumerating — this is the hot path
    when ParquetBCDataset feeds the cache it got back from
    ``replay_round`` (ADR-0011 perf: avoid the duplicate enumeration that
    otherwise happens once during replay and once again in legal_mask).

    Imports are deferred to avoid an action_space → tichu_engine cycle at
    module import time.
    """
    import numpy as np
    from tichu_engine.legality import legal_actions, legal_bomb_interrupts
    from tichu_training.bc.heads import HEAD_LOGIT_DIMS

    if decision_type not in {"play", "wish", "dragon_assignment"}:
        raise ValueError(
            f"unsupported decision_type for legal_mask: {decision_type!r} "
            "(expected one of play, wish, dragon_assignment)"
        )
    K = HEAD_LOGIT_DIMS[decision_type]
    mask = np.zeros(K, dtype=bool)

    if decision_type == "play":
        if player == game_state.public.current_player:
            actions = cached_actions if cached_actions is not None else legal_actions(game_state)
            for action in actions:
                try:
                    idx = play_intent_index(action)
                except ValueError:
                    # Engine produced an action shape the v1 Action Space
                    # cannot represent (e.g. FullHouse with Phoenix in
                    # both triple and pair, or Phoenix-following-Mahjong).
                    # Skip — the mask just doesn't flag that slot legal;
                    # the trainer won't pick it.
                    continue
                if 0 <= idx < K:
                    mask[idx] = True
        else:
            # Bomb interrupts: a different legality universe; the cached
            # `legal_actions(state)` doesn't apply.
            for bomb in legal_bomb_interrupts(game_state, player):
                try:
                    idx = play_intent_index(bomb)
                except ValueError:
                    continue
                if 0 <= idx < K:
                    mask[idx] = True
        return mask

    if decision_type == "wish":
        # Every wish slot is legal during MahjongWishPending — no per-rank
        # legality constraints in v1.
        mask[:] = True
        return mask

    if decision_type == "dragon_assignment":
        # Both opponents are always legal during DragonGivePending.
        mask[:] = True
        return mask

    raise ValueError(
        f"unsupported decision_type for legal_mask: {decision_type!r} "
        "(expected one of play, wish, dragon_assignment)"
    )


def bc_target_for_concrete(
    decision_type: str,
    concrete_action,
    *,
    winner_seat: int | None = None,
) -> int:
    """Dispatch helper used by ParquetBCDataset to produce the per-decision
    target int for a BCExample. See ADR-0011 §3 for the API shape.

    `winner_seat` is required only for `decision_type == "dragon_assignment"`.
    """
    if decision_type == "play":
        return play_intent_index(concrete_action)
    if decision_type == "wish":
        return wish_intent_index(concrete_action)
    if decision_type == "dragon_assignment":
        if winner_seat is None:
            raise ValueError("winner_seat is required for dragon_assignment targets")
        return dragon_intent_index(concrete_action, winner_seat)
    raise ValueError(
        f"unsupported decision_type for BC target: {decision_type!r} "
        "(expected one of play, wish, dragon_assignment)"
    )
