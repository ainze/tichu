"""Featurizer v1: PrivateState → fixed-shape float32 ndarray.

Pure function. No I/O, no globals, no hash-seed-sensitive ordering. Output
shape is `(FEATURIZER_OUTPUT_DIM,)` for every legal PrivateState (including
schupfen and dragon-give-pending phases).

Section layout is documented in `SECTION_DIMS`. The trick-top-combo and
play-history sections are one-hot over the v1 action space.

Notes / known v1 simplifications:
  - "Phoenix played" reflects whether Phoenix is visible in the *current
    trick*; we have no PrivateState memory of earlier tricks in the round.
  - "Schupfen received" is populated only while `SchupfenPending` is active;
    once applied, the visible state no longer retains the per-direction
    breakdown, so the section is zeros outside the pending phase. Either of
    these can be enriched in a v2 by augmenting the state schema.
"""

from typing import Iterable

import numpy as np

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, SpecialCard, Suit
from tichu_engine.combinations import (
    FourOfAKindBomb,
    FullHouse,
    Pair,
    PairStep,
    Single,
    Straight,
    StraightFlushBomb,
    Triple,
)
from tichu_engine.state import (
    DragonGivePending,
    MahjongWishPending,
    PrivateState,
    PublicState,
    SchupfenPending,
)
from tichu_training.action_space import (
    ACTION_SPACE_SIZE,
    PlayFourBomb,
    PlayFullHouse,
    PlayPair,
    PlayPairStep,
    PlaySingle,
    PlayStraight,
    PlayStraightFlushBomb,
    PlayTriple,
    encode,
)


FEATURIZER_VERSION: str = "v1"

# Section sizes — exact dims, ordered for the output concatenation.
SECTION_DIMS: dict[str, int] = {
    "own_hand": 56,
    "hand_sizes": 4,
    "team_scores": 2,
    "round_points": 4,
    "out_order": 16,
    "tichu_callers": 4,
    "grand_tichu_callers": 4,
    "mahjong_wish": 15,
    "current_player": 4,
    "phase": 5,
    "trick_top_combo": ACTION_SPACE_SIZE,
    "trick_passes": 4,
    "play_history": 8 * ACTION_SPACE_SIZE,
    "schupfen_received": 168,
    "phoenix_played": 1,
}
FEATURIZER_OUTPUT_DIM: int = sum(SECTION_DIMS.values())


# ---------------------------------------------------------------------------
# Card slot mapping (stable across processes).
# ---------------------------------------------------------------------------


_SUIT_ORDER = (Suit.JADE, Suit.SWORD, Suit.PAGODA, Suit.STAR)
_NATURAL_SLOT_BY_CARD: dict[Card, int] = {}
for _suit_idx, _suit in enumerate(_SUIT_ORDER):
    for _rank in range(2, 15):
        _slot = _suit_idx * 13 + (_rank - 2)
        _NATURAL_SLOT_BY_CARD[Card(_suit, _rank)] = _slot

# Special-card slots come after the 52 naturals, in a deterministic order.
_SPECIAL_SLOT: dict[SpecialCard, int] = {
    MAHJONG: 52,
    DOG: 53,
    PHOENIX: 54,
    DRAGON: 55,
}


def _card_slot(card: object) -> int:
    if isinstance(card, Card):
        return _NATURAL_SLOT_BY_CARD[card]
    if isinstance(card, SpecialCard):
        return _SPECIAL_SLOT[card]
    raise TypeError(f"unsupported card type: {type(card).__name__}")


# ---------------------------------------------------------------------------
# Combination → action-space index.
# ---------------------------------------------------------------------------


_SUIT_NAME = {
    Suit.JADE: "jade",
    Suit.SWORD: "sword",
    Suit.PAGODA: "pagoda",
    Suit.STAR: "star",
}


def _combo_cards(combo: object) -> tuple:
    if isinstance(combo, Single):
        return (combo.card,)
    if isinstance(combo, Pair):
        return (combo.a, combo.b)
    if isinstance(combo, Triple):
        return (combo.a, combo.b, combo.c)
    if isinstance(combo, FullHouse):
        return _combo_cards(combo.triple) + _combo_cards(combo.pair)
    if isinstance(combo, PairStep):
        out: list = []
        for p in combo.pairs:
            out.extend(_combo_cards(p))
        return tuple(out)
    if isinstance(combo, (Straight, StraightFlushBomb)):
        return tuple(combo.cards)
    if isinstance(combo, FourOfAKindBomb):
        return (combo.a, combo.b, combo.c, combo.d)
    return ()


def _has_phoenix(cards: Iterable[object]) -> bool:
    return any(c is PHOENIX for c in cards)


def _combination_to_action_index(combo: object) -> int | None:
    """Map an engine combination to the canonical action-space index.

    Returns None if the combination cannot be represented (e.g. unknown type
    or a degenerate case the v1 action space does not enumerate).
    """
    if combo is None:
        return None
    if isinstance(combo, Single):
        card = combo.card
        if isinstance(card, Card):
            action = PlaySingle(suit=_SUIT_NAME[card.suit], rank=card.rank)
        elif card is PHOENIX:
            if combo.as_rank is not None:
                action = PlaySingle(phoenix_as_rank=int(combo.as_rank))
            else:
                action = PlaySingle(special="phoenix")
        elif card is MAHJONG:
            action = PlaySingle(special="mahjong")
        elif card is DOG:
            action = PlaySingle(special="dog")
        elif card is DRAGON:
            action = PlaySingle(special="dragon")
        else:
            return None
        return _lookup(action)
    if isinstance(combo, Pair):
        return _lookup(PlayPair(rank=combo.rank, with_phoenix=_has_phoenix(_combo_cards(combo))))
    if isinstance(combo, Triple):
        return _lookup(PlayTriple(rank=combo.rank, with_phoenix=_has_phoenix(_combo_cards(combo))))
    if isinstance(combo, FullHouse):
        triple_has_phx = _has_phoenix(_combo_cards(combo.triple))
        pair_has_phx = _has_phoenix(_combo_cards(combo.pair))
        if triple_has_phx and pair_has_phx:
            return None
        pos = "triple" if triple_has_phx else "pair" if pair_has_phx else "none"
        return _lookup(PlayFullHouse(triple_rank=combo.triple.rank, pair_rank=combo.pair.rank, phoenix_position=pos))
    if isinstance(combo, PairStep):
        pairs = combo.pairs
        phx_pos = next(
            (i for i, p in enumerate(pairs) if _has_phoenix(_combo_cards(p))),
            None,
        )
        return _lookup(PlayPairStep(start_rank=pairs[0].rank, length=len(pairs), phoenix_position=phx_pos))
    if isinstance(combo, Straight):
        cards = combo.cards
        has_mj = any(c is MAHJONG for c in cards)
        start = 1 if has_mj else combo.rank
        phx_pos = combo.phoenix_as_rank - start if combo.phoenix_as_rank is not None else None
        if phx_pos is not None and not 0 <= phx_pos < len(cards):
            phx_pos = None
        return _lookup(PlayStraight(start_rank=start, length=len(cards), phoenix_position=phx_pos))
    if isinstance(combo, FourOfAKindBomb):
        return _lookup(PlayFourBomb(rank=combo.rank))
    if isinstance(combo, StraightFlushBomb):
        suit_name = _SUIT_NAME[combo.cards[0].suit]
        return _lookup(PlayStraightFlushBomb(suit=suit_name, start_rank=combo.cards[0].rank, length=combo.length))
    return None


def _lookup(action) -> int | None:
    try:
        return encode(action)
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# featurize
# ---------------------------------------------------------------------------


def featurize(private_state: PrivateState) -> np.ndarray:
    out = np.zeros(FEATURIZER_OUTPUT_DIM, dtype=np.float32)
    cursor = 0
    pub: PublicState = private_state.public

    # 1. Own hand (56 dims).
    for card in private_state.hand:
        out[cursor + _card_slot(card)] = 1.0
    cursor += SECTION_DIMS["own_hand"]

    # 2. Hand sizes / 14.
    for i in range(4):
        out[cursor + i] = pub.hand_sizes[i] / 14.0
    cursor += SECTION_DIMS["hand_sizes"]

    # 3. Team scores / 1000.
    out[cursor] = pub.scores[0] / 1000.0
    out[cursor + 1] = pub.scores[1] / 1000.0
    cursor += SECTION_DIMS["team_scores"]

    # 4. Round points / 100.
    for i in range(4):
        out[cursor + i] = pub.round_points_by_player[i] / 100.0
    cursor += SECTION_DIMS["round_points"]

    # 5. Out-order one-hot: 4 slots × 4 seats. Each slot encodes who finished in that position.
    for i in range(4):
        if i < len(pub.out_order):
            seat = pub.out_order[i]
            if 0 <= seat < 4:
                out[cursor + i * 4 + seat] = 1.0
    cursor += SECTION_DIMS["out_order"]

    # 6. Tichu callers.
    for seat in pub.tichu_callers:
        if 0 <= seat < 4:
            out[cursor + seat] = 1.0
    cursor += SECTION_DIMS["tichu_callers"]

    # 7. Grand Tichu callers.
    for seat in pub.grand_tichu_callers:
        if 0 <= seat < 4:
            out[cursor + seat] = 1.0
    cursor += SECTION_DIMS["grand_tichu_callers"]

    # 8. Mahjong wish: ranks 2..14 = 13 dims, slot 13 = "no wish active", slot 14 = "wish fulfilled / N/A".
    wish = pub.mahjong_wish
    if wish is None:
        out[cursor + 13] = 1.0
    elif 2 <= wish <= 14:
        out[cursor + (wish - 2)] = 1.0
    else:
        out[cursor + 14] = 1.0
    cursor += SECTION_DIMS["mahjong_wish"]

    # 9. Current player one-hot.
    if 0 <= pub.current_player < 4:
        out[cursor + pub.current_player] = 1.0
    cursor += SECTION_DIMS["current_player"]

    # 10. Phase one-hot.
    phase_idx = _phase_index(pub)
    out[cursor + phase_idx] = 1.0
    cursor += SECTION_DIMS["phase"]

    # 11. Trick top combo one-hot over action space.
    top = pub.trick.top_combination
    if top is not None:
        idx = _combination_to_action_index(top)
        if idx is not None:
            out[cursor + idx] = 1.0
    cursor += SECTION_DIMS["trick_top_combo"]

    # 12. Trick passes one-hot.
    for seat in pub.trick.passes:
        if 0 <= seat < 4:
            out[cursor + seat] = 1.0
    cursor += SECTION_DIMS["trick_passes"]

    # 13. Play history (last 8 plays in the current trick, oldest→newest, right-aligned).
    plays = pub.trick.plays[-8:]
    pad = 8 - len(plays)
    for i, play in enumerate(plays):
        idx = _combination_to_action_index(play.combination)
        if idx is not None:
            slot = (pad + i) * ACTION_SPACE_SIZE + idx
            out[cursor + slot] = 1.0
    cursor += SECTION_DIMS["play_history"]

    # 14. Schupfen received (3 × 56) — only meaningful while SchupfenPending.
    pending = pub.pending_decision
    if isinstance(pending, SchupfenPending):
        # Submissions to *this* player from the other three seats.
        my_seat = private_state.player
        # Direction labels are relative to the giver. For receiver `my_seat`:
        #   from previous-seat: giver = (my_seat - 1) % 4, gave-to-next is index 0 of giver's submission
        #   from partner-seat:  giver = (my_seat + 2) % 4, gave-to-partner is index 1
        #   from next-seat:     giver = (my_seat + 1) % 4, gave-to-previous is index 2
        slots = [
            ((my_seat - 1) % 4, 0),
            ((my_seat + 2) % 4, 1),
            ((my_seat + 1) % 4, 2),
        ]
        for direction_idx, (giver, sub_idx) in enumerate(slots):
            submission = pending.submitted[giver] if giver < len(pending.submitted) else None
            if submission is None:
                continue
            try:
                card = submission[sub_idx]
            except (TypeError, IndexError):
                continue
            if card is None:
                continue
            out[cursor + direction_idx * 56 + _card_slot(card)] = 1.0
    cursor += SECTION_DIMS["schupfen_received"]

    # 15. Phoenix-played flag — true iff Phoenix is visible in the current trick.
    phoenix_in_trick = any(
        any(c is PHOENIX for c in _combo_cards(play.combination))
        for play in pub.trick.plays
    )
    out[cursor] = 1.0 if phoenix_in_trick else 0.0
    cursor += SECTION_DIMS["phoenix_played"]

    assert cursor == FEATURIZER_OUTPUT_DIM
    return out


def _phase_index(public: PublicState) -> int:
    pending = public.pending_decision
    if pending is None:
        return 0  # normal play
    if isinstance(pending, SchupfenPending):
        return 1
    if isinstance(pending, DragonGivePending):
        return 2
    if isinstance(pending, MahjongWishPending):
        return 3
    return 4  # terminal / unknown
