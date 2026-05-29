"""Featurizer v3: PrivateState → fixed-shape float32 ndarray.

Pure function. No I/O, no globals, no hash-seed-sensitive ordering. Output
shape is `(FEATURIZER_OUTPUT_DIM,)` for every legal PrivateState (including
pending-decision phases).

Section layout is documented in `SECTION_DIMS`. The trick-top-combo section
is one-hot over the v1 action space.

v3 over v2: drops three sections totalling 14,641 dims (~88% of v2).
See [ADR-0015](../../docs/adr/0015-featurizer-v3-drop-leaky-and-redundant-sections.md)
for the rationale. Summary:

  - `play_history` (14,472 dims, 87% of v2): dropped. Encoded the last
    8 plays of the *current trick* one-hot over ACTION_SPACE_SIZE. The
    last play is already in `trick_top_combo`
    (`trick.top_combination == trick.plays[-1].combination`), so the
    only marginal information was within-trick sequence — not worth
    87% of the corpus footprint.
  - `schupfen_received` (168 dims): dropped — **rules violation**. The
    v2 encoding gave per-giver attribution of received cards (3
    direction slots × 56 cards), which a rule-abiding player does not
    know. Training on this signal made the deployed model
    structurally dependent on information unavailable at honest play.
  - `phoenix_played` (1 dim): dropped — redundant with `seen_cards`.

Net dim: 16,624 → 1,983. Per-row at f32: 66.5 KB → 7.7 KB (8.4×
smaller bundle). v2 checkpoints are not loadable against v3 — the
version pin catches this.
"""

import numpy as np

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, SpecialCard, Suit
from tichu_engine.state import (
    DragonGivePending,
    MahjongWishPending,
    PrivateState,
    PublicState,
    SchupfenPending,
)
from tichu_training.action_space import (
    ACTION_SPACE_SIZE,
    play_intent_index,
)


FEATURIZER_VERSION: str = "v3"

# Section sizes — exact dims, ordered for the output concatenation.
# v3: play_history / schupfen_received / phoenix_played removed.
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
    "seen_cards": 56,
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


def _combination_to_action_index(combo: object) -> int | None:
    """Map an engine combination to the canonical action-space index.

    Thin null-tolerant wrapper over ``play_intent_index`` (the canonical
    concrete→Intent resolver in ``action_space``). Returns None for cases
    the v1 action space cannot represent (e.g. FullHouse with Phoenix in
    both triple and pair) so the featurizer can zero the slot rather than
    crash.
    """
    if combo is None:
        return None
    try:
        return play_intent_index(combo)
    except (KeyError, ValueError):
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

    # 13. Seen-cards multi-hot over 56 slots — every card played so far this
    # round, accumulated by the engine in `played_cards_this_round` and reset
    # at round boundaries by `_finalise_round`.
    for card in pub.played_cards_this_round:
        out[cursor + _card_slot(card)] = 1.0
    cursor += SECTION_DIMS["seen_cards"]

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
