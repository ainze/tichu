"""Featurizer v4: PrivateState → fixed-shape float32 ndarray.

Pure function. No I/O, no globals, no hash-seed-sensitive ordering. Output
shape is `(FEATURIZER_OUTPUT_DIM,)` for every legal PrivateState (including
pending-decision phases).

Section layout is documented in `SECTION_DIMS`. The trick-top-combo section
uses a 50-dim union-of-fields layout
(`intent_kind[8] + primary_rank[15] + secondary_rank[13] + length[13]
+ phoenix_used[1]`) — see `TRICK_TOP_COMBO_SUBFIELDS` and
[ADR-0017](../../docs/adr/0017-featurizer-v4-compact-trick-top-combo.md).

v4 over v3: replaces the 1,809-dim one-hot over the v1 Action Space with
a 50-dim compact encoding. 1,223 of the 1,809 v3 slots were phoenix-
variants that each fire ~1 in tens of thousands of rows; their marginal
information vs `seen_cards[phoenix]` and `(primary_rank, length)` is
~zero. Net dim: 1,983 → 224 (−89%). Per-row at f32: 7.7 KB → 0.9 KB
(8.85× smaller bundle). v3 checkpoints are not loadable against v4 —
the version pin catches this.

v3 over v2 (kept for context): dropped `play_history` (14,472 dims),
`schupfen_received` (168 dims — rules violation, ADR-0015), and
`phoenix_played` (1 dim — redundant with `seen_cards`).
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


FEATURIZER_VERSION: str = "v4"

# v4: trick_top_combo is a 50-dim union-of-fields layout, NOT a one-hot
# over the v1 Action Space. The action-space dependency is severed on
# the input side; the play head still outputs ACTION_SPACE_SIZE logits.
# See ADR-0017.
TRICK_TOP_COMBO_SUBFIELDS: dict[str, int] = {
    "intent_kind": 8,       # Single/Pair/Triple/FullHouse/PairStep/Straight/FourBomb/SFBomb
    "primary_rank": 15,     # slot 0 = Mahjong (rank 1); 1..13 = ranks 2..14; 14 = Dragon
    "secondary_rank": 13,   # FullHouse pair_rank only; slots 0..12 = ranks 2..14
    "length": 13,           # PairStep / Straight / SFBomb; slots 0..12 = lengths 2..14
    "phoenix_used": 1,      # 1 iff Phoenix participates in the combo
}

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
    "trick_top_combo": sum(TRICK_TOP_COMBO_SUBFIELDS.values()),
    "trick_passes": 4,
    "seen_cards": 56,
}
FEATURIZER_OUTPUT_DIM: int = sum(SECTION_DIMS.values())

# Sections whose `featurize()` output is a continuous ratio (negatives
# possible) rather than a 0/1 indicator. Every *other* section is written
# as exactly 1.0 (one-hot / multi-hot), so the materialised bundle bit-packs
# all indicator columns and stores only these as float. This is the single
# source of truth for the binary/continuous split; the materialised writer
# records the resulting column list into the bundle manifest, and the reader
# reconstructs the dense (D,) vector from it. Changing the featurizer layout
# bumps FEATURIZER_VERSION, which the bundle pin already enforces. See
# [ADR-0019](../../docs/adr/0019-bit-pack-materialised-bundle.md).
CONTINUOUS_SECTIONS: frozenset[str] = frozenset(
    {"hand_sizes", "team_scores", "round_points"}
)


def _continuous_feature_columns() -> tuple[int, ...]:
    cols: list[int] = []
    cursor = 0
    for name, dim in SECTION_DIMS.items():
        if name in CONTINUOUS_SECTIONS:
            cols.extend(range(cursor, cursor + dim))
        cursor += dim
    return tuple(cols)


# Ascending column indices of the continuous (non-bit-packable) features.
# For v4 this is (56..65): hand_sizes[4] + team_scores[2] + round_points[4].
CONTINUOUS_FEATURE_COLUMNS: tuple[int, ...] = _continuous_feature_columns()

# Internal offsets inside the 50-dim trick_top_combo section.
_OFF_INTENT_KIND = 0
_OFF_PRIMARY_RANK = _OFF_INTENT_KIND + TRICK_TOP_COMBO_SUBFIELDS["intent_kind"]
_OFF_SECONDARY_RANK = _OFF_PRIMARY_RANK + TRICK_TOP_COMBO_SUBFIELDS["primary_rank"]
_OFF_LENGTH = _OFF_SECONDARY_RANK + TRICK_TOP_COMBO_SUBFIELDS["secondary_rank"]
_OFF_PHOENIX_USED = _OFF_LENGTH + TRICK_TOP_COMBO_SUBFIELDS["length"]

# intent_kind slot per combination type — matches action_space canonical order.
_KIND_SINGLE = 0
_KIND_PAIR = 1
_KIND_TRIPLE = 2
_KIND_FULL_HOUSE = 3
_KIND_PAIR_STEP = 4
_KIND_STRAIGHT = 5
_KIND_FOUR_BOMB = 6
_KIND_SF_BOMB = 7


# Card slot mapping lives in `tichu_training.card_slots` (torch-free, public)
# so the Schupfen Network's encoder and decoder both pull from one source.
# Re-export the forward direction under the historic private name so call
# sites inside this module keep their existing form.
from tichu_training.card_slots import card_slot as _card_slot  # noqa: E402


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

    # 11. Trick top combo: v4 union-of-fields layout. See ADR-0017.
    top = pub.trick.top_combination
    if top is not None:
        _emit_trick_top_combo(out, cursor, top)
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


def _emit_trick_top_combo(out: np.ndarray, cursor: int, combo: object) -> None:
    """Write the 50-dim trick_top_combo section for the given top combination.

    Caller guarantees `combo is not None`. Unrecognised shapes leave the
    section all-zero (defensive — matches v3 behaviour for shapes the
    encoder cannot represent).
    """
    # Deferred to avoid module-load cycle: tichu_engine.combinations does
    # not import featurizer, but the rest of tichu_engine pulls in modules
    # that import this one transitively when run through certain entry
    # points. Local import keeps the import graph clean.
    from tichu_engine.combinations import (
        FourOfAKindBomb, FullHouse, Pair, PairStep, Single, Straight,
        StraightFlushBomb, Triple,
    )

    if isinstance(combo, Single):
        card = combo.card
        out[cursor + _OFF_INTENT_KIND + _KIND_SINGLE] = 1.0
        # combo.rank is a float for Phoenix (1.5 lead, N+0.5 following) and
        # Dragon (25), an int for naturals (2..14) and Mahjong (1), and NaN
        # for Dog (which can never be a trick top in practice).
        r = combo.rank
        if r == 25:
            out[cursor + _OFF_PRIMARY_RANK + 14] = 1.0
        elif r == r:  # not NaN
            # int truncation collapses 1.5→1 (slot 0), N.5→N (slot N-1).
            out[cursor + _OFF_PRIMARY_RANK + (int(r) - 1)] = 1.0
        if card is PHOENIX:
            out[cursor + _OFF_PHOENIX_USED] = 1.0
        return

    if isinstance(combo, Pair):
        out[cursor + _OFF_INTENT_KIND + _KIND_PAIR] = 1.0
        out[cursor + _OFF_PRIMARY_RANK + (combo.rank - 1)] = 1.0
        if combo.a is PHOENIX or combo.b is PHOENIX:
            out[cursor + _OFF_PHOENIX_USED] = 1.0
        return

    if isinstance(combo, Triple):
        out[cursor + _OFF_INTENT_KIND + _KIND_TRIPLE] = 1.0
        out[cursor + _OFF_PRIMARY_RANK + (combo.rank - 1)] = 1.0
        if combo.a is PHOENIX or combo.b is PHOENIX or combo.c is PHOENIX:
            out[cursor + _OFF_PHOENIX_USED] = 1.0
        return

    if isinstance(combo, FullHouse):
        out[cursor + _OFF_INTENT_KIND + _KIND_FULL_HOUSE] = 1.0
        out[cursor + _OFF_PRIMARY_RANK + (combo.triple.rank - 1)] = 1.0
        # secondary_rank slot R-2 = rank R for R in 2..14.
        out[cursor + _OFF_SECONDARY_RANK + (combo.pair.rank - 2)] = 1.0
        triple_has_phx = combo.triple.a is PHOENIX or combo.triple.b is PHOENIX or combo.triple.c is PHOENIX
        pair_has_phx = combo.pair.a is PHOENIX or combo.pair.b is PHOENIX
        if triple_has_phx or pair_has_phx:
            out[cursor + _OFF_PHOENIX_USED] = 1.0
        return

    if isinstance(combo, PairStep):
        out[cursor + _OFF_INTENT_KIND + _KIND_PAIR_STEP] = 1.0
        # combo.rank is the start_rank (lowest pair's rank); combo.length is len(pairs).
        out[cursor + _OFF_PRIMARY_RANK + (combo.rank - 1)] = 1.0
        out[cursor + _OFF_LENGTH + (combo.length - 2)] = 1.0
        if any(p.a is PHOENIX or p.b is PHOENIX for p in combo.pairs):
            out[cursor + _OFF_PHOENIX_USED] = 1.0
        return

    if isinstance(combo, Straight):
        out[cursor + _OFF_INTENT_KIND + _KIND_STRAIGHT] = 1.0
        # combo.rank is the lowest effective rank (1 for mahjong-led).
        out[cursor + _OFF_PRIMARY_RANK + (combo.rank - 1)] = 1.0
        out[cursor + _OFF_LENGTH + (combo.length - 2)] = 1.0
        if PHOENIX in combo.cards:
            out[cursor + _OFF_PHOENIX_USED] = 1.0
        return

    if isinstance(combo, FourOfAKindBomb):
        # FourOfAKindBomb cannot contain Phoenix (engine enforces this) — no
        # phoenix_used branch needed.
        out[cursor + _OFF_INTENT_KIND + _KIND_FOUR_BOMB] = 1.0
        out[cursor + _OFF_PRIMARY_RANK + (combo.rank - 1)] = 1.0
        return

    if isinstance(combo, StraightFlushBomb):
        # StraightFlushBomb cannot contain Phoenix (engine enforces this).
        # Suit dropped from v4 encoding — see ADR-0017, zero beat-relevant signal.
        out[cursor + _OFF_INTENT_KIND + _KIND_SF_BOMB] = 1.0
        out[cursor + _OFF_PRIMARY_RANK + (combo.rank - 1)] = 1.0
        out[cursor + _OFF_LENGTH + (combo.length - 2)] = 1.0
        return
