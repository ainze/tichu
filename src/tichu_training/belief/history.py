"""Belief input History block — compressed per-opponent decision projections.

Extends the 224-dim policy Feature Vector with a compact History block so the
Belief Model sees the cross-Trick negative information the policy featurizer
discards (which top each opponent has *declined* to beat). See
[ADR-0028](../../docs/adr/0028-belief-input-compressed-history-projections.md).

Layout of the extended belief input (B-full, 307 dims):

    [ policy_224 | H1 declined_top (18) | H2 lead (6) | H3 pressure (3) | H4 play_time (56) ]

The three tiers are nested **prefixes** of this one vector, so we materialise
once at B-full and select the tier by column prefix at train time:

    A      = [:224]   policy only (the ADR-0021 floor)
    B_core = [:251]   + H1 + H2 + H3   (the negative-information channels)
    B_full = [:307]   + H4             (+ the positive play-order channel)

History values are accumulated as the replay walks decisions in order, indexed
by absolute seat, then read out in relative-seat order (next / partner /
previous) to match the label convention. All values are normalised to [0, 1].
"""

from __future__ import annotations

import numpy as np

from tichu_engine.combinations import (
    FullHouse, Pair, PairStep, Single, Straight, Triple,
)
from tichu_training.card_slots import card_slot
from tichu_training.featurizer import (
    CONTINUOUS_FEATURE_COLUMNS, FEATURIZER_OUTPUT_DIM,
)

BELIEF_INPUT_VERSION = "v1"

_NUM_CARDS = 56
_NUM_TYPES = 6            # Single, Pair, Triple, FullHouse, PairStep, Straight (bombs excluded)
_LEAD_CAP = 14.0         # tricks-led normaliser (a Round has <= ~14 Tricks)

# History block sub-dims.
_H1 = 3 * _NUM_TYPES      # declined_top: 3 opponents x 6 types = 18
_H2 = 3 * 2               # lead_summary: 3 x (count, lowest-single) = 6
_H3 = 3                   # pass_pressure: 3
_H4 = _NUM_CARDS          # play_time: 56
HISTORY_DIM = _H1 + _H2 + _H3 + _H4                      # 83
BELIEF_FEATURE_DIM = FEATURIZER_OUTPUT_DIM + HISTORY_DIM  # 224 + 83 = 307

# Tier -> input column count (nested prefixes; see module docstring).
TIER_DIMS: dict[str, int] = {
    "A": FEATURIZER_OUTPUT_DIM,                       # 224
    "B_core": FEATURIZER_OUTPUT_DIM + _H1 + _H2 + _H3,  # 251
    "B_full": BELIEF_FEATURE_DIM,                     # 307
}

# Combination type -> index 0..5 (matches the featurizer's intent_kind order).
_TYPE_SINGLE, _TYPE_PAIR, _TYPE_TRIPLE = 0, 1, 2
_TYPE_FULL_HOUSE, _TYPE_PAIR_STEP, _TYPE_STRAIGHT = 3, 4, 5


def belief_continuous_columns() -> tuple[int, ...]:
    """Continuous (non-bit-packable) column indices of the 307-dim belief input:
    the policy's continuous columns plus every History column (all floats)."""
    return tuple(CONTINUOUS_FEATURE_COLUMNS) + tuple(
        range(FEATURIZER_OUTPUT_DIM, BELIEF_FEATURE_DIM)
    )


def _combo_type_and_rank(combo: object) -> tuple[int, int] | None:
    """(type_idx 0..5, primary rank 1..14) for a non-bomb Combination, else None.

    Bombs return None (excluded from the decline channel); the Dog (NaN rank)
    returns None — it can never be a Trick top anyway."""
    if isinstance(combo, Single):
        r = combo.rank
        if r != r:                       # NaN (Dog)
            return None
        return (_TYPE_SINGLE, max(1, min(14, int(r))))   # Phoenix 1.5->1, Dragon 25->14
    if isinstance(combo, Pair):
        return (_TYPE_PAIR, combo.rank)
    if isinstance(combo, Triple):
        return (_TYPE_TRIPLE, combo.rank)
    if isinstance(combo, FullHouse):
        return (_TYPE_FULL_HOUSE, combo.triple.rank)
    if isinstance(combo, PairStep):
        return (_TYPE_PAIR_STEP, combo.rank)
    if isinstance(combo, Straight):
        return (_TYPE_STRAIGHT, combo.rank)
    return None                          # FourOfAKindBomb / StraightFlushBomb / other


class HistoryAccumulator:
    """Per-Round, per-absolute-seat accumulation of decline / lead / pressure /
    play-order history. Updated once per Play Decision as the replay walks them
    in order; read out (relative-seat ordered) into the History block."""

    def __init__(self) -> None:
        self.declined_max = [[0] * _NUM_TYPES for _ in range(4)]  # max rank declined per type
        self.lead_count = [0] * 4
        self.lead_min_single = [0] * 4    # 0 = no Single lead yet; else rank 2..14
        self.decisions = [0] * 4
        self.passes = [0] * 4
        self.play_idx = 0                 # global play counter (Passes don't advance it)
        self.card_play_time = [0] * _NUM_CARDS  # 0 = unplayed; else 1-based play index

    def update(self, parsed_action, pre_state) -> None:
        """Fold one Play/Pass Decision into the accumulator. `pre_state` is the
        GameState *before* the action (so its Trick top is what a Pass declined)."""
        seat = parsed_action.player
        if not 0 <= seat < 4:
            return
        kind = parsed_action.kind
        if kind == "pass":
            self.decisions[seat] += 1
            self.passes[seat] += 1
            tr = _combo_type_and_rank(pre_state.public.trick.top_combination)
            if tr is not None:
                tidx, rank = tr
                if rank > self.declined_max[seat][tidx]:
                    self.declined_max[seat][tidx] = rank
        elif kind == "play":
            self.decisions[seat] += 1
            cards = list(parsed_action.cards or ())
            if pre_state.public.trick.leader is None:   # leading a fresh Trick
                self.lead_count[seat] += 1
                if len(cards) == 1 and hasattr(cards[0], "rank"):
                    r = cards[0].rank
                    if isinstance(r, int) and 2 <= r <= 14:
                        cur = self.lead_min_single[seat]
                        self.lead_min_single[seat] = r if cur == 0 else min(cur, r)
            self.play_idx += 1
            for c in cards:
                self.card_play_time[card_slot(c)] = self.play_idx

    def block(self, seat: int) -> np.ndarray:
        """The 83-dim History block from `seat`'s view (opponents next/partner/
        previous). H4 play_time is player-anonymous (same for every seat)."""
        rel = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
        h1 = [self.declined_max[o][t] / 14.0 for o in rel for t in range(_NUM_TYPES)]
        h2: list[float] = []
        for o in rel:
            h2.append(min(self.lead_count[o], 14) / _LEAD_CAP)
            h2.append(self.lead_min_single[o] / 14.0)
        h3 = [
            (self.passes[o] / self.decisions[o]) if self.decisions[o] else 0.0
            for o in rel
        ]
        cap = max(1, self.play_idx)
        h4 = [t / cap for t in self.card_play_time]
        return np.array(h1 + h2 + h3 + h4, dtype=np.float32)


def extend_features(policy_features: np.ndarray, acc: HistoryAccumulator, seat: int) -> np.ndarray:
    """Concatenate the 224 policy features with the 83-dim History block -> 307."""
    return np.concatenate([policy_features, acc.block(seat)]).astype(np.float32, copy=False)
