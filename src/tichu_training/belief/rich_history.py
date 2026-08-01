"""Rich play-history projections for the **Belief Model** (ADR-0041 follow-up).

Featurizer v6 carries ADR-0028's B-core negative-information channels — but they
are **27 numbers for an entire Round**: `declined_top` is `max(rank)` per opponent
per combo type, and it drops

* **context** — a Pass while the decliner's *partner* holds the Trick is near-zero
  information; a Pass while an *opponent* is winning a fat Trick is strong
  evidence of inability. Both write the same update.
* **length** — `combo_type_and_rank` returns `(type, primary_rank)` only, so
  declining a 5-long and a 10-long Straight to the same rank are identical.
* **Bombs** — `combo_type_and_rank` returns `None` for Bombs, so declining to
  bomb records nothing at all.
* **stakes** — the point value of the declined Trick is absent.
* **the Mahjong-wish void** — a rules-derived *certainty*, discarded entirely.

This module reconstructs those channels so the ADR-0041 pre-check can measure an
upper bound on what any history representation could buy, rather than concluding
from an impoverished input. It is a measurement instrument: nothing here is on
the serving path.
"""

from __future__ import annotations

import numpy as np

from tichu_engine.combinations import PairStep, Straight, combo_type_and_rank
from tichu_engine.engine import trick_point_value
from tichu_engine.deck import Card
from tichu_engine.legality import BombInterrupt, Pass, _cards_in

_NUM_OPPONENTS = 3
_NUM_CARDS = 56
_NUM_TYPES = 6
_RANKS = 13  # wish ranks 2..14

# Per-opponent layout.
_DECLINED_CTX = _NUM_TYPES * 3   # max rank declined per type x {partner/opp/none} winning
_DECLINED_COUNT = _NUM_TYPES
_DECLINED_LEN = 2                # max length declined for PairStep / Straight
_DECLINED_BOMB = 1               # times declined to beat a Bomb
_DECLINED_STAKES = 1             # richest Trick declined
_WISH_VOID = _RANKS              # PROVEN void (lead ducked an active wish)
_WISH_SOFT = _RANKS              # evidence only (followed without the wished rank)
_LEADS = 3                       # tricks led, lowest single lead, highest single lead
_PRESSURE = 2                    # pass fraction, decision count

PER_OPPONENT = (
    _DECLINED_CTX + _DECLINED_COUNT + _DECLINED_LEN + _DECLINED_BOMB
    + _DECLINED_STAKES + _WISH_VOID + _WISH_SOFT + _LEADS + _PRESSURE
)
RICH_HISTORY_DIM = _NUM_OPPONENTS * PER_OPPONENT + _NUM_CARDS


def wish_void_deduction(action, wish: int | None, *, is_lead: bool) -> int | None:
    """The rank `action` **proves** its player does not hold, or None.

    `_apply_wish` restricts to wish-fulfilling actions *among already-legal ones*.
    When **leading**, every combination in hand is legal, so holding a natural
    card of the wished rank would force it — leading anything else is proof of a
    void. When **following**, candidates are only combinations that beat the top,
    so a player may hold the rank and be unable to use it; and a Pass means no
    legal action existed. Neither is proof, and encoding them as one would put a
    false certainty into the features.
    """
    if wish is None or not is_lead or isinstance(action, (Pass, BombInterrupt)):
        return None
    return None if fulfills_wish(action, wish) else wish


def fulfills_wish(action, wish_rank: int) -> bool:
    """Does `action` put a **natural** card of `wish_rank` on the table?

    `legality._fulfills_wish` calls `_cards_in`, which raises on a
    `BombInterrupt`, so every wish check routes through `action_cards` instead.
    Phoenix substituting for the wished rank does not count — same rule as the
    engine's own check.
    """
    return any(
        isinstance(c, Card) and c.rank == wish_rank for c in action_cards(action)
    )


def action_cards(action) -> list:
    """The cards an action puts on the table.

    A `BombInterrupt` (a non-current player seizing the Trick with a Bomb) wraps
    its Bomb rather than being a Combination, so `_cards_in` raises on it — the
    BSW replay produces these, self-play does not. Anything unrecognised yields
    an empty list: this accumulator is a measurement, not a validator, and must
    not abort a corpus scan over one exotic action.
    """
    if isinstance(action, Pass):
        return []
    combo = action.bomb if isinstance(action, BombInterrupt) else action
    try:
        return list(_cards_in(combo))
    except Exception:  # noqa: BLE001
        return []


def _combo_length(combo) -> int:
    return len(action_cards(combo)) if isinstance(combo, (PairStep, Straight)) else 0


class RichHistory:
    """Per-Round, per-absolute-seat accumulation of the channels v6 discards.

    Folded once per Play Decision, in order, from the `pre_state` (so the Trick
    top is what a Pass declined). Read out relative-seat ordered
    (next / partner / previous) to match the label convention.
    """

    def __init__(self) -> None:
        self.declined_ctx = np.zeros((4, _NUM_TYPES, 3), dtype=np.float32)
        self.declined_count = np.zeros((4, _NUM_TYPES), dtype=np.float32)
        self.declined_len = np.zeros((4, 2), dtype=np.float32)
        self.declined_bomb = np.zeros(4, dtype=np.float32)
        self.declined_stakes = np.zeros(4, dtype=np.float32)
        self.wish_void = np.zeros((4, _RANKS), dtype=np.float32)
        self.wish_soft = np.zeros((4, _RANKS), dtype=np.float32)
        self.lead_count = np.zeros(4, dtype=np.float32)
        self.lead_min_single = np.zeros(4, dtype=np.float32)
        self.lead_max_single = np.zeros(4, dtype=np.float32)
        self.passes = np.zeros(4, dtype=np.float32)
        self.decisions = np.zeros(4, dtype=np.float32)
        self.play_idx = 0
        self.card_play_time = np.zeros(_NUM_CARDS, dtype=np.float32)

    def update(self, seat: int, action, pre_state) -> None:
        if not 0 <= seat < 4:
            return
        public = pre_state.public
        trick = public.trick
        is_lead = trick.leader is None
        wish = public.mahjong_wish
        self.decisions[seat] += 1

        if isinstance(action, Pass):
            self.passes[seat] += 1
            self._fold_decline(seat, trick)
            return

        # A non-Pass play.
        rank = wish_void_deduction(action, wish, is_lead=is_lead)
        if rank is not None:
            self.wish_void[seat, rank - 2] = 1.0
        elif wish is not None and not is_lead and not fulfills_wish(action, wish):
            self.wish_soft[seat, wish - 2] = 1.0

        if is_lead:
            self.lead_count[seat] += 1
            single = _single_lead_rank(action)
            if single is not None:
                cur_min = self.lead_min_single[seat]
                self.lead_min_single[seat] = (
                    single if cur_min == 0 else min(cur_min, single)
                )
                self.lead_max_single[seat] = max(self.lead_max_single[seat], single)

        self.play_idx += 1
        from tichu_training.card_slots import card_slot

        for card in action_cards(action):
            self.card_play_time[card_slot(card)] = self.play_idx

    def _fold_decline(self, seat: int, trick) -> None:
        top = trick.top_combination
        if top is None:
            return
        stakes = float(trick_point_value(trick))
        self.declined_stakes[seat] = max(self.declined_stakes[seat], stakes)

        tr = combo_type_and_rank(top)
        if tr is None:  # a Bomb (or the Dog) — the channel v6 drops entirely.
            self.declined_bomb[seat] += 1.0
            return

        tidx, rank = tr
        winner = trick.leader
        # Context from the DECLINER's own perspective.
        if winner is None:
            ctx = 2
        elif winner % 2 == seat % 2:
            ctx = 0  # partner is winning — declining says almost nothing
        else:
            ctx = 1  # an opponent is winning — declining is real evidence
        self.declined_ctx[seat, tidx, ctx] = max(
            self.declined_ctx[seat, tidx, ctx], rank
        )
        self.declined_count[seat, tidx] += 1.0
        length = _combo_length(top)
        if length:
            slot = 0 if tidx == 4 else 1
            self.declined_len[seat, slot] = max(self.declined_len[seat, slot], length)

    def block(self, seat: int) -> np.ndarray:
        """`(RICH_HISTORY_DIM,)` from `seat`'s view, opponents in relative-seat
        order. All channels normalised to roughly [0, 1]."""
        rel = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
        parts: list[np.ndarray] = []
        for opp in rel:
            decisions = max(1.0, float(self.decisions[opp]))
            parts.append(np.concatenate([
                (self.declined_ctx[opp] / 14.0).reshape(-1),
                np.minimum(self.declined_count[opp], 10.0) / 10.0,
                self.declined_len[opp] / 14.0,
                np.minimum(self.declined_bomb[opp:opp + 1], 5.0) / 5.0,
                np.clip(self.declined_stakes[opp:opp + 1], 0.0, 25.0) / 25.0,
                self.wish_void[opp],
                self.wish_soft[opp],
                np.array([
                    min(self.lead_count[opp], 14.0) / 14.0,
                    self.lead_min_single[opp] / 14.0,
                    self.lead_max_single[opp] / 14.0,
                    self.passes[opp] / decisions,
                    min(self.decisions[opp], 20.0) / 20.0,
                ], dtype=np.float32),
            ]).astype(np.float32))
        cap = max(1.0, float(self.play_idx))
        parts.append(self.card_play_time / cap)
        return np.concatenate(parts).astype(np.float32)


def _single_lead_rank(action) -> int | None:
    cards = action_cards(action)
    if len(cards) != 1:
        return None
    rank = getattr(cards[0], "rank", None)
    if isinstance(rank, int) and 2 <= rank <= 14:
        return rank
    return None
