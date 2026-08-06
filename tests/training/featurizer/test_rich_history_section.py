"""v7 (ADR-0044): the featurizer READS the engine's Rich History Block.

Two properties matter and both are easy to get subtly wrong:

  * **Relative-seat ordering** (next / partner / previous), like every other
    per-opponent v6 section. Absolute-seat planes would break under the
    Tournament's seat-swap and the trunk would learn seat-specific junk.
  * **Normalisation to roughly [0, 1]**, with the RAW values kept in engine
    state — the v6 convention ("Raw ranks — the featurizer normalises rank/14").

Self is deliberately absent: a seat's own decline history is not information it
needs inferred, and including it would waste a quarter of the block.
"""

import numpy as np

from tichu_engine.card_slots import card_slot
from tichu_engine.cards import Card, Suit
from tichu_engine.rich_history import (
    CTX_OPPONENT_WINNING,
    INTENT_SINGLE,
    RichHistory,
)
from tichu_engine.state import PrivateState, PublicState, Trick
from tichu_training.featurizer import SECTION_DIMS, SECTION_OFFSETS, featurize


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _private(seat: int, rich: RichHistory) -> PrivateState:
    hand = frozenset({_c(Suit.JADE, 3), _c(Suit.SWORD, 4)})
    public = PublicState(
        current_player=seat,
        hand_sizes=(2, 2, 2, 2),
        scores=(0, 0),
        trick=Trick.empty(),
        rich_history=rich,
    )
    return PrivateState(player=seat, hand=hand, public=public)


def _section(vec: np.ndarray, name: str) -> np.ndarray:
    off = SECTION_OFFSETS[name]
    return vec[off:off + SECTION_DIMS[name]]


def test_decline_context_is_relative_seat_ordered_and_normalised():
    """Seat 0 acts; seat 2 (its PARTNER, relative index 1) declined a King.

    The value must land in the partner's slice — not seat 2's absolute slot — and
    arrive as 13/14, matching the v6 `declined_top` normalisation.
    """
    rich = RichHistory().with_decline(2, INTENT_SINGLE, CTX_OPPONENT_WINNING, 13)
    ctx = _section(featurize(_private(0, rich)), "declined_ctx")

    per_opp = SECTION_DIMS["declined_ctx"] // 3
    slot = INTENT_SINGLE * 3 + CTX_OPPONENT_WINNING
    partner = 1  # next=0, partner=1, previous=2
    assert ctx[partner * per_opp + slot] == np.float32(13.0 / 14.0)
    # The other two opponents' slices stay empty.
    assert ctx[0 * per_opp + slot] == 0.0
    assert ctx[2 * per_opp + slot] == 0.0


def test_the_acting_seats_own_history_is_not_in_the_block():
    """Seat 0's own decline must not appear anywhere: the block is a per-OPPONENT
    projection, and leaking self into it would both waste dims and let the net
    key on its own past instead of inferring."""
    rich = RichHistory().with_decline(0, INTENT_SINGLE, CTX_OPPONENT_WINNING, 13)
    ctx = _section(featurize(_private(0, rich)), "declined_ctx")
    assert not ctx.any()


def test_play_order_is_normalised_by_the_plays_so_far():
    """`card_play_time` is a RECENCY channel: the newest play is ~1.0 and older
    ones decay toward 0, so it stays comparable across Rounds of different
    lengths. Unplayed cards stay exactly 0."""
    jade3, sword4 = _c(Suit.JADE, 3), _c(Suit.SWORD, 4)
    rich = (
        RichHistory()
        .with_play([card_slot(jade3)])
        .with_play([card_slot(sword4)])
    )

    times = _section(featurize(_private(0, rich)), "card_play_time")
    assert times[card_slot(jade3)] == np.float32(0.5)   # play 1 of 2
    assert times[card_slot(sword4)] == np.float32(1.0)  # play 2 of 2
    assert times[card_slot(_c(Suit.STAR, 9))] == 0.0    # never played


def test_the_block_is_identically_zero_at_schupfen_and_grand_tichu():
    """ADR-0044 widens the Call Networks instead of retraining them at v7, and
    this is the proof that licenses it.

    Schupfen is the pre-play card pass; Grand Tichu is called on the first eight
    cards, before Schupfen. `RichHistory` accumulates only on Play Decisions and
    resets per Round, so at those Decisions all 233 dims are identically zero —
    retraining those nets on v7 features could not change them. Zero-initialising
    the added input columns is therefore exactly function-preserving, not
    approximately.
    """
    from tichu_engine.state import deal_for_schupfen
    from tichu_training.featurizer import V6_FEATURIZER_OUTPUT_DIM

    state = deal_for_schupfen(seed=7)
    assert state.public.rich_history == RichHistory(), (
        "a freshly dealt Round must carry an empty accumulator"
    )
    for seat in range(4):
        vec = featurize(state.private_view(seat))
        tail = vec[V6_FEATURIZER_OUTPUT_DIM:]
        assert tail.shape == (233,)
        assert not tail.any(), f"seat {seat} has a non-zero rich block at Schupfen"
