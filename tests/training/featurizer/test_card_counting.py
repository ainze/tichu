"""H3 pre-check probe: remaining-cards-by-rank multi-hot.

Ranks still UNSEEN among opponents = full deck - own hand - cards played this
Round. Derivable from v6 (own_hand + played_by); this block makes it explicit so
the residual pre-check can measure whether an explicit representation helps.
Layout: indices 0..12 = natural ranks 2..14; 13=Dragon 14=Phoenix 15=Mahjong
16=Dog.
"""

import numpy as np

from tichu_engine.cards import Card, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_engine.state import PrivateState, PublicState, Trick

from tichu_training.card_counting import (
    REMAINING_BLOCK_DIM, remaining_by_rank_block,
)

_SUITS = (Suit.JADE, Suit.SWORD, Suit.PAGODA, Suit.STAR)


def _ps(hand, played=frozenset()) -> PrivateState:
    public = PublicState(
        current_player=0, hand_sizes=(len(hand), 14, 14, 14),
        scores=(0, 0), trick=Trick.empty(),
        played_cards_this_round=frozenset(played),
    )
    return PrivateState(player=0, hand=frozenset(hand), public=public)


def test_holding_all_four_of_a_rank_clears_that_rank_bit() -> None:
    hand = {Card(s, 5) for s in _SUITS}  # all four 5s
    block = remaining_by_rank_block(_ps(hand))
    assert block.shape == (REMAINING_BLOCK_DIM,)
    assert block[3] == 0.0                       # rank 5 -> none unseen
    assert block[0] == 1.0 and block[12] == 1.0  # ranks 2 and 14 still out
    assert block[13] == 1.0                      # Dragon still unseen


def test_cards_played_this_round_clear_their_rank() -> None:
    # Hold four 5s; all four 6s already played this round -> ranks 5 and 6 gone.
    hand = {Card(s, 5) for s in _SUITS}
    played = {Card(s, 6) for s in _SUITS}
    block = remaining_by_rank_block(_ps(hand, played))
    assert block[3] == 0.0  # rank 5 (held)
    assert block[4] == 0.0  # rank 6 (all played)
    assert block[5] == 1.0  # rank 7 still out


def test_self_held_special_clears_its_bit() -> None:
    # Holding the Dragon means it is not unseen among opponents.
    block = remaining_by_rank_block(_ps({DRAGON, Card(Suit.JADE, 2)}))
    assert block[13] == 0.0  # Dragon accounted for
    assert block[14] == 1.0  # Phoenix still out


def test_start_of_round_unheld_ranks_are_all_present() -> None:
    # A hand with no rank-9 card leaves all four 9s unseen -> bit set.
    hand = {Card(Suit.JADE, r) for r in range(2, 9)}  # 2..8, no 9s
    block = remaining_by_rank_block(_ps(hand))
    assert block[7] == 1.0   # rank 9 fully unseen
    assert block[16] == 1.0  # Dog unseen
