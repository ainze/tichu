"""Assembled stakes feature block (ADR-0039): the 5 dims the pre-check appends
to the v6 Feature Vector and the v7 featurizer will emit —
[trick_point_value / 25, winner_self, winner_next, winner_partner, winner_prev].
All-zero on an empty Trick.
"""

import numpy as np

from tichu_engine.cards import Card, DRAGON, Suit
from tichu_engine.combinations import Single
from tichu_engine.state import Play, PrivateState, PublicState, Trick

from tichu_training.trick_stakes import STAKES_BLOCK_DIM, stakes_block


_FULL_HAND = frozenset(Card(Suit.JADE, r) for r in range(2, 15)) | {Card(Suit.SWORD, 2)}


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _ps(player: int, *plays: tuple[int, object]) -> PrivateState:
    ps = tuple(Play(player=p, combination=c) for p, c in plays)
    trick = Trick(plays=ps, leader=ps[0].player if ps else None)
    public = PublicState(
        current_player=player, hand_sizes=(14, 14, 14, 14),
        scores=(0, 0), trick=trick,
    )
    return PrivateState(player=player, hand=_FULL_HAND, public=public)


def test_empty_trick_block_is_all_zero() -> None:
    block = stakes_block(_ps(0))
    assert block.shape == (STAKES_BLOCK_DIM,)
    assert block.dtype == np.float32
    assert not block.any()


def test_partner_winning_dragon_pile_sets_scalar_and_partner_slot() -> None:
    # Partner (seat 2) holds the Dragon (+25); acting seat 0.
    block = stakes_block(_ps(0, (2, Single(DRAGON))))
    assert block[0] == 1.0  # 25 / 25
    assert list(block[1:]) == [0.0, 0.0, 1.0, 0.0]  # partner slot


def test_scalar_is_normalised_by_twenty_five() -> None:
    # Self leads a 10 (10 pts) -> 10/25 = 0.4 in the self slot.
    block = stakes_block(_ps(0, (0, Single(_c(Suit.JADE, 10)))))
    assert abs(block[0] - 0.4) < 1e-6
    assert list(block[1:]) == [1.0, 0.0, 0.0, 0.0]


def test_block_winner_is_the_raiser_not_the_leader() -> None:
    # Seat 1 leads (10 pts), seat 3 raises with a King; acting seat 0.
    # Winner slot is seat 3 (previous), scalar 20/25.
    block = stakes_block(_ps(
        0,
        (1, Single(_c(Suit.JADE, 10))),
        (3, Single(_c(Suit.SWORD, 13))),
    ))
    assert abs(block[0] - 0.8) < 1e-6  # 20 / 25
    assert list(block[1:]) == [0.0, 0.0, 0.0, 1.0]  # previous, not the leader
