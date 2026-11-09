"""Regression: the third player going out with a Mahjong-containing play.

The play leaves a wish pending, so the round can only finish once the wish is
declared. The trick that play started must still be scored: a lead of
Mahjong-2-3-4-5 carries 5 points, and they belong to the player who went out.
"""

from dataclasses import replace

import pytest

from tichu_engine.cards import Card, MAHJONG, Suit
from tichu_engine.combinations import Straight
from tichu_engine.engine import step
from tichu_engine.legality import MahjongWish, legal_actions
from tichu_engine.state import GameState, MahjongWishPending, PublicState, Trick


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _third_out_leads_mahjong_straight() -> GameState:
    """Seats 0 and 1 are out (no slam). Seat 2 leads its last five cards,
    Mahjong-2-3-4-5; seat 3 is left holding a King (10 points)."""
    straight_cards = frozenset(
        {MAHJONG, _c(Suit.JADE, 2), _c(Suit.SWORD, 3), _c(Suit.PAGODA, 4), _c(Suit.STAR, 5)}
    )
    hands = (frozenset(), frozenset(), straight_cards, frozenset({_c(Suit.JADE, 13)}))
    public = PublicState(
        current_player=2,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=Trick.empty(),
    )
    public = replace(public, out_order=(0, 1))
    return GameState(hands=hands, public=public)


def _the_straight(state: GameState) -> Straight:
    (straight,) = [a for a in legal_actions(state) if isinstance(a, Straight) and len(a.cards) == 5]
    return straight


@pytest.mark.parametrize("wish", [None, 7])
def test_trick_points_counted_when_third_player_goes_out_with_mahjong(wish):
    state = _third_out_leads_mahjong_straight()

    after_play, _, _, _ = step(state, _the_straight(state))
    assert after_play.public.pending_decision == MahjongWishPending(player=2)

    final, _, done, _ = step(after_play, MahjongWish(rank=wish))

    assert done
    # Team 0 gets the trick's 5 points (seat 2 won it) plus seat 3's King (10)
    # in hand, which goes to the opponents of the last player.
    assert final.public.scores == (15, 0)
