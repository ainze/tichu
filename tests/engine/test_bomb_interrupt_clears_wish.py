"""Regression (#109): an out-of-turn bomb fulfils a Mahjong wish like any play.

A wish is cleared once a natural card of the wished rank is played. That must
hold for a Bomb Interrupt too — otherwise later seats stay forced by a wish
that has already been granted.
"""

from dataclasses import replace

from tichu_engine.cards import Card, Suit
from tichu_engine.combinations import FourOfAKindBomb, Single
from tichu_engine.engine import step
from tichu_engine.legality import PASS, BombInterrupt, legal_actions
from tichu_engine.state import GameState, PublicState, Trick


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _four(rank: int) -> frozenset:
    return frozenset(_c(s, rank) for s in (Suit.JADE, Suit.SWORD, Suit.PAGODA, Suit.STAR))


def _bomb(rank: int) -> FourOfAKindBomb:
    return FourOfAKindBomb(*sorted(_four(rank), key=lambda c: c.suit.value))


def _nine_is_wished_and_seat_0_has_led() -> GameState:
    """A wish for 9 is active. Seat 0 (no 9) has led a 3; seat 1 is to act.
    Seat 2 holds four 9s and four 5s; seat 3 holds J-Q (the wish cannot bind it)."""
    lead = Single(_c(Suit.SWORD, 3))
    hands = (
        frozenset({lead.card, _c(Suit.JADE, 2)}),
        frozenset({_c(Suit.JADE, 4), _c(Suit.JADE, 6)}),
        _four(9) | _four(5) | {_c(Suit.JADE, 7)},
        frozenset({_c(Suit.JADE, 11), _c(Suit.JADE, 12)}),
    )
    public = PublicState(
        current_player=0,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=Trick.empty(),
    )
    public = replace(public, mahjong_wish=9)
    after_lead, _, _, _ = step(GameState(hands=hands, public=public), lead)
    assert after_lead.public.mahjong_wish == 9
    return after_lead


def test_out_of_turn_bomb_of_the_wished_rank_clears_the_wish():
    state = _nine_is_wished_and_seat_0_has_led()

    after, _, _, _ = step(state, BombInterrupt(player=2, bomb=_bomb(9)))

    assert after.public.mahjong_wish is None
    # Seat 3 is next and cannot beat the bomb: it may simply pass.
    assert legal_actions(after) == frozenset({PASS})


def test_out_of_turn_bomb_without_the_wished_rank_leaves_the_wish_active():
    state = _nine_is_wished_and_seat_0_has_led()

    after, _, _, _ = step(state, BombInterrupt(player=2, bomb=_bomb(5)))

    assert after.public.mahjong_wish == 9
