"""Trick point-value (H1): sum of card points in the current Trick pile.

Card points: 5->5, 10->10, K(13)->10, Dragon->+25, Phoenix->-25; all else 0.
Public `trick_point_value(trick)` is the single source of the scoring mapping,
reused by the v7 featurizer's `trick_point_value` feature.
"""

from tichu_engine.cards import Card, DRAGON, PHOENIX, Suit
from tichu_engine.combinations import Single
from tichu_engine.engine import trick_point_value
from tichu_engine.state import Play, Trick


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _trick(*plays: tuple[int, object]) -> Trick:
    """Build a Trick from (player, combination) pairs; leader = first player."""
    ps = tuple(Play(player=p, combination=c) for p, c in plays)
    return Trick(plays=ps, leader=ps[0].player if ps else None)


def test_single_five_is_worth_five() -> None:
    trick = _trick((0, Single(_c(Suit.JADE, 5))))
    assert trick_point_value(trick) == 5


def test_empty_trick_has_no_points() -> None:
    assert trick_point_value(Trick.empty()) == 0


def test_non_point_card_is_worth_zero() -> None:
    trick = _trick((0, Single(_c(Suit.JADE, 7))))
    assert trick_point_value(trick) == 0


def test_ten_and_king_are_each_worth_ten() -> None:
    assert trick_point_value(_trick((0, Single(_c(Suit.JADE, 10))))) == 10
    assert trick_point_value(_trick((0, Single(_c(Suit.JADE, 13))))) == 10


def test_dragon_is_worth_twenty_five() -> None:
    assert trick_point_value(_trick((0, Single(DRAGON)))) == 25


def test_phoenix_nets_the_pile_negative() -> None:
    # Phoenix (-25) played over a King (+10) leaves the pile at -15.
    trick = _trick(
        (0, Single(_c(Suit.JADE, 13))),
        (1, Single(PHOENIX)),
    )
    assert trick_point_value(trick) == -15


def test_points_accumulate_across_a_raised_pile() -> None:
    # Lead a 10, another seat raises with a King: both stay in the pile -> 20.
    trick = _trick(
        (0, Single(_c(Suit.JADE, 10))),
        (2, Single(_c(Suit.SWORD, 13))),
    )
    assert trick_point_value(trick) == 20
