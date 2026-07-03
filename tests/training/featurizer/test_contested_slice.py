"""Contested-trick slice predicate (ADR-0039 pre-check).

A decision is on the contested-trick slice iff all three hold:
  1. the Trick is non-empty (there is a standing top to overtake),
  2. the pile has non-zero point stakes (negative counts — an avoid-the-pile
     decision is still point-relevant), and
  3. the acting seat has at least one legal non-Pass action (it *can* contest).

This is the slice on which H1/H2's residual lift is measured; getting it wrong
silently biases the whole pre-check, hence it is unit-pinned.
"""

from tichu_engine.cards import Card, DRAGON, PHOENIX, Suit
from tichu_engine.combinations import Single
from tichu_engine.legality import Pass
from tichu_engine.state import Play, PrivateState, PublicState, Trick

from tichu_training.trick_stakes import is_contested_trick_decision


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


def test_beatable_pile_with_points_is_contested() -> None:
    # Seat 1 leads a 10 (10 pts); acting seat can legally beat it with a King.
    ps = _ps(0, (1, Single(_c(Suit.JADE, 10))))
    legal = frozenset({Pass(), Single(_c(Suit.SWORD, 13))})
    assert is_contested_trick_decision(ps, legal) is True


def test_leading_an_empty_trick_is_not_contested() -> None:
    # Acting seat is leading: no standing pile to overtake, even with legal plays.
    ps = _ps(0)
    legal = frozenset({Single(_c(Suit.JADE, 10))})
    assert is_contested_trick_decision(ps, legal) is False


def test_only_pass_legal_is_not_contested() -> None:
    # Pile has points, but the acting seat can only Pass -> cannot contest.
    ps = _ps(0, (1, Single(_c(Suit.JADE, 10))))
    legal = frozenset({Pass()})
    assert is_contested_trick_decision(ps, legal) is False


def test_zero_point_pile_is_not_contested() -> None:
    # A beatable pile with no point stakes is outside the slice.
    ps = _ps(0, (1, Single(_c(Suit.JADE, 7))))  # a 7 is worth 0
    legal = frozenset({Pass(), Single(_c(Suit.SWORD, 9))})
    assert is_contested_trick_decision(ps, legal) is False


def test_negative_pile_still_counts_as_stakes() -> None:
    # Phoenix-only pile nets -25; avoiding it is still a point-relevant choice.
    ps = _ps(0, (1, Single(PHOENIX)))
    legal = frozenset({Pass(), Single(DRAGON)})
    assert is_contested_trick_decision(ps, legal) is True
