"""Current-trick winner (H2): relative-seat index of the player who holds the
top Combination (`trick.plays[-1].player`) — the seat you overtake if you beat
the pile. NOT the trick *leader* (opener); the two differ after any raise.

Relative-seat convention matches the rest of the featurizer:
0=self, 1=next, 2=partner, 3=previous = (winner_seat - acting_player) % 4.
"""

from tichu_engine.cards import Card, Suit
from tichu_engine.combinations import Single
from tichu_engine.state import Play, PrivateState, PublicState, Trick

from tichu_training.trick_stakes import current_trick_winner_relseat


_FULL_HAND = frozenset(Card(Suit.JADE, r) for r in range(2, 15)) | {Card(Suit.SWORD, 2)}


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _ps(player: int, *plays: tuple[int, object]) -> PrivateState:
    ps = tuple(Play(player=p, combination=c) for p, c in plays)
    trick = Trick(plays=ps, leader=ps[0].player if ps else None)
    public = PublicState(
        current_player=player,
        hand_sizes=(14, 14, 14, 14),
        scores=(0, 0),
        trick=trick,
    )
    return PrivateState(player=player, hand=_FULL_HAND, public=public)


def test_self_only_play_makes_self_the_winner() -> None:
    ps = _ps(0, (0, Single(_c(Suit.JADE, 7))))
    assert current_trick_winner_relseat(ps) == 0


def test_empty_trick_has_no_winner() -> None:
    ps = _ps(0)
    assert current_trick_winner_relseat(ps) is None


def test_winner_is_the_raiser_not_the_leader() -> None:
    # Seat 1 leads a 9, seat 3 raises with a King. Acting player is 0.
    # Winner is the raiser (seat 3 -> previous -> 3), NOT the leader (seat 1).
    ps = _ps(
        0,
        (1, Single(_c(Suit.JADE, 9))),
        (3, Single(_c(Suit.SWORD, 13))),
    )
    assert ps.public.trick.leader == 1  # opener is seat 1
    assert current_trick_winner_relseat(ps) == 3  # winner is seat 3, not seat 1


def test_partner_winning_maps_to_slot_two() -> None:
    # Partner of seat 0 is seat 2 (across the table).
    ps = _ps(0, (2, Single(_c(Suit.JADE, 14))))
    assert current_trick_winner_relseat(ps) == 2


def test_relative_seat_is_invariant_to_acting_player() -> None:
    # The same absolute winner (seat 2) yields a different relative index
    # depending on who is acting — this is what keeps the feature seat-invariant.
    play = (2, Single(_c(Suit.JADE, 14)))
    assert current_trick_winner_relseat(_ps(0, play)) == 2  # (2-0)%4
    assert current_trick_winner_relseat(_ps(1, play)) == 1  # (2-1)%4
    assert current_trick_winner_relseat(_ps(3, play)) == 3  # (2-3)%4
