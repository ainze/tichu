"""Bomb Window protocol (ADR-0045): which seats get an out-of-turn chance to bomb
after an action, and in what order. Adopted from the tichu-nuxt arena contract.

Every test drives the real engine with `step`, then asks `bomb_window_queue`
which seats to offer a Bomb Window.
"""

from tichu_engine.cards import Card, Suit
from tichu_engine.combinations import Single
from tichu_engine.engine import step
from tichu_engine.legality import bomb_window_queue
from tichu_engine.state import GameState, PublicState, Trick


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _bomb(rank: int) -> frozenset:
    return frozenset(_c(s, rank) for s in (Suit.JADE, Suit.SWORD, Suit.PAGODA, Suit.STAR))


def _filler(*ranks: int) -> frozenset:
    return frozenset(_c(Suit.JADE, r) for r in ranks)


def _leading(hands: dict[int, frozenset], current_player: int = 0) -> GameState:
    """A mid-round state with an empty Trick, `current_player` to lead."""
    full = tuple(hands.get(p, frozenset()) for p in range(4))
    public = PublicState(
        current_player=current_player,
        hand_sizes=tuple(len(h) for h in full),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=Trick.empty(),
    )
    return GameState(hands=full, public=public)


def test_a_seat_holding_a_beating_bomb_gets_a_window_after_a_play():
    lead = Single(_c(Suit.SWORD, 9))
    state = _leading({
        0: frozenset({lead.card}) | _filler(2),
        1: _filler(3),
        2: _bomb(5) | _filler(4),
        3: _filler(6),
    })
    after, _, _, _ = step(state, lead)
    assert bomb_window_queue(after, lead, actor=0) == (2,)


def test_windows_are_offered_in_turn_order_from_the_player():
    # Seat 1 plays; seat 2 is now to act. Seats 3 and 0 both hold beating bombs:
    # turn order from seat 1 asks 3 before 0 (not seat-number order).
    lead = Single(_c(Suit.SWORD, 9))
    state = _leading({
        0: _bomb(5) | _filler(2),
        1: frozenset({lead.card}) | _filler(3),
        2: _filler(4),
        3: _bomb(6) | _filler(7),
    }, current_player=1)
    after, _, _, _ = step(state, lead)
    assert bomb_window_queue(after, lead, actor=1) == (3, 0)


def test_no_window_opens_after_a_pass():
    from tichu_engine.legality import PASS

    lead = Single(_c(Suit.SWORD, 9))
    state = _leading({
        0: frozenset({lead.card}) | _filler(2),
        1: _filler(3),
        2: _filler(4),
        3: _bomb(5) | _filler(6),
    })
    after_lead, _, _, _ = step(state, lead)
    after_pass, _, _, _ = step(after_lead, PASS)  # seat 1 passes; seat 2 to act
    assert bomb_window_queue(after_pass, PASS, actor=1) == ()


def test_no_window_opens_after_the_dog():
    # The Dog passes the lead to the partner and leaves no top to bomb.
    from tichu_engine.cards import DOG

    dog = Single(DOG)
    state = _leading({
        0: frozenset({DOG}) | _filler(2),
        1: _bomb(5) | _filler(3),
        2: _filler(4),
        3: _bomb(6) | _filler(7),
    })
    after, _, _, _ = step(state, dog)
    assert bomb_window_queue(after, dog, actor=0) == ()


def _mahjong_lead_state():
    from tichu_engine.cards import MAHJONG

    return _leading({
        0: frozenset({MAHJONG}) | _filler(2),
        1: _filler(3),
        2: _bomb(5) | _filler(4),
        3: _filler(6),
    })


def test_no_window_opens_while_the_mahjong_wish_is_pending():
    from tichu_engine.cards import MAHJONG

    mahjong = Single(MAHJONG)
    after, _, _, _ = step(_mahjong_lead_state(), mahjong)
    assert after.public.pending_decision is not None  # the wish is still owed
    assert bomb_window_queue(after, mahjong, actor=0) == ()


def test_the_window_on_a_mahjong_opens_once_the_wish_is_made():
    from tichu_engine.cards import MAHJONG
    from tichu_engine.legality import MahjongWish

    after_play, _, _, _ = step(_mahjong_lead_state(), Single(MAHJONG))
    wish = MahjongWish(rank=7)
    after_wish, _, _, _ = step(after_play, wish)
    assert bomb_window_queue(after_wish, wish, actor=0) == (2,)


def test_windows_reopen_on_a_window_bomb_for_seats_that_can_beat_it():
    # Seat 2 bombs (5s) out of turn over seat 0's lead. Windows re-open on the
    # bomb, from the bomber: seat 0 (6s) can beat it, seat 1 (4s) cannot.
    from tichu_engine.combinations import FourOfAKindBomb
    from tichu_engine.legality import BombInterrupt

    lead = Single(_c(Suit.SWORD, 9))
    fives = FourOfAKindBomb(*sorted(_bomb(5), key=lambda c: c.suit.value))
    state = _leading({
        0: frozenset({lead.card}) | _bomb(6),
        1: _bomb(4) | _filler(3),
        2: _bomb(5) | _filler(2),
        3: _filler(7),
    })
    after_lead, _, _, _ = step(state, lead)
    interrupt = BombInterrupt(player=2, bomb=fives)
    after_bomb, _, _, _ = step(after_lead, interrupt)
    assert after_bomb.public.current_player == 3
    assert bomb_window_queue(after_bomb, interrupt, actor=2) == (0,)
