"""PrivateState JSON codec — round-trip."""

import pytest

from tichu_engine.cards import DRAGON, MAHJONG, PHOENIX, Card, Suit
from tichu_engine.combinations import Pair, Single
from tichu_engine.legality import DragonGive, MahjongWish, PASS, SchupfenPass
from tichu_engine.state import (
    DragonGivePending,
    GameState,
    MahjongWishPending,
    Play,
    PublicState,
    SchupfenPending,
    Trick,
    deal_initial_state,
    deal_for_schupfen,
)

from tichu_inference.codec import (
    action_to_json,
    private_state_from_json,
    private_state_to_json,
)


def test_deal_initial_round_trip():
    state = deal_initial_state(seed=0)
    ps = state.private_view(0)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_schupfen_pending_round_trip():
    state = deal_for_schupfen(seed=0)
    ps = state.private_view(0)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_mid_trick_round_trip():
    state = deal_initial_state(seed=0)
    public = state.public
    # Synthesise a mid-trick state with a top Pair using whichever rank
    # appears at least twice in seat 0's hand.
    by_rank: dict[int, list[Card]] = {}
    for c in state.hands[0]:
        if isinstance(c, Card):
            by_rank.setdefault(c.rank, []).append(c)
    a, b = next(cards for cards in by_rank.values() if len(cards) >= 2)[:2]
    pair = Pair(a, b)
    trick = Trick(plays=(Play(player=0, combination=pair),), leader=0)
    public = PublicState(
        current_player=1,
        hand_sizes=public.hand_sizes,
        scores=public.scores,
        trick=trick,
        round_points_by_player=(1, 2, 3, 4),
        out_order=(),
        tichu_callers=frozenset({2}),
        grand_tichu_callers=frozenset(),
        mahjong_wish=7,
    )
    ps = state.private_view(1).__class__(player=1, hand=state.hands[1], public=public)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_dragon_give_pending_round_trip():
    state = deal_initial_state(seed=0)
    public = state.public
    public = PublicState(
        current_player=public.current_player,
        hand_sizes=public.hand_sizes,
        scores=public.scores,
        trick=public.trick,
        pending_decision=DragonGivePending(winner=2, points=25),
    )
    ps = state.private_view(0).__class__(player=0, hand=state.hands[0], public=public)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_mahjong_wish_pending_round_trip():
    state = deal_initial_state(seed=0)
    public = state.public
    public = PublicState(
        current_player=public.current_player,
        hand_sizes=public.hand_sizes,
        scores=public.scores,
        trick=public.trick,
        pending_decision=MahjongWishPending(player=public.current_player),
    )
    ps = state.private_view(0).__class__(player=0, hand=state.hands[0], public=public)
    blob = private_state_to_json(ps)
    restored = private_state_from_json(blob)
    assert restored == ps


def test_action_to_json_single():
    blob = action_to_json(Single(Card(suit=Suit.JADE, rank=5)))
    assert blob["kind"] == "Single"


def test_action_to_json_pass():
    assert action_to_json(PASS) == {"kind": "Pass"}


def test_action_to_json_dragon_give():
    assert action_to_json(DragonGive(target=2)) == {"kind": "DragonGive", "target": 2}


def test_action_to_json_mahjong_wish():
    assert action_to_json(MahjongWish(rank=7)) == {"kind": "MahjongWish", "rank": 7}
    assert action_to_json(MahjongWish(rank=None)) == {"kind": "MahjongWish", "rank": None}


def test_action_to_json_schupfen_pass():
    a = Card(Suit.JADE, 4)
    b = Card(Suit.SWORD, 5)
    c = Card(Suit.PAGODA, 6)
    blob = action_to_json(SchupfenPass(to_next=a, to_partner=b, to_previous=c))
    assert blob["kind"] == "SchupfenPass"
    assert "to_next" in blob and "to_partner" in blob and "to_previous" in blob


def test_malformed_payload_raises_helpful_error():
    with pytest.raises(ValueError):
        private_state_from_json({"hand": [0, 1, 2]})  # missing player + public.


def test_special_cards_roundtrip():
    state = deal_initial_state(seed=0)
    # Find which player holds Dragon, Phoenix, Mahjong — ensure they survive round-trip.
    for ps_player in range(4):
        ps = state.private_view(ps_player)
        if not (DRAGON in ps.hand or PHOENIX in ps.hand or MAHJONG in ps.hand):
            continue
        blob = private_state_to_json(ps)
        restored = private_state_from_json(blob)
        assert restored.hand == ps.hand
