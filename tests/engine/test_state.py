"""Tichu game state.

Covers Trick (cards currently on the table), PublicState (what all players see),
PrivateState (own hand + public view), and the initial deal.
"""

import pytest

from tichu_engine.cards import Card, MAHJONG, PHOENIX, Suit
from tichu_engine.combinations import Pair, Single
from tichu_engine.state import (
    PrivateState,
    PublicState,
    Trick,
    deal_initial_state,
)


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


# ---- Trick ----

def test_empty_trick_has_no_plays_and_no_leader():
    t = Trick.empty()
    assert t.plays == ()
    assert t.leader is None
    assert t.top_combination is None


def test_trick_after_one_play_records_player_and_combination():
    s = Single(_c(Suit.JADE, 7))
    t = Trick.empty().add_play(player=0, combination=s)
    assert len(t.plays) == 1
    assert t.plays[0].player == 0
    assert t.plays[0].combination is s
    assert t.leader == 0
    assert t.top_combination is s


def test_trick_after_a_pass_keeps_previous_leader():
    s = Single(_c(Suit.JADE, 7))
    t = Trick.empty().add_play(player=0, combination=s).add_pass(player=1)
    assert t.leader == 0
    assert t.top_combination is s
    assert t.passes == frozenset({1})


def test_trick_after_two_plays_updates_leader_and_top():
    seven = Single(_c(Suit.JADE, 7))
    nine = Single(_c(Suit.SWORD, 9))
    t = Trick.empty().add_play(player=0, combination=seven).add_play(player=1, combination=nine)
    assert t.leader == 1
    assert t.top_combination is nine


# ---- PublicState ----

def test_public_state_hand_sizes_sum_to_total_cards_in_play():
    ps = PublicState(
        current_player=0,
        hand_sizes=(14, 14, 14, 14),
        scores=(0, 0),
        trick=Trick.empty(),
    )
    assert sum(ps.hand_sizes) == 56


def test_public_state_is_immutable():
    ps = PublicState(
        current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0), trick=Trick.empty()
    )
    with pytest.raises(Exception):
        ps.current_player = 1  # type: ignore[misc]


def test_public_state_rejects_invalid_current_player():
    with pytest.raises(ValueError):
        PublicState(
            current_player=4, hand_sizes=(14, 14, 14, 14), scores=(0, 0), trick=Trick.empty()
        )


def test_public_state_rejects_wrong_number_of_hand_sizes():
    with pytest.raises(ValueError):
        PublicState(
            current_player=0, hand_sizes=(14, 14, 14), scores=(0, 0), trick=Trick.empty()  # type: ignore[arg-type]
        )


# ---- PrivateState ----

def test_private_state_exposes_hand_and_public_view():
    hand = frozenset({_c(Suit.JADE, 7), _c(Suit.SWORD, 9)})
    public = PublicState(
        current_player=0, hand_sizes=(2, 0, 0, 0), scores=(0, 0), trick=Trick.empty()
    )
    pv = PrivateState(player=0, hand=hand, public=public)
    assert pv.player == 0
    assert pv.hand == hand
    assert pv.public is public


def test_private_state_hand_size_must_match_public_hand_size():
    # PrivateState is the per-player view; the player's own hand size must match
    # what the public state advertises for that player.
    hand = frozenset({_c(Suit.JADE, 7)})  # size 1
    public = PublicState(
        current_player=0, hand_sizes=(2, 0, 0, 0), scores=(0, 0), trick=Trick.empty()  # claims 2
    )
    with pytest.raises(ValueError):
        PrivateState(player=0, hand=hand, public=public)


# ---- Initial deal ----

def test_deal_initial_state_gives_each_player_fourteen_cards():
    state = deal_initial_state(seed=0)
    public = state.public
    assert public.hand_sizes == (14, 14, 14, 14)
    for player in range(4):
        priv = state.private_view(player)
        assert len(priv.hand) == 14


def test_deal_initial_state_partitions_the_full_deck():
    state = deal_initial_state(seed=0)
    all_cards = set()
    for player in range(4):
        all_cards |= set(state.private_view(player).hand)
    assert len(all_cards) == 56  # full Tichu deck


def test_deal_initial_state_mahjong_holder_leads():
    state = deal_initial_state(seed=0)
    mahjong_holder = next(
        p for p in range(4) if MAHJONG in state.private_view(p).hand
    )
    assert state.public.current_player == mahjong_holder


def test_deal_initial_state_is_deterministic_with_seed():
    a = deal_initial_state(seed=42)
    b = deal_initial_state(seed=42)
    for player in range(4):
        assert a.private_view(player).hand == b.private_view(player).hand


def test_deal_initial_state_different_seeds_give_different_deals():
    a = deal_initial_state(seed=0)
    b = deal_initial_state(seed=1)
    # At least one player should have a different hand.
    assert any(
        a.private_view(p).hand != b.private_view(p).hand for p in range(4)
    )
