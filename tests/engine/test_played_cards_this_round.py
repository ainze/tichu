"""`PublicState.played_cards_this_round` accumulates over a round and resets
at round boundaries. v2 featurizer reads this for the `seen_cards` section."""

import pytest

from tichu_engine.cards import Card, DOG, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_engine.combinations import FourOfAKindBomb, Pair, Single
from tichu_engine.engine import step, _finalise_round
from tichu_engine.legality import BombInterrupt, PASS
from tichu_engine.state import GameState, PublicState, Trick


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


def _state(hands_dict: dict[int, frozenset], top=None, current_player: int = 0,
           played: frozenset = frozenset()) -> GameState:
    hands: list[frozenset] = [frozenset()] * 4
    for p, h in hands_dict.items():
        hands[p] = h
    trick = Trick.empty()
    if top is not None:
        leader = (current_player - 1) % 4
        trick = trick.add_play(player=leader, combination=top)
    public = PublicState(
        current_player=current_player,
        hand_sizes=tuple(len(h) for h in hands),  # type: ignore[arg-type]
        scores=(0, 0),
        trick=trick,
        played_cards_this_round=played,
    )
    return GameState(hands=tuple(hands), public=public)


def test_played_cards_defaults_empty():
    state = _state({0: frozenset({_c(Suit.JADE, 7)})})
    assert state.public.played_cards_this_round == frozenset()


def test_single_play_accumulates_into_played_cards():
    seven = _c(Suit.JADE, 7)
    state = _state(
        {0: frozenset({seven, _c(Suit.SWORD, 3)}), 1: frozenset({_c(Suit.STAR, 10)})},
    )
    next_state, _, _, _ = step(state, Single(seven))
    assert seven in next_state.public.played_cards_this_round
    assert len(next_state.public.played_cards_this_round) == 1


def test_pair_accumulates_both_cards():
    a = _c(Suit.JADE, 7)
    b = _c(Suit.SWORD, 7)
    state = _state(
        {
            0: frozenset({a, b}),
            1: frozenset({_c(Suit.STAR, 10), _c(Suit.JADE, 10)}),
            2: frozenset({_c(Suit.PAGODA, 11)}),
            3: frozenset({_c(Suit.SWORD, 12)}),
        },
    )
    next_state, _, _, _ = step(state, Pair(a, b))
    assert {a, b}.issubset(next_state.public.played_cards_this_round)


def test_pass_does_not_change_played_cards():
    seven = _c(Suit.JADE, 7)
    eight = _c(Suit.SWORD, 8)
    state = _state(
        {0: frozenset({seven}), 1: frozenset({_c(Suit.STAR, 10)})},
        top=Single(eight),
        current_player=1,
        played=frozenset({eight}),
    )
    next_state, _, _, _ = step(state, PASS)
    assert next_state.public.played_cards_this_round == frozenset({eight})


def test_dog_play_accumulates_dog():
    state = _state(
        {
            0: frozenset({DOG}),
            1: frozenset({_c(Suit.STAR, 10)}),
            2: frozenset({_c(Suit.PAGODA, 11)}),
            3: frozenset({_c(Suit.SWORD, 12)}),
        },
    )
    next_state, _, _, _ = step(state, Single(DOG))
    assert DOG in next_state.public.played_cards_this_round


def test_multiple_plays_accumulate_across_steps():
    seven = _c(Suit.JADE, 7)
    eight = _c(Suit.SWORD, 8)
    nine = _c(Suit.PAGODA, 9)
    state = _state(
        {
            0: frozenset({seven, _c(Suit.STAR, 2)}),
            1: frozenset({eight, _c(Suit.STAR, 3)}),
            2: frozenset({nine, _c(Suit.STAR, 4)}),
            3: frozenset({_c(Suit.STAR, 5), _c(Suit.STAR, 6)}),
        },
    )
    s1, _, _, _ = step(state, Single(seven))
    s2, _, _, _ = step(s1, Single(eight))
    s3, _, _, _ = step(s2, Single(nine))
    assert {seven, eight, nine}.issubset(s3.public.played_cards_this_round)


def test_bomb_interrupt_accumulates_bomb_cards():
    sevens = [
        _c(Suit.JADE, 7),
        _c(Suit.SWORD, 7),
        _c(Suit.PAGODA, 7),
        _c(Suit.STAR, 7),
    ]
    top_card = _c(Suit.JADE, 10)
    state = _state(
        {
            0: frozenset({_c(Suit.STAR, 14)}),  # leader, already played top
            2: frozenset(sevens + [_c(Suit.STAR, 8)]),
            1: frozenset({_c(Suit.STAR, 9)}),
            3: frozenset({_c(Suit.STAR, 11)}),
        },
        top=Single(top_card),
        current_player=1,
    )
    bomb = FourOfAKindBomb(*sevens)
    next_state, _, _, _ = step(state, BombInterrupt(player=2, bomb=bomb))
    assert set(sevens).issubset(next_state.public.played_cards_this_round)


def test_finalise_round_resets_played_cards():
    # Set up post-round state with accumulated played cards and hands such
    # that the round is "done" (3 of 4 out). `_finalise_round` must wipe
    # played_cards_this_round so the next round starts clean.
    public = PublicState(
        current_player=0,
        hand_sizes=(0, 0, 0, 2),
        scores=(0, 0),
        trick=Trick.empty(),
        out_order=(0, 1, 2),
        played_cards_this_round=frozenset({_c(Suit.JADE, 7), DOG, DRAGON}),
    )
    hands = (frozenset(), frozenset(), frozenset(),
             frozenset({_c(Suit.STAR, 4), _c(Suit.STAR, 5)}))
    state = GameState(hands=hands, public=public)
    finalised = _finalise_round(state)
    assert finalised.public.played_cards_this_round == frozenset()


def test_phoenix_in_play_accumulates_as_phoenix():
    # Phoenix substitutes for a rank in a Pair; _cards_in returns PHOENIX
    # itself, so the seen-cards view records the special card (not the
    # rank it stood in for).
    seven = _c(Suit.JADE, 7)
    state = _state(
        {
            0: frozenset({seven, PHOENIX}),
            1: frozenset({_c(Suit.STAR, 10), _c(Suit.JADE, 10)}),
            2: frozenset({_c(Suit.PAGODA, 11)}),
            3: frozenset({_c(Suit.SWORD, 12)}),
        },
    )
    next_state, _, _, _ = step(state, Pair(seven, PHOENIX))
    assert PHOENIX in next_state.public.played_cards_this_round
    assert seven in next_state.public.played_cards_this_round
