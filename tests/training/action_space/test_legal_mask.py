"""`legal_mask(decision_type, game_state, player) -> ndarray[K, bool]`.

For the play head (K=1809), the mask flags every legal ConcreteAction's
intent index from `legal_actions(state)` (or `legal_bomb_interrupts` when
the actor is not the current player). For wish (K=14) and dragon (K=2),
every slot is legal at the moment the engine is in the corresponding
pending decision.
"""

import numpy as np
import pytest

from tichu_engine.cards import Card, DRAGON, Suit
from tichu_engine.combinations import Pair, Single
from tichu_engine.state import (
    DragonGivePending, GameState, MahjongWishPending, PublicState, Trick,
)
from tichu_training.action_space import (
    Pass as IntentPass,
    PlayPair, PlaySingle,
    encode, legal_mask, play_intent_index,
)


def _public(current_player: int, hand_sizes, *,
            trick: Trick | None = None,
            pending=None,
            mahjong_wish: int | None = None) -> PublicState:
    return PublicState(
        current_player=current_player,
        hand_sizes=hand_sizes,
        scores=(0, 0),
        trick=trick or Trick.empty(),
        mahjong_wish=mahjong_wish,
        pending_decision=pending,
    )


def _game(hands, public) -> GameState:
    return GameState(hands=tuple(frozenset(h) for h in hands), public=public)


# --- play head -----------------------------------------------------------

def test_play_mask_leading_flags_every_legal_combo():
    """Leading player with a tiny hand of two 5s: the legal action set is
    {Single(5J), Single(5S), Pair(5J,5S)} — three intent indices."""
    hand = [Card(Suit.JADE, 5), Card(Suit.SWORD, 5)]
    public = _public(current_player=0, hand_sizes=(2, 0, 0, 0))
    state = _game([hand, [], [], []], public)

    mask = legal_mask("play", state, player=0)

    assert mask.dtype == bool
    assert mask.shape == (1809,)

    legal_idx = {
        encode(PlaySingle(suit="jade", rank=5)),
        encode(PlaySingle(suit="sword", rank=5)),
        encode(PlayPair(rank=5, with_phoenix=False)),
    }
    for idx in legal_idx:
        assert mask[idx], f"expected intent index {idx} to be flagged legal"
    # Total count: exactly three slots flagged.
    assert mask.sum() == len(legal_idx)


def test_play_mask_following_includes_pass_and_only_beating_combos():
    """Following with a top-Single-of-7: only Singles of rank > 7 are legal,
    plus Pass."""
    hand = [Card(Suit.JADE, 9), Card(Suit.SWORD, 3)]
    top_combo = Single(Card(Suit.STAR, 7))
    trick = Trick(
        plays=(__import__("tichu_engine.state", fromlist=["Play"]).Play(
            player=1, combination=top_combo,
        ),),
        leader=1,
    )
    public = _public(current_player=0, hand_sizes=(2, 0, 0, 0), trick=trick)
    state = _game([hand, [], [], []], public)

    mask = legal_mask("play", state, player=0)
    # Single(JADE, 9) and PASS are legal; Single(SWORD, 3) is not.
    assert mask[encode(PlaySingle(suit="jade", rank=9))]
    assert mask[encode(IntentPass())]
    assert not mask[encode(PlaySingle(suit="sword", rank=3))]


def test_play_mask_non_current_player_returns_only_bomb_interrupts():
    """If the parsed actor is not the engine's current player, the mask
    enumerates legal bomb interrupts only. With a small hand of two 5s
    (no bomb), the mask is all-zeros."""
    hand = [Card(Suit.JADE, 5), Card(Suit.SWORD, 5)]
    public = _public(current_player=1, hand_sizes=(2, 0, 0, 0))
    state = _game([hand, [], [], []], public)

    mask = legal_mask("play", state, player=0)
    assert mask.shape == (1809,)
    assert not mask.any(), "no bombs in hand → empty mask for the non-current player"


# --- wish head -----------------------------------------------------------

def test_wish_mask_all_14_legal_when_pending():
    public = _public(
        current_player=0, hand_sizes=(1, 0, 0, 0),
        pending=MahjongWishPending(player=0),
    )
    state = _game([[Card(Suit.JADE, 5)], [], [], []], public)
    mask = legal_mask("wish", state, player=0)
    assert mask.shape == (14,)
    assert mask.all()


# --- dragon_assignment head ---------------------------------------------

def test_dragon_mask_both_opponents_legal():
    public = _public(
        current_player=0, hand_sizes=(1, 0, 0, 0),
        pending=DragonGivePending(winner=0, points=25),
    )
    state = _game([[Card(Suit.JADE, 5)], [], [], []], public)
    mask = legal_mask("dragon_assignment", state, player=0)
    assert mask.shape == (2,)
    assert mask.all()


# --- dispatcher errors --------------------------------------------------

def test_unknown_decision_type_raises():
    public = _public(current_player=0, hand_sizes=(0, 0, 0, 0))
    state = _game([[], [], [], []], public)
    with pytest.raises(ValueError, match="unsupported decision_type"):
        legal_mask("schupfen", state, player=0)
    with pytest.raises(ValueError, match="unsupported decision_type"):
        legal_mask("call_tichu", state, player=0)
