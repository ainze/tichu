"""Schupfen: the pre-play 3-card pass.

Each player passes one card to each of the three other players. Once all four
submissions are in, the engine applies the exchange and sets the current player
to the Mahjong holder for the first trick.
"""

import pytest

from tichu_engine.cards import Card, MAHJONG, Suit
from tichu_engine.engine import step
from tichu_engine.legality import SchupfenPass, legal_actions
from tichu_engine.state import (
    SchupfenPending,
    deal_for_schupfen,
)


def _c(suit: Suit, rank: int) -> Card:
    return Card(suit=suit, rank=rank)


# ---- Initial state ----

def test_deal_for_schupfen_starts_with_pending_decision():
    state = deal_for_schupfen(seed=0)
    assert isinstance(state.public.pending_decision, SchupfenPending)
    assert state.public.pending_decision.submitted == (None, None, None, None)
    assert state.public.current_player == 0
    assert state.public.hand_sizes == (14, 14, 14, 14)


def test_deal_for_schupfen_is_deterministic_with_seed():
    a = deal_for_schupfen(seed=42)
    b = deal_for_schupfen(seed=42)
    for player in range(4):
        assert a.hands[player] == b.hands[player]


# ---- Legal actions during schupfen ----

def test_legal_actions_during_schupfen_returns_three_card_permutations():
    state = deal_for_schupfen(seed=0)
    options = legal_actions(state)
    # 14 * 13 * 12 = 2184 valid orderings.
    assert len(options) == 14 * 13 * 12
    sample = next(iter(options))
    assert isinstance(sample, SchupfenPass)


# ---- Recording submissions ----

def test_schupfen_step_records_submission_and_advances_player():
    state = deal_for_schupfen(seed=0)
    hand0 = list(state.hands[0])
    action = SchupfenPass(to_next=hand0[0], to_partner=hand0[1], to_previous=hand0[2])
    next_state, _, _, _ = step(state, action)
    pending = next_state.public.pending_decision
    assert isinstance(pending, SchupfenPending)
    assert pending.submitted[0] == (hand0[0], hand0[1], hand0[2])
    assert pending.submitted[1] is None
    assert next_state.public.current_player == 1


# ---- Full exchange ----

def _submit_all_schupfens(state):
    """Helper: each player passes their first three cards (in iteration order)
    in (next, partner, previous) order."""
    for _ in range(4):
        current = state.public.current_player
        hand = list(state.hands[current])
        action = SchupfenPass(to_next=hand[0], to_partner=hand[1], to_previous=hand[2])
        state, _, _, _ = step(state, action)
    return state


def test_schupfen_after_all_submissions_clears_pending_and_sets_mahjong_leader():
    state = deal_for_schupfen(seed=0)
    state = _submit_all_schupfens(state)
    assert state.public.pending_decision is None
    mahjong_holder = next(p for p in range(4) if MAHJONG in state.hands[p])
    assert state.public.current_player == mahjong_holder


def test_schupfen_preserves_hand_sizes():
    state = deal_for_schupfen(seed=0)
    state = _submit_all_schupfens(state)
    assert state.public.hand_sizes == (14, 14, 14, 14)
    for p in range(4):
        assert len(state.hands[p]) == 14


def test_schupfen_exchanges_cards_between_players():
    state = deal_for_schupfen(seed=0)
    # Capture each player's chosen "to_next" card (will end up with player+1).
    sent_to_next = {}
    initial_hands = state.hands
    for _ in range(4):
        current = state.public.current_player
        hand = list(state.hands[current])
        sent_to_next[current] = hand[0]
        action = SchupfenPass(to_next=hand[0], to_partner=hand[1], to_previous=hand[2])
        state, _, _, _ = step(state, action)
    # After exchange, each player's hand contains the to_next card from their predecessor.
    for sender in range(4):
        recipient = (sender + 1) % 4
        assert sent_to_next[sender] in state.hands[recipient]
        assert sent_to_next[sender] not in state.hands[sender]


def test_schupfen_total_cards_preserved():
    state = deal_for_schupfen(seed=0)
    initial_all = set()
    for p in range(4):
        initial_all |= state.hands[p]
    state = _submit_all_schupfens(state)
    final_all = set()
    for p in range(4):
        final_all |= state.hands[p]
    assert initial_all == final_all  # no cards added or lost
