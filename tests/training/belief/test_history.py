"""Belief History block (ADR-0028): tier dims, decline/lead/pressure/play-time
accumulation, and the 307-dim feature extension."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tichu_engine.cards import Card, Suit
from tichu_engine.combinations import FourOfAKindBomb, Pair, Single
from tichu_training.card_slots import card_slot
from tichu_training.belief.history import (
    BELIEF_FEATURE_DIM,
    HISTORY_DIM,
    TIER_DIMS,
    HistoryAccumulator,
    _combo_type_and_rank,
    belief_continuous_columns,
    extend_features,
)
from tichu_training.featurizer import CONTINUOUS_FEATURE_COLUMNS, FEATURIZER_OUTPUT_DIM


def _pass(seat, top, leader=0):
    state = SimpleNamespace(public=SimpleNamespace(
        trick=SimpleNamespace(top_combination=top, leader=leader)))
    return SimpleNamespace(player=seat, kind="pass", cards=None), state


def _play(seat, cards, leader):
    state = SimpleNamespace(public=SimpleNamespace(
        trick=SimpleNamespace(top_combination=None, leader=leader)))
    return SimpleNamespace(player=seat, kind="play", cards=cards), state


def test_tier_dims_are_nested_prefixes():
    # Belief tiers are defined relative to the policy Feature Vector, so they
    # track FEATURIZER_OUTPUT_DIM (591 at v6). NB: post-v6 the B_core history
    # channels (declined/lead/pass) now ALSO live in the policy features —
    # a redundancy the deferred belief-rung redesign must address (ADR-0038).
    assert HISTORY_DIM == 83
    assert BELIEF_FEATURE_DIM == FEATURIZER_OUTPUT_DIM + HISTORY_DIM
    assert TIER_DIMS == {
        "A": FEATURIZER_OUTPUT_DIM,
        "B_core": FEATURIZER_OUTPUT_DIM + 27,
        "B_full": FEATURIZER_OUTPUT_DIM + HISTORY_DIM,
    }


def test_belief_continuous_columns_cover_policy_plus_all_history():
    cols = belief_continuous_columns()
    # policy continuous + every History column (appended after the policy block).
    history_cols = range(FEATURIZER_OUTPUT_DIM, FEATURIZER_OUTPUT_DIM + HISTORY_DIM)
    assert set(cols) == set(CONTINUOUS_FEATURE_COLUMNS) | set(history_cols)
    assert len(cols) == len(CONTINUOUS_FEATURE_COLUMNS) + HISTORY_DIM


def test_combo_type_and_rank():
    assert _combo_type_and_rank(Single(card=Card(Suit.JADE, 9))) == (0, 9)
    assert _combo_type_and_rank(
        Pair(Card(Suit.JADE, 7), Card(Suit.SWORD, 7))) == (1, 7)
    # Bombs are excluded from the decline channel.
    bomb = FourOfAKindBomb(
        Card(Suit.JADE, 5), Card(Suit.SWORD, 5),
        Card(Suit.PAGODA, 5), Card(Suit.STAR, 5),
    )
    assert _combo_type_and_rank(bomb) is None


def test_pass_records_declined_top_for_acting_seat():
    acc = HistoryAccumulator()
    pa, st = _pass(seat=1, top=Pair(Card(Suit.JADE, 7), Card(Suit.SWORD, 7)))
    acc.update(pa, st)
    assert acc.declined_max[1][1] == 7   # type 1 = Pair
    assert acc.passes[1] == 1 and acc.decisions[1] == 1
    # A later, lower decline does not lower the running max.
    pa2, st2 = _pass(seat=1, top=Pair(Card(Suit.JADE, 4), Card(Suit.SWORD, 4)))
    acc.update(pa2, st2)
    assert acc.declined_max[1][1] == 7


def test_block_reads_opponents_in_relative_order():
    acc = HistoryAccumulator()
    # Seat 2 declines a Single-10.
    pa, st = _pass(seat=2, top=Single(card=Card(Suit.JADE, 10)))
    acc.update(pa, st)
    # From seat 0's view, opponents are next=1, partner=2, previous=3.
    block = acc.block(seat=0)
    assert block.shape == (HISTORY_DIM,)
    # H1 is 3 opp x 6 types; partner (opp_idx 1), Single (type 0) -> index 6.
    assert block[6] == 10 / 14.0
    # next (opp_idx 0) declined nothing.
    assert block[0] == 0.0


def test_play_lead_and_play_time():
    acc = HistoryAccumulator()
    c = Card(Suit.JADE, 5)
    pa, st = _play(seat=3, cards=[c], leader=None)   # leader None -> a lead
    acc.update(pa, st)
    assert acc.lead_count[3] == 1
    assert acc.lead_min_single[3] == 5
    assert acc.play_idx == 1
    assert acc.card_play_time[card_slot(c)] == 1


def test_extend_features_appends_history_and_preserves_policy_prefix():
    acc = HistoryAccumulator()
    policy = np.arange(FEATURIZER_OUTPUT_DIM, dtype=np.float32)
    out = extend_features(policy, acc, seat=0)
    assert out.shape == (BELIEF_FEATURE_DIM,)
    np.testing.assert_array_equal(out[:FEATURIZER_OUTPUT_DIM], policy)
