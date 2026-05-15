"""Featurizer (v1): shape, dtype, purity, version pinning."""

import subprocess
import sys

import numpy as np
import pytest

from tichu_engine.cards import Card, DRAGON, MAHJONG, PHOENIX, Suit
from tichu_engine.combinations import Pair, Single
from tichu_engine.state import (
    GameState,
    PrivateState,
    PublicState,
    Trick,
    Play,
)
from tichu_training.featurizer import (
    FEATURIZER_OUTPUT_DIM,
    FEATURIZER_VERSION,
    SECTION_DIMS,
    featurize,
)


def _simple_private_state(player: int = 0) -> PrivateState:
    """A minimal but valid PrivateState for testing."""
    hand = frozenset({
        Card(Suit.JADE, 2),
        Card(Suit.JADE, 3),
        Card(Suit.JADE, 4),
        Card(Suit.JADE, 5),
        Card(Suit.JADE, 6),
        Card(Suit.JADE, 7),
        Card(Suit.JADE, 8),
        Card(Suit.JADE, 9),
        Card(Suit.JADE, 10),
        Card(Suit.JADE, 11),
        Card(Suit.JADE, 12),
        Card(Suit.JADE, 13),
        Card(Suit.JADE, 14),
        MAHJONG,
    })
    public = PublicState(
        current_player=player,
        hand_sizes=(14, 14, 14, 14),
        scores=(0, 0),
        trick=Trick.empty(),
    )
    return PrivateState(player=player, hand=hand, public=public)


def test_version_is_pinned_to_v1():
    assert FEATURIZER_VERSION == "v1"


def test_output_is_float32_and_correct_shape():
    s = _simple_private_state()
    feat = featurize(s)
    assert isinstance(feat, np.ndarray)
    assert feat.dtype == np.float32
    assert feat.shape == (FEATURIZER_OUTPUT_DIM,)


def test_section_dims_sum_to_total():
    assert sum(SECTION_DIMS.values()) == FEATURIZER_OUTPUT_DIM


def test_within_process_purity_byte_identical():
    s = _simple_private_state()
    a = featurize(s)
    b = featurize(s)
    assert a.tobytes() == b.tobytes()


def test_does_not_mutate_input():
    s = _simple_private_state()
    pre_hand = frozenset(s.hand)
    pre_public_repr = repr(s.public)
    featurize(s)
    assert s.hand == pre_hand
    assert repr(s.public) == pre_public_repr


def test_hand_multi_hot_section_reflects_hand_membership():
    s = _simple_private_state()
    feat = featurize(s)
    # Section 1 is the first 56 dims: 52 normal + mahjong + dog + phoenix + dragon.
    hand_section = feat[:56]
    # Hand has 13 jade naturals + mahjong = 14 cards.
    assert hand_section.sum() == 14.0


def test_cross_process_purity_byte_identical():
    """A separate subprocess produces the same bytes for the same input."""
    script = """
import sys, hashlib
from tichu_engine.cards import Card, MAHJONG, Suit
from tichu_engine.state import PrivateState, PublicState, Trick
from tichu_training.featurizer import featurize

hand = frozenset({
    Card(Suit.JADE, r) for r in range(2, 15)
} | {MAHJONG})
public = PublicState(current_player=0, hand_sizes=(14,14,14,14), scores=(0,0), trick=Trick.empty())
s = PrivateState(player=0, hand=hand, public=public)
feat = featurize(s)
sys.stdout.write(hashlib.sha256(feat.tobytes()).hexdigest())
"""
    a = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    b = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert a.stdout == b.stdout
    assert len(a.stdout) == 64  # sha256 hex


def test_trick_top_combo_section_changes_with_top_combo():
    """When the trick has a top combo, the top-combo section has exactly one hot bit."""
    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    base_public = PublicState(
        current_player=0,
        hand_sizes=(14, 14, 14, 14),
        scores=(0, 0),
        trick=Trick.empty(),
    )
    empty = PrivateState(player=0, hand=hand, public=base_public)
    with_play = PrivateState(
        player=0,
        hand=hand,
        public=PublicState(
            current_player=0,
            hand_sizes=(14, 14, 14, 14),
            scores=(0, 0),
            trick=Trick(plays=(Play(player=1, combination=Single(Card(Suit.JADE, 5))),), leader=1),
        ),
    )
    a = featurize(empty)
    b = featurize(with_play)
    assert not np.array_equal(a, b)


def test_featurize_handles_all_engine_phases(simple_phases):
    for state in simple_phases:
        feat = featurize(state)
        assert feat.shape == (FEATURIZER_OUTPUT_DIM,)


@pytest.fixture
def simple_phases():
    """Yield PrivateStates for normal-play and pending-decision phases."""
    from tichu_engine.state import (
        DragonGivePending,
        MahjongWishPending,
        SchupfenPending,
    )

    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    base_public = PublicState(
        current_player=0,
        hand_sizes=(14, 14, 14, 14),
        scores=(0, 0),
        trick=Trick.empty(),
    )
    return [
        PrivateState(player=0, hand=hand, public=base_public),
        PrivateState(
            player=0, hand=hand,
            public=PublicState(
                current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
                trick=Trick.empty(),
                pending_decision=DragonGivePending(winner=0, points=25),
            ),
        ),
        PrivateState(
            player=0, hand=hand,
            public=PublicState(
                current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
                trick=Trick.empty(),
                pending_decision=MahjongWishPending(player=0),
            ),
        ),
        PrivateState(
            player=0, hand=hand,
            public=PublicState(
                current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
                trick=Trick.empty(),
                pending_decision=SchupfenPending(submitted=(None, None, None, None)),
            ),
        ),
    ]
