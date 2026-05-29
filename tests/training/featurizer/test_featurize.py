"""Featurizer (v3): shape, dtype, purity, version pinning."""

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


def test_version_is_pinned_to_v3():
    assert FEATURIZER_VERSION == "v3"


def test_dropped_v2_sections_are_absent():
    """v3 drops play_history (87% of v2), schupfen_received (rules
    violation — see ADR-0015), and phoenix_played (redundant with
    seen_cards). These keys must not reappear in SECTION_DIMS."""
    for dropped in ("play_history", "schupfen_received", "phoenix_played"):
        assert dropped not in SECTION_DIMS, (
            f"{dropped} was dropped in v3 — re-adding it needs an ADR "
            f"and a featurizer version bump"
        )


def test_v3_total_dim_is_1983():
    assert FEATURIZER_OUTPUT_DIM == 1983


def test_schupfen_pending_state_does_not_leak_per_giver_attribution():
    """ADR-0015 — schupfen_received was a rules violation that revealed
    which opponent gave the player which card. Regression: featurising a
    SchupfenPending state with all four submissions filled must produce
    the same output as featurising the same state with no submissions
    (modulo the phase one-hot). The featurizer must not encode anything
    about `pending.submitted`."""
    from tichu_engine.state import SchupfenPending

    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})

    # Empty schupfen pending (no one submitted yet).
    empty_pending = PrivateState(
        player=0, hand=hand,
        public=PublicState(
            current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
            trick=Trick.empty(),
            pending_decision=SchupfenPending(
                submitted=(None, None, None, None),
            ),
        ),
    )
    # Same state but with all four players having submitted distinctive
    # 3-card piles — the leaky v2 featurizer would have surfaced this
    # via the schupfen_received section.
    filled_pending = PrivateState(
        player=0, hand=hand,
        public=PublicState(
            current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
            trick=Trick.empty(),
            pending_decision=SchupfenPending(
                submitted=(
                    (Card(Suit.SWORD, 2), Card(Suit.SWORD, 3), Card(Suit.SWORD, 4)),
                    (Card(Suit.PAGODA, 5), Card(Suit.PAGODA, 6), Card(Suit.PAGODA, 7)),
                    (Card(Suit.STAR, 8), Card(Suit.STAR, 9), Card(Suit.STAR, 10)),
                    (Card(Suit.SWORD, 11), Card(Suit.SWORD, 12), Card(Suit.SWORD, 13)),
                ),
            ),
        ),
    )
    np.testing.assert_array_equal(
        featurize(empty_pending), featurize(filled_pending),
        err_msg="featurize() must not encode pending.submitted — that "
                "leaks per-giver attribution of received cards (ADR-0015)",
    )


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


# ---- v2: seen_cards section ----

def test_seen_cards_section_dim_is_56():
    assert SECTION_DIMS["seen_cards"] == 56


def _seen_cards_section(feat: np.ndarray) -> np.ndarray:
    """Slice the trailing seen_cards section from a featurised state."""
    start = FEATURIZER_OUTPUT_DIM - SECTION_DIMS["seen_cards"]
    return feat[start:]


def test_seen_cards_section_is_empty_when_nothing_played():
    s = _simple_private_state()
    feat = featurize(s)
    assert _seen_cards_section(feat).sum() == 0.0


def test_seen_cards_section_reflects_played_cards_this_round():
    played = frozenset({Card(Suit.JADE, 7), DRAGON, PHOENIX})
    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    public = PublicState(
        current_player=0,
        hand_sizes=(14, 14, 14, 14),
        scores=(0, 0),
        trick=Trick.empty(),
        played_cards_this_round=played,
    )
    s = PrivateState(player=0, hand=hand, public=public)
    feat = featurize(s)
    seen = _seen_cards_section(feat)
    assert seen.sum() == 3.0


def test_seen_cards_includes_specials_separately_from_naturals():
    # Phoenix and a natural 7 occupy different slots in the 56-dim section.
    played = frozenset({Card(Suit.JADE, 7), PHOENIX})
    hand = frozenset({Card(Suit.SWORD, r) for r in range(2, 15)} | {MAHJONG})
    public = PublicState(
        current_player=0,
        hand_sizes=(14, 14, 14, 14),
        scores=(0, 0),
        trick=Trick.empty(),
        played_cards_this_round=played,
    )
    s = PrivateState(player=0, hand=hand, public=public)
    seen = _seen_cards_section(featurize(s))
    # Exactly two distinct bits set.
    assert int(seen.sum()) == 2
    assert int((seen > 0).sum()) == 2


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
