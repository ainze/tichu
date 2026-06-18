"""Featurizer (v4): shape, dtype, purity, version pinning."""

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
    TRICK_TOP_COMBO_SUBFIELDS,
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


def test_version_is_pinned_to_v6():
    # v6 (ADR-0038): played_by + schupfen_received + B-core + trick_leader.
    assert FEATURIZER_VERSION == "v6"


def test_dropped_sections_are_absent():
    """play_history and phoenix_played stay dropped (ADR-0015); seen_cards is
    replaced by per-player played_by (ADR-0038). schupfen_received is *back*
    self-only at v6 — ADR-0015's rules-violation finding was corrected. v4
    still rejects phoenix_position and suit inside trick_top_combo (ADR-0017)."""
    for dropped in ("play_history", "phoenix_played", "seen_cards"):
        assert dropped not in SECTION_DIMS, (
            f"{dropped} is dropped — re-adding it needs an ADR "
            f"and a featurizer version bump"
        )
    # schupfen_received is legitimately present again at v6 (self-only).
    assert "schupfen_received" in SECTION_DIMS
    for forbidden_subfield in ("phoenix_position", "suit"):
        assert forbidden_subfield not in TRICK_TOP_COMBO_SUBFIELDS, (
            f"{forbidden_subfield} was rejected in ADR-0017 — re-adding "
            f"it needs an ADR and a featurizer version bump"
        )


def test_v6_total_dim_is_591():
    assert FEATURIZER_OUTPUT_DIM == 591


def test_trick_top_combo_section_is_50_dims():
    assert SECTION_DIMS["trick_top_combo"] == 50


def test_trick_top_combo_subfields_sum_to_50():
    assert sum(TRICK_TOP_COMBO_SUBFIELDS.values()) == 50
    assert TRICK_TOP_COMBO_SUBFIELDS == {
        "intent_kind": 8,
        "primary_rank": 15,
        "secondary_rank": 13,
        "length": 13,
        "phoenix_used": 1,
    }


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


def _trick_top_combo_section(feat: np.ndarray) -> np.ndarray:
    """Slice the 50-dim trick_top_combo section out of a featurised state.

    Slice is computed off SECTION_DIMS rather than hard-coded so the test
    survives section-order tweaks; the section's *layout* (intent_kind,
    primary_rank, ...) is what we actually assert against."""
    cursor = 0
    for name, dim in SECTION_DIMS.items():
        if name == "trick_top_combo":
            return feat[cursor:cursor + dim]
        cursor += dim
    raise AssertionError("trick_top_combo not in SECTION_DIMS")


def _section_offsets() -> dict[str, slice]:
    """Return per-subfield slices inside the 50-dim trick_top_combo section."""
    offs: dict[str, slice] = {}
    c = 0
    for name, dim in TRICK_TOP_COMBO_SUBFIELDS.items():
        offs[name] = slice(c, c + dim)
        c += dim
    return offs


def _featurize_with_top(top_combo) -> np.ndarray:
    """Build a minimal PrivateState whose trick has the given top combo."""
    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    return featurize(PrivateState(
        player=0, hand=hand,
        public=PublicState(
            current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
            trick=Trick(
                plays=(Play(player=1, combination=top_combo),),
                leader=1,
            ),
        ),
    ))


def test_trick_top_combo_section_is_all_zero_when_leading():
    """No current top → the 50-dim section is exactly zero (ADR-0017).

    Distinguishability from "non-empty top" is via intent_kind bits, which
    are populated only when there is a top combination."""
    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    leading = PrivateState(
        player=0, hand=hand,
        public=PublicState(
            current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
            trick=Trick.empty(),
        ),
    )
    section = _trick_top_combo_section(featurize(leading))
    assert section.shape == (50,)
    assert section.sum() == 0.0


def test_trick_top_single_natural_rank_sets_intent_kind_and_primary_rank():
    """Single(Card(JADE, 7)) → intent_kind[Single]=1, primary_rank[6]=1."""
    section = _trick_top_combo_section(_featurize_with_top(Single(Card(Suit.JADE, 7))))
    offs = _section_offsets()
    # intent_kind: Single = slot 0
    assert section[offs["intent_kind"]][0] == 1.0
    assert section[offs["intent_kind"]].sum() == 1.0
    # primary_rank: rank 7 → slot 6 (slot 0 = Mahjong, slot N-1 = rank N for N in 2..14)
    assert section[offs["primary_rank"]][6] == 1.0
    assert section[offs["primary_rank"]].sum() == 1.0
    # secondary_rank, length, phoenix_used: all zero
    assert section[offs["secondary_rank"]].sum() == 0.0
    assert section[offs["length"]].sum() == 0.0
    assert section[offs["phoenix_used"]].sum() == 0.0
    # Total = 2 bits.
    assert section.sum() == 2.0


def test_trick_top_single_dragon_sets_primary_rank_slot_14():
    """Dragon → primary_rank[14], no phoenix_used."""
    section = _trick_top_combo_section(_featurize_with_top(Single(DRAGON)))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][0] == 1.0
    assert section[offs["primary_rank"]][14] == 1.0
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["phoenix_used"]].sum() == 0.0
    assert section.sum() == 2.0


def test_trick_top_single_mahjong_sets_primary_rank_slot_0():
    """Mahjong (rank 1) → primary_rank[0], no phoenix_used."""
    section = _trick_top_combo_section(_featurize_with_top(Single(MAHJONG)))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][0] == 1.0
    assert section[offs["primary_rank"]][0] == 1.0
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["phoenix_used"]].sum() == 0.0
    assert section.sum() == 2.0


def test_trick_top_single_phoenix_as_lead_is_rank_1_with_phoenix_used():
    """Phoenix played alone with no top → engine rank 1.5 → encoded as
    primary_rank[0] (the rank-1 / Mahjong slot) + phoenix_used."""
    section = _trick_top_combo_section(_featurize_with_top(Single(PHOENIX)))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][0] == 1.0
    assert section[offs["primary_rank"]][0] == 1.0
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["phoenix_used"]][0] == 1.0
    assert section.sum() == 3.0


def test_trick_top_single_phoenix_following_rank_n_is_rank_n_with_phoenix_used():
    """Phoenix played onto a rank-N single (engine rank N+0.5) → encoded
    as primary_rank[N-1] + phoenix_used. Tests N=7."""
    section = _trick_top_combo_section(
        _featurize_with_top(Single.phoenix_following(top_rank=7)),
    )
    offs = _section_offsets()
    assert section[offs["intent_kind"]][0] == 1.0
    assert section[offs["primary_rank"]][6] == 1.0  # rank 7 → slot 6
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["phoenix_used"]][0] == 1.0
    assert section.sum() == 3.0


def test_trick_top_pair_natural():
    """Pair of 7s natural → intent_kind[Pair]=1, primary_rank[6]=1, no phoenix."""
    section = _trick_top_combo_section(
        _featurize_with_top(Pair(Card(Suit.JADE, 7), Card(Suit.SWORD, 7))),
    )
    offs = _section_offsets()
    assert section[offs["intent_kind"]][1] == 1.0  # Pair = slot 1
    assert section[offs["intent_kind"]].sum() == 1.0
    assert section[offs["primary_rank"]][6] == 1.0
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["phoenix_used"]].sum() == 0.0
    assert section.sum() == 2.0


def test_trick_top_pair_with_phoenix():
    """Pair of 9s with phoenix → intent_kind[Pair]=1, primary_rank[8]=1, phoenix_used=1."""
    section = _trick_top_combo_section(
        _featurize_with_top(Pair(Card(Suit.JADE, 9), PHOENIX)),
    )
    offs = _section_offsets()
    assert section[offs["intent_kind"]][1] == 1.0
    assert section[offs["primary_rank"]][8] == 1.0  # rank 9 → slot 8 (slot N-1 for rank N)
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["phoenix_used"]][0] == 1.0
    assert section.sum() == 3.0


def test_trick_top_full_house_phoenix_in_triple():
    """FullHouse triple=10 pair=5 with phoenix in triple → intent_kind[FullHouse]=1,
    primary_rank[9]=1 (rank 10), secondary_rank[3]=1 (rank 5, slot R-2), phoenix_used=1."""
    from tichu_engine.combinations import FullHouse, Triple
    triple = Triple(Card(Suit.JADE, 10), Card(Suit.SWORD, 10), PHOENIX)
    pair = Pair(Card(Suit.JADE, 5), Card(Suit.SWORD, 5))
    section = _trick_top_combo_section(_featurize_with_top(FullHouse(triple, pair)))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][3] == 1.0  # FullHouse = slot 3
    assert section[offs["intent_kind"]].sum() == 1.0
    assert section[offs["primary_rank"]][9] == 1.0  # triple_rank 10 → slot 9
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["secondary_rank"]][3] == 1.0  # pair_rank 5 → slot 3 (slot R-2)
    assert section[offs["secondary_rank"]].sum() == 1.0
    assert section[offs["phoenix_used"]][0] == 1.0
    assert section[offs["length"]].sum() == 0.0
    assert section.sum() == 4.0


def test_trick_top_full_house_phoenix_in_pair():
    """FullHouse triple=8 pair=12 with phoenix in pair → phoenix_used=1, both rank slots set."""
    from tichu_engine.combinations import FullHouse, Triple
    triple = Triple(Card(Suit.JADE, 8), Card(Suit.SWORD, 8), Card(Suit.PAGODA, 8))
    pair = Pair(Card(Suit.JADE, 12), PHOENIX)
    section = _trick_top_combo_section(_featurize_with_top(FullHouse(triple, pair)))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][3] == 1.0
    assert section[offs["primary_rank"]][7] == 1.0   # triple_rank 8 → slot 7
    assert section[offs["secondary_rank"]][10] == 1.0  # pair_rank 12 → slot 10
    assert section[offs["phoenix_used"]][0] == 1.0
    assert section.sum() == 4.0


def test_trick_top_triple_with_phoenix():
    """Triple of 11s with phoenix → intent_kind[Triple]=1, primary_rank[10]=1, phoenix_used=1."""
    from tichu_engine.combinations import Triple
    section = _trick_top_combo_section(
        _featurize_with_top(Triple(Card(Suit.JADE, 11), Card(Suit.SWORD, 11), PHOENIX)),
    )
    offs = _section_offsets()
    assert section[offs["intent_kind"]][2] == 1.0  # Triple = slot 2
    assert section[offs["intent_kind"]].sum() == 1.0
    assert section[offs["primary_rank"]][10] == 1.0  # rank 11 → slot 10
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["phoenix_used"]][0] == 1.0
    assert section[offs["secondary_rank"]].sum() == 0.0
    assert section[offs["length"]].sum() == 0.0
    assert section.sum() == 3.0


def test_trick_top_pair_step_natural():
    """PairStep start=4 length=3 (pairs of 4, 5, 6) →
    intent_kind[PairStep]=1, primary_rank[3]=1, length[1]=1 (length 3 → slot 1)."""
    from tichu_engine.combinations import PairStep
    pairs = (
        Pair(Card(Suit.JADE, 4), Card(Suit.SWORD, 4)),
        Pair(Card(Suit.JADE, 5), Card(Suit.SWORD, 5)),
        Pair(Card(Suit.JADE, 6), Card(Suit.SWORD, 6)),
    )
    section = _trick_top_combo_section(_featurize_with_top(PairStep(pairs)))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][4] == 1.0  # PairStep = slot 4
    assert section[offs["intent_kind"]].sum() == 1.0
    assert section[offs["primary_rank"]][3] == 1.0   # start_rank 4 → slot 3
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["length"]][1] == 1.0          # length 3 → slot 1 (slot L-2)
    assert section[offs["length"]].sum() == 1.0
    assert section[offs["phoenix_used"]].sum() == 0.0
    assert section[offs["secondary_rank"]].sum() == 0.0
    assert section.sum() == 3.0


def test_trick_top_straight_mahjong_led():
    """Mahjong-led straight 1..5 → primary_rank[0]=1 (Mahjong slot), length[3]=1 (length 5)."""
    from tichu_engine.combinations import Straight
    cards = (
        MAHJONG,
        Card(Suit.JADE, 2), Card(Suit.JADE, 3),
        Card(Suit.JADE, 4), Card(Suit.JADE, 5),
    )
    section = _trick_top_combo_section(_featurize_with_top(Straight(cards)))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][5] == 1.0  # Straight = slot 5
    assert section[offs["intent_kind"]].sum() == 1.0
    assert section[offs["primary_rank"]][0] == 1.0  # rank 1 → slot 0
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["length"]][3] == 1.0        # length 5 → slot 3 (L-2)
    assert section[offs["length"]].sum() == 1.0
    assert section[offs["phoenix_used"]].sum() == 0.0
    assert section.sum() == 3.0


def test_trick_top_straight_natural_with_phoenix():
    """Straight 7..11 with phoenix substituting for 9 →
    primary_rank[6]=1, length[3]=1, phoenix_used=1."""
    from tichu_engine.combinations import Straight
    cards = (
        Card(Suit.JADE, 7), Card(Suit.JADE, 8), PHOENIX,
        Card(Suit.JADE, 10), Card(Suit.JADE, 11),
    )
    section = _trick_top_combo_section(
        _featurize_with_top(Straight(cards, phoenix_as_rank=9)),
    )
    offs = _section_offsets()
    assert section[offs["intent_kind"]][5] == 1.0
    assert section[offs["primary_rank"]][6] == 1.0  # start 7 → slot 6
    assert section[offs["length"]][3] == 1.0
    assert section[offs["phoenix_used"]][0] == 1.0
    assert section.sum() == 4.0


def test_trick_top_pair_step_with_phoenix():
    """PairStep with phoenix in one of the pairs → phoenix_used=1."""
    from tichu_engine.combinations import PairStep
    pairs = (
        Pair(Card(Suit.JADE, 8), Card(Suit.SWORD, 8)),
        Pair(Card(Suit.JADE, 9), PHOENIX),
    )
    section = _trick_top_combo_section(_featurize_with_top(PairStep(pairs)))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][4] == 1.0
    assert section[offs["primary_rank"]][7] == 1.0   # start 8 → slot 7
    assert section[offs["length"]][0] == 1.0          # length 2 → slot 0
    assert section[offs["phoenix_used"]][0] == 1.0
    assert section.sum() == 4.0


def test_trick_top_four_bomb_never_sets_phoenix_or_length():
    """FourBomb of 8s → intent_kind[FourBomb]=1, primary_rank[7]=1; no phoenix
    (bombs can't carry it), no length (FourBomb has fixed count)."""
    from tichu_engine.combinations import FourOfAKindBomb
    bomb = FourOfAKindBomb(
        Card(Suit.JADE, 8), Card(Suit.SWORD, 8),
        Card(Suit.PAGODA, 8), Card(Suit.STAR, 8),
    )
    section = _trick_top_combo_section(_featurize_with_top(bomb))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][6] == 1.0   # FourBomb = slot 6
    assert section[offs["intent_kind"]].sum() == 1.0
    assert section[offs["primary_rank"]][7] == 1.0  # rank 8 → slot 7
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["secondary_rank"]].sum() == 0.0
    assert section[offs["length"]].sum() == 0.0
    assert section[offs["phoenix_used"]].sum() == 0.0
    assert section.sum() == 2.0


def test_trick_top_straight_flush_bomb_sets_kind_rank_and_length():
    """SFBomb start=4 length=5 (4..8 jade) → intent_kind[SFBomb]=1,
    primary_rank[3]=1, length[3]=1, no phoenix (SFBomb can't carry it),
    no suit (Q3: suit dropped — zero beat-relevant signal)."""
    from tichu_engine.combinations import StraightFlushBomb
    bomb = StraightFlushBomb(tuple(Card(Suit.JADE, r) for r in range(4, 9)))
    section = _trick_top_combo_section(_featurize_with_top(bomb))
    offs = _section_offsets()
    assert section[offs["intent_kind"]][7] == 1.0   # SFBomb = slot 7
    assert section[offs["intent_kind"]].sum() == 1.0
    assert section[offs["primary_rank"]][3] == 1.0  # start 4 → slot 3
    assert section[offs["primary_rank"]].sum() == 1.0
    assert section[offs["length"]][3] == 1.0        # length 5 → slot 3
    assert section[offs["length"]].sum() == 1.0
    assert section[offs["secondary_rank"]].sum() == 0.0
    assert section[offs["phoenix_used"]].sum() == 0.0
    assert section.sum() == 3.0


def test_featurize_handles_all_engine_phases(simple_phases):
    for state in simple_phases:
        feat = featurize(state)
        assert feat.shape == (FEATURIZER_OUTPUT_DIM,)


# ---- v6: section helpers + new sections ----


def _section(feat: np.ndarray, name: str) -> np.ndarray:
    """Slice a named section out of a featurised state via SECTION_DIMS."""
    cursor = 0
    for n, d in SECTION_DIMS.items():
        if n == name:
            return feat[cursor:cursor + d]
        cursor += d
    raise AssertionError(f"{name} not in SECTION_DIMS")


def _play_state(acting: int, leader: int) -> PrivateState:
    """A play-phase PrivateState whose current trick was led by `leader`."""
    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    return PrivateState(
        player=acting, hand=hand,
        public=PublicState(
            current_player=acting, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
            trick=Trick(
                plays=(Play(player=leader, combination=Single(Card(Suit.JADE, 7))),),
                leader=leader,
            ),
        ),
    )


def test_trick_leader_is_relative_seat_one_hot():
    """`trick_leader[4]` marks who led the current trick, in relative-seat
    order [self, next, partner, previous]. Empty trick → all-zero (ADR-0038)."""
    # acting=0: absolute leader L lands in relative slot (L - 0) % 4.
    for leader, slot in ((0, 0), (1, 1), (2, 2), (3, 3)):
        sec = _section(featurize(_play_state(acting=0, leader=leader)), "trick_leader")
        assert sec.shape == (4,)
        assert sec[slot] == 1.0
        assert sec.sum() == 1.0
    # Relativity: acting=2, leader=3 (the seat to acting's left) → slot 1.
    sec = _section(featurize(_play_state(acting=2, leader=3)), "trick_leader")
    assert sec[1] == 1.0
    assert sec.sum() == 1.0


def _played_by_planes(feat: np.ndarray) -> np.ndarray:
    return _section(feat, "played_by").reshape(4, 56)


def test_played_by_attributes_each_card_to_its_player_relative_seat():
    """`played_by[4][56]` puts each played card in the plane of the seat that
    played it, relative-seat ordered [self, next, partner, previous]. ADR-0038."""
    from tichu_training.card_slots import card_slot

    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    pub = PublicState(
        current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
        trick=Trick.empty(),
        played_cards_by_player=(
            frozenset({Card(Suit.SWORD, 3)}),   # self  → plane 0
            frozenset({PHOENIX}),               # next  → plane 1
            frozenset({DRAGON}),                # partner → plane 2
            frozenset({Card(Suit.SWORD, 5)}),   # previous → plane 3
        ),
    )
    planes = _played_by_planes(featurize(PrivateState(player=0, hand=hand, public=pub)))
    assert planes[0][card_slot(Card(Suit.SWORD, 3))] == 1.0
    assert planes[1][card_slot(PHOENIX)] == 1.0
    assert planes[2][card_slot(DRAGON)] == 1.0
    assert planes[3][card_slot(Card(Suit.SWORD, 5))] == 1.0
    assert planes.sum() == 4.0

    # Relativity: viewed from acting seat 2, a card played by seat 3 (next) → plane 1.
    pub2 = PublicState(
        current_player=2, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
        trick=Trick.empty(),
        played_cards_by_player=(frozenset(), frozenset(), frozenset(), frozenset({DRAGON})),
    )
    planes2 = _played_by_planes(featurize(PrivateState(player=2, hand=hand, public=pub2)))
    assert planes2[1][card_slot(DRAGON)] == 1.0
    assert planes2.sum() == 1.0


def test_declined_top_encodes_max_declined_rank_per_opponent_relative_seat():
    """`declined_top[3 opp × 6 types]` carries, per opponent (relative-seat
    [next, partner, previous]) and per non-bomb intent type, the max primary
    rank that opponent declined to beat, normalised rank/14. ADR-0028 / ADR-0038."""
    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    pub = PublicState(
        current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
        trick=Trick.empty(),
        declined_top_by_player=(
            (0, 0, 0, 0, 0, 0),    # self  (seat 0) — never read
            (13, 7, 0, 0, 0, 0),   # next  (seat 1): Single=13, Pair=7
            (0, 0, 0, 0, 0, 0),    # partner (seat 2): none
            (0, 0, 0, 0, 0, 10),   # previous (seat 3): Straight=10
        ),
    )
    sec = _section(featurize(PrivateState(player=0, hand=hand, public=pub)),
                   "declined_top").reshape(3, 6)
    assert sec[0][0] == pytest.approx(13 / 14)   # next, Single
    assert sec[0][1] == pytest.approx(7 / 14)    # next, Pair
    assert sec[1].sum() == 0.0                    # partner, none
    assert sec[2][5] == pytest.approx(10 / 14)   # previous, Straight


def test_lead_summary_encodes_tricks_led_and_lowest_single_lead_relative_seat():
    """`lead_summary[3 opp × 2]`: per opponent (relative-seat), tricks_led
    normalised by a fixed cap, and the lowest single-lead rank / 14. ADR-0028."""
    from tichu_training.featurizer import LEAD_TRICKS_CAP

    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    pub = PublicState(
        current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
        trick=Trick.empty(),
        lead_summary_by_player=(
            (0, 0),    # self
            (3, 4),    # next: led 3 tricks, lowest single lead rank 4
            (0, 0),    # partner
            (1, 14),   # previous: led 1 trick, lowest single lead rank 14
        ),
    )
    sec = _section(featurize(PrivateState(player=0, hand=hand, public=pub)),
                   "lead_summary").reshape(3, 2)
    assert sec[0][0] == pytest.approx(3 / LEAD_TRICKS_CAP)  # next, tricks_led
    assert sec[0][1] == pytest.approx(4 / 14)               # next, lowest single
    assert sec[1].sum() == 0.0                               # partner
    assert sec[2][0] == pytest.approx(1 / LEAD_TRICKS_CAP)  # previous, tricks_led
    assert sec[2][1] == pytest.approx(14 / 14)              # previous, lowest single


def test_pass_pressure_is_pass_fraction_per_opponent_relative_seat():
    """`pass_pressure[3 opp]`: fraction of each opponent's play decisions that
    were Passes, relative-seat ordered. No decisions → 0. ADR-0028."""
    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    pub = PublicState(
        current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
        trick=Trick.empty(),
        pass_stats_by_player=(
            (0, 0),   # self
            (3, 6),   # next: 3 of 6 → 0.5
            (0, 0),   # partner: no decisions → 0
            (2, 2),   # previous: 2 of 2 → 1.0
        ),
    )
    sec = _section(featurize(PrivateState(player=0, hand=hand, public=pub)),
                   "pass_pressure")
    assert sec.shape == (3,)
    assert sec[0] == pytest.approx(0.5)
    assert sec[1] == 0.0
    assert sec[2] == pytest.approx(1.0)


def test_schupfen_received_encodes_self_received_cards_by_direction():
    """`schupfen_received[3][56]` is self-only: the acting seat's three received
    cards, indexed by relative give-direction [from_next, from_partner,
    from_previous]. ADR-0038 (corrects ADR-0015's rules-violation finding)."""
    from tichu_training.card_slots import card_slot

    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {MAHJONG})
    s = PrivateState(
        player=0, hand=hand,
        public=PublicState(
            current_player=0, hand_sizes=(14, 14, 14, 14), scores=(0, 0),
            trick=Trick.empty(),
        ),
        schupfen_received=(Card(Suit.SWORD, 3), PHOENIX, Card(Suit.SWORD, 5)),
    )
    sec = _section(featurize(s), "schupfen_received").reshape(3, 56)
    assert sec[0][card_slot(Card(Suit.SWORD, 3))] == 1.0  # from next
    assert sec[1][card_slot(PHOENIX)] == 1.0              # from partner
    assert sec[2][card_slot(Card(Suit.SWORD, 5))] == 1.0  # from previous
    assert sec.sum() == 3.0


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
