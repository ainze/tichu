"""Schupfen training dataset + loop behaviour.

Q3/Q4 of the schupfen design pass: structured 3-tuple target, hand-masked.
The synthetic dataset is the smoke-path equivalent of `SyntheticCallDataset`
— deterministic, no archive, no parquet — and is the source the CLI smoke
config trains against.
"""

import numpy as np
import torch

from tichu_engine.cards import Card, Suit
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.bc.schupfen_training import (
    SchupfenExample,
    SyntheticSchupfenDataset,
    _schupfen_examples_for_round,
    train_one_schupfen_epoch,
)
from tichu_training.bsw.records import ParsedAction, ParsedRound
from tichu_training.card_slots import card_slot
from tichu_training.featurizer import SECTION_DIMS


def test_synthetic_examples_have_three_distinct_target_slots_in_the_hand_mask():
    ds = SyntheticSchupfenDataset(seed=0, n_examples=32, feature_dim=32, skill_buckets=10)
    examples = list(ds)
    assert len(examples) == 32
    for ex in examples:
        assert isinstance(ex, SchupfenExample)
        # target is a 3-tuple (to_next, to_partner, to_previous) of slot ids
        assert ex.target.shape == (3,)
        assert ex.target.dtype.kind == "i"
        assert len(set(int(s) for s in ex.target)) == 3, "schupfen sends 3 distinct cards"
        # hand_mask is a length-56 multi-hot; every targeted slot must be in the hand
        assert ex.hand_mask.shape == (56,)
        for slot in ex.target:
            assert ex.hand_mask[int(slot)] == 1.0, (
                f"target slot {int(slot)} not in hand_mask — unlearnable example"
            )


def test_train_one_schupfen_epoch_reduces_loss_on_synthetic_data(tmp_path):
    """One pass of train_one_schupfen_epoch over a fixed-seed synthetic
    set should drive the loss meaningfully down — proving the 3-head
    masked-CE loss + optimizer wiring actually trains the network."""
    torch.manual_seed(0)
    examples = list(SyntheticSchupfenDataset(seed=0, n_examples=64, feature_dim=32))
    net = SchupfenNetwork(feature_dim=32, skill_buckets=10, skill_dim=4, hidden=32)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
    log_path = tmp_path / "schupfen_step.csv"

    initial_loss = train_one_schupfen_epoch(
        net, examples, optimizer, batch_size=16, log_path=log_path,
    )
    for _ in range(19):
        final_loss = train_one_schupfen_epoch(
            net, examples, optimizer, batch_size=16, log_path=log_path,
        )
    assert final_loss < initial_loss * 0.5, (
        f"loss did not converge: initial={initial_loss:.3f} final={final_loss:.3f}"
    )


# --- Slice 5 — Parquet adapter per-round behaviour --------------------------

def _seat0_hand_14() -> frozenset:
    """A deterministic 14-card hand for seat 0. Only the slot identities matter."""
    return frozenset({Card(Suit.JADE, r) for r in range(2, 15)} | {Card(Suit.SWORD, 2)})


def _zero_hand() -> frozenset:
    """A 14-card hand for seats 1..3 — distinct from seat 0's, just shape-valid."""
    return frozenset({Card(Suit.PAGODA, r) for r in range(2, 15)} | {Card(Suit.STAR, 2)})


def _build_parsed_round(*, grand_callers, tichu_callers, seat0_pass):
    hands = (_seat0_hand_14(), _zero_hand(), _zero_hand(), _zero_hand())
    schupfen = (
        ParsedAction(
            player=0, kind="schupfen",
            schupfen_to_next=seat0_pass[0],
            schupfen_to_partner=seat0_pass[1],
            schupfen_to_previous=seat0_pass[2],
        ),
        ParsedAction(player=1, kind="schupfen",
                     schupfen_to_next=Card(Suit.PAGODA, 5),
                     schupfen_to_partner=Card(Suit.PAGODA, 6),
                     schupfen_to_previous=Card(Suit.PAGODA, 7)),
        ParsedAction(player=2, kind="schupfen",
                     schupfen_to_next=Card(Suit.PAGODA, 8),
                     schupfen_to_partner=Card(Suit.PAGODA, 9),
                     schupfen_to_previous=Card(Suit.PAGODA, 10)),
        ParsedAction(player=3, kind="schupfen",
                     schupfen_to_next=Card(Suit.PAGODA, 11),
                     schupfen_to_partner=Card(Suit.PAGODA, 12),
                     schupfen_to_previous=Card(Suit.PAGODA, 13)),
    )
    return ParsedRound(
        round_index=0,
        pre_deal_hands=((), (), (), ()),
        start_hands=hands,
        grand_tichu_callers=frozenset(grand_callers),
        tichu_callers=frozenset(tichu_callers),
        schupfen=schupfen,
        plays=(),
        ergebnis=(0, 0),
        handles=("Alice", "Bob", "Carol", "Dave"),
    )


def _grand_tichu_callers_section(features):
    """Extract the 4-dim grand_tichu_callers section from a feature vector."""
    offset = sum(
        v for k, v in SECTION_DIMS.items()
        if k in ("own_hand", "hand_sizes", "team_scores",
                 "round_points", "out_order", "tichu_callers")
    )
    return features[offset:offset + 4]


def _tichu_callers_section(features):
    offset = sum(
        v for k, v in SECTION_DIMS.items()
        if k in ("own_hand", "hand_sizes", "team_scores",
                 "round_points", "out_order")
    )
    return features[offset:offset + 4]


def test_per_round_emits_four_examples_one_per_seat():
    seat0_cards = (Card(Suit.JADE, 14), Card(Suit.JADE, 13), Card(Suit.JADE, 12))
    parsed = _build_parsed_round(grand_callers=set(), tichu_callers=set(),
                                 seat0_pass=seat0_cards)
    examples = _schupfen_examples_for_round(parsed, skill_lookup={}, neutral_decile=10,
                                            sample_weight=1.0)
    assert len(examples) == 4


def test_per_round_seat0_target_matches_recorded_schupfen_cards():
    seat0_cards = (Card(Suit.JADE, 14), Card(Suit.JADE, 13), Card(Suit.JADE, 12))
    parsed = _build_parsed_round(grand_callers=set(), tichu_callers=set(),
                                 seat0_pass=seat0_cards)
    examples = _schupfen_examples_for_round(parsed, skill_lookup={}, neutral_decile=10,
                                            sample_weight=1.0)
    seat0 = next(e for e in examples if e.skill_decile is not None)
    # seat ordering preserved → first example is seat 0
    seat0 = examples[0]
    expected = np.array([card_slot(c) for c in seat0_cards], dtype=np.int64)
    assert (seat0.target == expected).all()


def test_per_round_features_encode_grand_tichu_callers_and_leave_tichu_callers_empty():
    """The Q3 bug catch: at schupfen time, grand-tichu calls are public
    (round-order: Grand-Tichu → Schupfen → Tichu → Tricks), so they MUST
    appear in the feature vector. Tichu calls happen later in the
    pipeline's timeline (ADR-0018: featurises at first-non-pass-play),
    so the section must be empty at schupfen time."""
    parsed = _build_parsed_round(
        grand_callers={1, 2},
        tichu_callers={0},   # would be a bug if this leaks
        seat0_pass=(Card(Suit.JADE, 14), Card(Suit.JADE, 13), Card(Suit.JADE, 12)),
    )
    examples = _schupfen_examples_for_round(parsed, skill_lookup={}, neutral_decile=10,
                                            sample_weight=1.0)
    for ex in examples:
        grand_sec = _grand_tichu_callers_section(ex.features)
        tichu_sec = _tichu_callers_section(ex.features)
        # grand-tichu callers populated as {1, 2}
        assert grand_sec[0] == 0.0
        assert grand_sec[1] == 1.0
        assert grand_sec[2] == 1.0
        assert grand_sec[3] == 0.0
        # tichu callers ALL zero — empty by design
        assert (tichu_sec == 0.0).all(), (
            "tichu_callers must be empty at schupfen time (ADR-0018)"
        )
