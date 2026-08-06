"""Belief input spec: the vector the Belief Model consumes IS the policy
Feature Vector (ADR-0028 B-core, absorbed by featurizer v6 in ADR-0038).

These are dimension pins. A future featurizer bump must fail HERE — loudly and
in one place — rather than silently mis-shaping belief inputs or leaving a
stale suffix appended to a wider policy vector, which is exactly what happened
when v6 widened the featurizer 224 -> 591 under the old History-block emit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tichu_training.belief.belief_materialised import (
    BeliefBundleWriter, MemmapBeliefDataset,
)
from tichu_training.belief.emit import belief_examples_for_round
from tichu_training.belief.input_spec import (
    BELIEF_FEATURE_DIM, belief_continuous_columns,
)
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.replay import replay_round
from tichu_training.featurizer import (
    CONTINUOUS_FEATURE_COLUMNS, FEATURIZER_OUTPUT_DIM, SECTION_DIMS,
    SECTION_OFFSETS, featurize,
)


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"
_PLAY_KINDS = {"play", "pass"}


def test_belief_feature_dim_is_exactly_the_featurizer_output_dim():
    """No History suffix, no tier prefix — the belief input is the policy
    Feature Vector, whole. Adjust this pin ONLY alongside a deliberate,
    documented belief-input change (and a BELIEF_INPUT_VERSION bump)."""
    assert BELIEF_FEATURE_DIM == FEATURIZER_OUTPUT_DIM


def test_belief_continuous_columns_are_the_policy_continuous_columns():
    assert belief_continuous_columns() == tuple(CONTINUOUS_FEATURE_COLUMNS)
    assert max(belief_continuous_columns()) < BELIEF_FEATURE_DIM


def test_emitted_belief_features_have_the_pinned_width():
    """End-to-end: the replay emit produces exactly BELIEF_FEATURE_DIM columns
    and they are the featurizer's own output, unmodified."""
    seen = 0
    for tch in sorted(_SAMPLES.glob("*.tch")):
        game = parse_tch(tch.read_bytes().decode("utf-8"), game_id=tch.stem)
        for rnd in game.rounds:
            replay = replay_round(rnd)
            if replay.final_state is None:
                continue
            examples = belief_examples_for_round(rnd, replay)
            expected = [
                (pa.player, ps)
                for (pa, _c), ps in zip(replay.decisions, replay.pre_decision_states)
                if ps is not None and pa.kind in _PLAY_KINDS and 0 <= pa.player < 4
            ]
            for ex, (seat, ps) in zip(examples, expected):
                assert ex.features.shape == (BELIEF_FEATURE_DIM,)
                np.testing.assert_array_equal(
                    ex.features, featurize(ps.private_view(seat)),
                )
                seen += 1
    assert seen > 0, "no belief examples produced from sample data — fixture broken"


def test_bcore_history_channels_come_from_the_featurizer_not_a_suffix():
    """The ADR-0028 B-core channels live INSIDE the emitted vector, as featurizer
    sections, not appended after it. Guards against the duplication the
    History-block emit reintroduced at v6.

    Containment is the invariant, not position. Through v6 B-core happened to
    close the vector, and this test asserted that — an incidental layout fact. v7
    (ADR-0044) appends the Rich History Block after it, so the positional form
    would fail while the behaviour it guards is untouched.
    """
    bcore_start = SECTION_OFFSETS["declined_top"]
    bcore_end = (
        SECTION_OFFSETS["pass_pressure"] + SECTION_DIMS["pass_pressure"]
    )
    assert bcore_end - bcore_start == 18 + 6 + 3   # H1 + H2 + H3, contiguous
    assert bcore_end <= FEATURIZER_OUTPUT_DIM      # inside the vector, not a suffix


def test_bundle_manifest_records_the_pinned_dim(tmp_path):
    """The materialised bundle stamps the same width, so a bundle built under a
    different belief input cannot be read back silently."""
    rng = np.random.default_rng(0)
    from tichu_training.belief.dataset import BeliefExample

    cont = np.asarray(belief_continuous_columns(), dtype=np.intp)
    examples = []
    for _ in range(4):
        feats = (rng.random(BELIEF_FEATURE_DIM) < 0.5).astype(np.float32)
        feats[cont] = rng.standard_normal(cont.size).astype(np.float32)
        labels = (rng.random((3, 56)) < 0.25).astype(np.float32)
        mask = np.broadcast_to(labels.any(axis=0), (3, 56)).copy()
        examples.append(BeliefExample(features=feats, labels=labels, mask=mask))

    out = tmp_path / "belief"
    writer = BeliefBundleWriter(out)
    writer.add_many(examples)
    writer.close()

    ds = MemmapBeliefDataset(out)
    assert ds.feature_dim == BELIEF_FEATURE_DIM
    for got, want in zip(ds, examples):
        np.testing.assert_allclose(got.features, want.features, atol=1e-6)
