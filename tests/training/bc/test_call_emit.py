"""Parity test for the share-not-mirror call emit (ADR-0020).

The consolidated `parse_bsw` pass feeds `tichu_examples_for_round` the round's
*full* replay; the standalone `ParquetCallDataset` feeds it an *early-stopped*
replay. Both must yield identical Tichu examples — the first non-Pass-Play
states are the same whether or not the replay continued past them. This is the
one place two different state-derivation routes must agree, so it is pinned
directly rather than relying on the round-trip tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tichu_training.bc.call_emit import tichu_examples_for_round
from tichu_training.bc.call_training import _all_seats_have_first_play
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.replay import replay_round


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def test_tichu_extraction_is_invariant_to_replay_truncation():
    seen_rounds = 0
    for tch in sorted(_SAMPLES.glob("*.tch")):
        game = parse_tch(tch.read_bytes().decode("utf-8"), game_id=tch.stem)
        for rnd in game.rounds:
            full = replay_round(rnd)
            if full.final_state is None:
                continue
            early = replay_round(rnd, early_stop=_all_seats_have_first_play)
            kw = dict(skill_lookup={}, neutral_decile=10, sample_weight=1.0)
            from_full = tichu_examples_for_round(rnd, full, **kw)
            from_early = tichu_examples_for_round(rnd, early, **kw)
            assert len(from_full) == len(from_early)
            for a, b in zip(from_full, from_early):
                np.testing.assert_array_equal(a.features, b.features)
                assert a.target == b.target
                assert a.skill_decile == b.skill_decile
                assert a.sample_weight == b.sample_weight
            seen_rounds += 1
    assert seen_rounds > 0, "no validated rounds in sample data — fixture broken"
