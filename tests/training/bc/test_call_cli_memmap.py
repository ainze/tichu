"""`train_calls` reading the materialised Calls bundle (`dataset: memmap`).

The memmap path lets `train_calls` consume the packed calls bundle that the
consolidated parse pass writes ([ADR-0020](../../../docs/adr/0020-one-parse-pass-many-task-bundles.md))
instead of re-replaying the archive. The load-bearing detail is the call_type
key prefix: the bundle keys carry a `call_` prefix (`CALL_TYPE_ORDER`) while the
CLI tag layer does not, so the memmap branch must map `<call_type>` →
`call_<call_type>`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tichu_training.bc.call_materialised import MemmapCallDataset, materialise_calls
from tichu_training.bc.call_training import SyntheticCallDataset
from tichu_training.cli.train_calls import _TAG_TO_CALL_TYPE, _build_dataset, main


@pytest.fixture
def calls_bundle(tmp_path: Path) -> Path:
    """Tiny two-type calls bundle with distinguishable slice sizes (20 vs 15)
    so a wrong-slice read is observable by length alone."""
    streams = {
        "call_tichu": list(
            SyntheticCallDataset(seed=1, n_examples=20, binary_features=True)
        ),
        "call_grand_tichu": list(
            SyntheticCallDataset(seed=2, n_examples=15, binary_features=True)
        ),
    }
    out_dir = tmp_path / "calls_bundle"
    materialise_calls({k: iter(v) for k, v in streams.items()}, out_dir)
    return out_dir


def test_build_dataset_memmap_maps_tag_to_prefixed_call_type(calls_bundle):
    """`_build_dataset` for CLI tag 'tichu'/'grand' must read the
    `call_tichu`/`call_grand_tichu` slice — mirroring `main`'s tag mapping."""
    config = {"dataset": "memmap", "dataset_kwargs": {"data_dir": str(calls_bundle)}}
    for tag, expected_key in (("tichu", "call_tichu"), ("grand", "call_grand_tichu")):
        got = list(_build_dataset(config, _TAG_TO_CALL_TYPE[tag]))
        want = list(MemmapCallDataset(calls_bundle, call_type=expected_key))
        assert len(got) == len(want)
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g.features, w.features)
            assert g.target == w.target


def test_cli_trains_both_networks_from_memmap_bundle(calls_bundle, tmp_path):
    """End-to-end: `main` with `dataset: memmap` trains both call networks off
    the packed bundle and saves their checkpoints — no archive re-replay."""
    config = {
        "dataset": "memmap",
        "dataset_kwargs": {"data_dir": str(calls_bundle)},
        "model": {"hidden": 16, "skill_dim": 4},
        "learning_rate": 5.0e-3,
        "batch_size": 8,
        "epochs": 2,
        "seed": 0,
    }
    config_path = tmp_path / "calls_memmap.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"

    rc = main(["--config", str(config_path), "--run-dir", str(run_dir)])
    assert rc == 0
    for tag in ("grand", "tichu"):
        assert (run_dir / "checkpoints" / f"{tag}_final.bin").exists()
        assert (run_dir / f"{tag}_step.csv").exists()


def test_cli_memmap_streams_with_shuffle_and_val(calls_bundle, tmp_path, monkeypatch):
    """The memmap path must STREAM (never materialise the slice into a list)
    yet still produce per-epoch val + calling-rate outputs under `shuffle`.

    Guards the full-corpus fix: `train_calls` would OOM if it called
    `list(dataset)` on the ~88M-row slice. The streaming path consumes
    `iter_batches`, never `__iter__`, so we make `__iter__` blow up — if the
    CLI ever materialises the dataset, the test fails loudly.
    """
    def _boom(self):
        raise AssertionError(
            "memmap dataset iterated as objects — should stream via iter_batches"
        )

    monkeypatch.setattr(MemmapCallDataset, "__iter__", _boom)

    config = {
        "dataset": "memmap",
        "dataset_kwargs": {"data_dir": str(calls_bundle)},
        "model": {"hidden": 16, "skill_dim": 4},
        "learning_rate": 5.0e-3,
        "batch_size": 8,
        "epochs": 2,
        "seed": 0,
        "shuffle": True,
        "val_frac": 0.2,
    }
    config_path = tmp_path / "calls_stream.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"

    rc = main(["--config", str(config_path), "--run-dir", str(run_dir), "--only", "tichu"])
    assert rc == 0
    assert (run_dir / "checkpoints" / "tichu_final.bin").exists()
    assert (run_dir / "tichu_step.csv").exists()
    # Streaming val + calling-rate outputs were produced.
    assert (run_dir / "tichu_val.csv").exists()
    assert (run_dir / "tichu_calling_rate_by_decile_epoch0.csv").exists()
    assert (run_dir / "tichu_calling_rate_by_decile_epoch1.csv").exists()
