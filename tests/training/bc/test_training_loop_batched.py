"""train_one_epoch_batched: parity with train_one_epoch on the same data,
loss-decrease smoke, and CSV/checkpoint contract preservation."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest
import torch

from tichu_training.bc.dataset import SyntheticBCDataset
from tichu_training.bc.heads import BCModel
from tichu_training.bc.materialised import MemmapBCDataset, materialise
from tichu_training.bc.training import (
    train_one_epoch,
    train_one_epoch_batched,
)
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM


def _make_small_model(feature_dim: int) -> BCModel:
    return BCModel(
        feature_dim=feature_dim,
        skill_buckets=10,
        skill_dim=4,
        trunk_hidden=16,
        trunk_depth=1,
        trunk_out_dim=8,
        head_hidden=16,
    )


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, int]:
    """A small materialised bundle backed by SyntheticBCDataset.

    Uses the real `FEATURIZER_OUTPUT_DIM` for the feature dim so the
    per-file size comfortably exceeds Windows' 64 KB memmap allocation
    granularity. With a smaller dim, the memmap open fails on Windows
    with WinError 8 even though the files write fine.

    Returns (data_dir, feature_dim).
    """
    feature_dim = FEATURIZER_OUTPUT_DIM
    examples = list(SyntheticBCDataset(
        seed=0, n_per_head=32, feature_dim=feature_dim, binary_features=True,
    ))
    out_dir = tmp_path / "bundle"
    materialise(iter(examples), out_dir, max_examples=len(examples))
    return out_dir, feature_dim


def test_batched_path_reduces_loss(bundle, tmp_path):
    """The fast path must train — basic loss-decrease smoke. Same shape
    as the test for train_one_epoch on SyntheticBCDataset."""
    data_dir, feature_dim = bundle
    torch.manual_seed(0)
    model = _make_small_model(feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    log_path = tmp_path / "step.csv"

    ds = MemmapBCDataset(data_dir)
    initial = train_one_epoch_batched(
        model, ds.iter_batches(8), optimizer,
        log_path=log_path, show_progress=False,
    )
    for _ in range(8):
        final = train_one_epoch_batched(
            model, ds.iter_batches(8), optimizer,
            log_path=log_path, show_progress=False,
        )
    assert final < initial


def test_batched_path_writes_csv_row_per_step(bundle, tmp_path):
    """One CSV row per fired batch — same contract as train_one_epoch."""
    data_dir, feature_dim = bundle
    model = _make_small_model(feature_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    log_path = tmp_path / "step.csv"

    ds = MemmapBCDataset(data_dir)
    n_batches = sum(1 for _ in ds.iter_batches(8))

    train_one_epoch_batched(
        model, ds.iter_batches(8), optimizer,
        log_path=log_path, show_progress=False,
    )

    with log_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == n_batches
    for r in rows:
        assert "step" in r
        assert "loss_total" in r
        assert "loss_play" in r


def test_batched_vs_per_example_loss_parity(bundle, tmp_path):
    """`train_one_epoch_batched(iter_batches)` and `train_one_epoch(iter)`
    must produce equivalent step counts on the same bundle with the same
    initial weights — both consume the same batches in the same order
    (per-head buffer fill semantics, preserved by `preserve_order=True`).

    Loss values are compared loosely: float-precision drift can build up
    over many ops, but the first-step loss must match closely because
    the very first batch fires from identical inputs and weights.
    """
    data_dir, feature_dim = bundle
    log_a = tmp_path / "step_a.csv"
    log_b = tmp_path / "step_b.csv"

    # Two identical models seeded the same way. `deepcopy` is the
    # bulletproof way to ensure both start with bit-identical weights;
    # constructing twice with the same seed isn't enough because module
    # construction order can interleave RNG draws.
    torch.manual_seed(0)
    model_a = _make_small_model(feature_dim)
    model_b = copy.deepcopy(model_a)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)

    ds = MemmapBCDataset(data_dir)

    # Path A: per-example __iter__ + train_one_epoch.
    train_one_epoch(
        model_a, list(ds), opt_a,
        batch_size=8, log_path=log_a, show_progress=False,
    )
    # Path B: iter_batches + train_one_epoch_batched.
    train_one_epoch_batched(
        model_b, ds.iter_batches(8), opt_b,
        log_path=log_b, show_progress=False,
    )

    with log_a.open("r", encoding="utf-8") as fh:
        rows_a = list(csv.DictReader(fh))
    with log_b.open("r", encoding="utf-8") as fh:
        rows_b = list(csv.DictReader(fh))

    # Both paths must fire the same number of steps.
    assert len(rows_a) == len(rows_b), (
        f"step-count mismatch: A={len(rows_a)} B={len(rows_b)}"
    )
    # First-step loss should match within float-precision noise — the
    # inputs are bit-identical, weights bit-identical, model identical.
    a0 = float(rows_a[0]["loss_total"])
    b0 = float(rows_b[0]["loss_total"])
    assert a0 == pytest.approx(b0, rel=1e-5, abs=1e-6), (
        f"first-step loss diverged: A={a0} B={b0}"
    )


def test_drop_last_skips_partial_batch(bundle, tmp_path):
    """`iter_batches(drop_last=True)` must produce strictly fewer steps
    when the per-head row counts aren't multiples of batch_size, and
    the batched trainer must fire one step per yielded batch."""
    data_dir, feature_dim = bundle
    model = _make_small_model(feature_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    ds = MemmapBCDataset(data_dir)
    n_keep = sum(1 for _ in ds.iter_batches(7, drop_last=False))
    n_drop = sum(1 for _ in ds.iter_batches(7, drop_last=True))
    assert n_drop < n_keep, "test bundle should have partial batches at bs=7"

    log_path = tmp_path / "step.csv"
    train_one_epoch_batched(
        model, ds.iter_batches(7, drop_last=True), optimizer,
        log_path=log_path, show_progress=False,
    )
    with log_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == n_drop


def test_batched_checkpoint_callback_fires_at_expected_cadence(bundle, tmp_path):
    """Mid-epoch checkpoint contract: callback fires every N batches,
    matching train_one_epoch's behaviour."""
    data_dir, feature_dim = bundle
    model = _make_small_model(feature_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    ds = MemmapBCDataset(data_dir)
    n_batches = sum(1 for _ in ds.iter_batches(8))

    fired_at: list[int] = []

    def ckpt_fn(step: int) -> None:
        fired_at.append(step)

    log_path = tmp_path / "step.csv"
    train_one_epoch_batched(
        model, ds.iter_batches(8), optimizer,
        log_path=log_path, show_progress=False,
        checkpoint_every_batches=2,
        checkpoint_fn=ckpt_fn,
    )
    # Should fire at step 2, 4, 6, ... up to the last multiple-of-2 step
    # not exceeding n_batches.
    expected = list(range(2, n_batches + 1, 2))
    assert fired_at == expected
