"""Training loop loss-decrease + checkpoint round-trip."""

import csv

import pytest
import torch

from tichu_training.bc.dataset import SyntheticBCDataset
from tichu_training.bc.heads import BCModel
from tichu_training.bc.training import (
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def _make_small_model(feature_dim: int = 16) -> BCModel:
    return BCModel(
        feature_dim=feature_dim,
        skill_buckets=10,
        skill_dim=4,
        trunk_hidden=16,
        trunk_depth=1,
        trunk_out_dim=8,
        head_hidden=16,
    )


def test_training_reduces_loss_on_synthetic_data(tmp_path):
    torch.manual_seed(0)
    feature_dim = 16
    model = _make_small_model(feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    dataset = SyntheticBCDataset(seed=0, n_per_head=16, feature_dim=feature_dim)
    log_path = tmp_path / "step.csv"

    initial = train_one_epoch(
        model, list(dataset), optimizer, batch_size=8, log_path=log_path,
    )
    # Multiple epochs to overfit the tiny dataset.
    for _ in range(8):
        final = train_one_epoch(
            model, list(dataset), optimizer, batch_size=8, log_path=log_path,
        )
    assert final < initial


def test_csv_log_has_one_row_per_step(tmp_path):
    feature_dim = 16
    model = _make_small_model(feature_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = list(SyntheticBCDataset(seed=0, n_per_head=4, feature_dim=feature_dim))
    log_path = tmp_path / "step.csv"

    train_one_epoch(model, dataset, optimizer, batch_size=2, log_path=log_path)

    with log_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 1
    for r in rows:
        assert "step" in r
        assert "loss_total" in r
        assert "loss_play" in r


def test_step_csv_is_written_per_batch_not_at_epoch_end(tmp_path):
    """The CSV must be flushed as each batch fires so long-running epochs
    (and crashes mid-epoch) leave on-disk progress, not an empty file."""
    feature_dim = 16
    model = _make_small_model(feature_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    log_path = tmp_path / "step.csv"

    dataset = list(SyntheticBCDataset(seed=0, n_per_head=4, feature_dim=feature_dim))
    play_only = [e for e in dataset if e.decision_type == "play"][:2]
    assert len(play_only) == 2

    line_counts_after_each_yield: list[int] = []

    def spy():
        for ex in play_only:
            yield ex
            line_counts_after_each_yield.append(
                len(log_path.read_text(encoding="utf-8").splitlines())
                if log_path.exists() else 0
            )

    train_one_epoch(
        model, spy(), optimizer,
        batch_size=2, log_path=log_path, show_progress=False,
    )

    # After yielding the 2nd 'play' example the buffer fills, _fire runs, and
    # the row must be on disk before the generator is asked for the next item.
    # If writes are deferred to epoch-end, the count stays at 0 throughout.
    assert line_counts_after_each_yield[-1] >= 2, (
        f"expected header + ≥1 data row visible mid-iteration, "
        f"got line counts {line_counts_after_each_yield}"
    )


def test_step_csv_survives_dataset_exception_mid_iteration(tmp_path):
    """When the data iterator blows up partway through an epoch, the rows
    that already fired must be on disk and the exception must propagate."""
    feature_dim = 16
    model = _make_small_model(feature_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    log_path = tmp_path / "step.csv"

    dataset = list(SyntheticBCDataset(seed=0, n_per_head=4, feature_dim=feature_dim))
    play_only = [e for e in dataset if e.decision_type == "play"][:2]

    def explodes_after_one_batch():
        for ex in play_only:
            yield ex
        raise RuntimeError("simulated mid-epoch crash")

    with pytest.raises(RuntimeError, match="simulated mid-epoch crash"):
        train_one_epoch(
            model, explodes_after_one_batch(), optimizer,
            batch_size=2, log_path=log_path, show_progress=False,
        )

    assert log_path.exists()
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert len(rows) >= 1, "expected the row from the batch that fired before the crash"


def test_checkpoint_round_trip(tmp_path):
    feature_dim = 16
    model = _make_small_model(feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Take one step so optimizer state has momentum buffers.
    dataset = list(SyntheticBCDataset(seed=0, n_per_head=4, feature_dim=feature_dim))
    train_one_epoch(model, dataset, optimizer, batch_size=4, log_path=tmp_path / "step.csv")

    path = tmp_path / "ck.bin"
    save_checkpoint(model, optimizer, step=42, path=path)

    # Re-instantiate the model + optimizer fresh and load.
    model2 = _make_small_model(feature_dim)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    step = load_checkpoint(path, model2, optimizer2)
    assert step == 42

    # Weights match after load.
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
