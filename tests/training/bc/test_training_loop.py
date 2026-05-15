"""Training loop loss-decrease + checkpoint round-trip."""

import csv

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
