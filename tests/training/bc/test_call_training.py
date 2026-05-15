"""Synthetic call dataset + training loop + skill-rate logging."""

import csv

import torch

from tichu_training.bc.call_model import GrandTichuCallNetwork
from tichu_training.bc.call_training import (
    SyntheticCallDataset,
    calling_rate_by_decile,
    train_one_call_epoch,
)


def test_synthetic_call_dataset_size_and_balance():
    examples = list(SyntheticCallDataset(seed=0, n_examples=200, positive_rate=0.3, feature_dim=8))
    assert len(examples) == 200
    positives = sum(1 for e in examples if e.target == 1)
    # Should be roughly 30% (within sampling tolerance).
    assert 40 <= positives <= 80


def test_synthetic_call_dataset_deterministic():
    a = list(SyntheticCallDataset(seed=1, n_examples=50, positive_rate=0.5, feature_dim=8))
    b = list(SyntheticCallDataset(seed=1, n_examples=50, positive_rate=0.5, feature_dim=8))
    for ea, eb in zip(a, b):
        assert ea.target == eb.target
        assert (ea.features == eb.features).all()


def test_training_reduces_loss_on_synthetic(tmp_path):
    torch.manual_seed(0)
    net = GrandTichuCallNetwork(feature_dim=16, skill_dim=4, hidden=16)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
    examples = list(SyntheticCallDataset(seed=0, n_examples=64, positive_rate=0.5, feature_dim=16))

    log_path = tmp_path / "step.csv"
    initial = train_one_call_epoch(net, examples, optimizer, batch_size=16, log_path=log_path)
    for _ in range(6):
        final = train_one_call_epoch(net, examples, optimizer, batch_size=16, log_path=log_path)
    assert final < initial


def test_csv_log_format(tmp_path):
    net = GrandTichuCallNetwork(feature_dim=8, skill_dim=2, hidden=8)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    examples = list(SyntheticCallDataset(seed=0, n_examples=10, positive_rate=0.5, feature_dim=8))
    log_path = tmp_path / "step.csv"
    train_one_call_epoch(net, examples, optimizer, batch_size=5, log_path=log_path)
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert rows and {"step", "loss", "accuracy"} <= set(rows[0].keys())


def test_calling_rate_by_decile_groups_correctly():
    net = GrandTichuCallNetwork(feature_dim=8, skill_dim=2, hidden=8)
    # All examples with skill_decile=3 and 7; rate is just whether the net predicts call.
    examples = list(SyntheticCallDataset(seed=0, n_examples=20, positive_rate=0.5, feature_dim=8))
    rates = calling_rate_by_decile(net, examples)
    # Every observed decile must have an entry.
    seen_deciles = {e.skill_decile for e in examples}
    assert set(rates.keys()) == seen_deciles
    for decile, rate in rates.items():
        assert 0.0 <= rate <= 1.0
