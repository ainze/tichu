"""SyntheticBeliefDataset + train_one_belief_epoch + calibration_table."""

import csv

import numpy as np
import torch

from tichu_training.belief.dataset import BeliefExample, SyntheticBeliefDataset
from tichu_training.belief.model import BeliefModel
from tichu_training.belief.training import (
    calibration_table,
    train_one_belief_epoch,
    write_calibration_csv,
)


def test_synthetic_dataset_is_deterministic():
    a = list(SyntheticBeliefDataset(seed=0, n_examples=10, feature_dim=8))
    b = list(SyntheticBeliefDataset(seed=0, n_examples=10, feature_dim=8))
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        assert np.array_equal(ea.features, eb.features)
        assert np.array_equal(ea.labels, eb.labels)
        assert np.array_equal(ea.mask, eb.mask)


def test_synthetic_labels_shape_and_validity():
    examples = list(SyntheticBeliefDataset(seed=0, n_examples=5, feature_dim=4))
    for e in examples:
        assert e.labels.shape == (3, 56)
        assert e.mask.shape == (3, 56)
        # Each card is held by at most one opponent (across the 3 rows).
        per_card = e.labels.sum(axis=0)
        assert (per_card <= 1).all()
        # Masked-out positions must have label 0 (consistent: card is elsewhere).
        assert ((e.labels == 1) & ~e.mask).sum() == 0


def test_synthetic_mask_excludes_own_hand_and_played_cards():
    examples = list(SyntheticBeliefDataset(seed=0, n_examples=5, feature_dim=4))
    for e in examples:
        # Mask is True iff position is "in play among opponents OR could be" —
        # i.e. the card is not known to be in own hand or already played.
        # Since labels of 1 mean "this opponent holds it", those must be masked-in.
        assert ((e.labels == 1) & ~e.mask).sum() == 0
        # At least some positions are masked out (own hand + played cards exist).
        assert (~e.mask).any()


def test_training_loss_decreases_over_synthetic_epochs(tmp_path):
    examples = list(SyntheticBeliefDataset(seed=0, n_examples=64, feature_dim=16))
    torch.manual_seed(0)
    m = BeliefModel(feature_dim=16, hidden=32)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    log_path = tmp_path / "step.csv"
    losses = []
    for _ in range(5):
        last = train_one_belief_epoch(m, examples, opt, batch_size=8, log_path=log_path)
        losses.append(last)
    assert losses[-1] < losses[0]


def test_training_csv_columns(tmp_path):
    examples = list(SyntheticBeliefDataset(seed=0, n_examples=16, feature_dim=8))
    torch.manual_seed(0)
    m = BeliefModel(feature_dim=8, hidden=16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    log_path = tmp_path / "step.csv"
    train_one_belief_epoch(m, examples, opt, batch_size=4, log_path=log_path)
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert rows
    assert {"step", "loss", "accuracy"} <= set(rows[0].keys())


def test_calibration_table_has_ten_rows_and_columns():
    examples = list(SyntheticBeliefDataset(seed=0, n_examples=32, feature_dim=8))
    torch.manual_seed(0)
    m = BeliefModel(feature_dim=8, hidden=16)
    table = calibration_table(m, examples)
    assert len(table) == 10
    expected = {"bucket_index", "p_low", "p_high", "predicted_mean", "empirical_freq", "n"}
    assert expected <= set(table[0].keys())


def test_calibration_csv_roundtrip(tmp_path):
    examples = list(SyntheticBeliefDataset(seed=0, n_examples=16, feature_dim=8))
    torch.manual_seed(0)
    m = BeliefModel(feature_dim=8, hidden=16)
    out_path = tmp_path / "calibration.csv"
    write_calibration_csv(m, examples, out_path)
    rows = list(csv.DictReader(out_path.open("r", encoding="utf-8")))
    assert len(rows) == 10
