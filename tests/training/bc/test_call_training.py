"""Synthetic call dataset + training loop + skill-rate logging."""

import csv

import numpy as np
import torch

from tichu_training.bc.call_model import GrandTichuCallNetwork
from tichu_training.bc.call_training import (
    CallExample,
    SyntheticCallDataset,
    _binary_auc,
    calling_rate_by_decile,
    evaluate_call_examples,
    split_examples_by_game,
    train_one_call_epoch,
)
from tichu_training.featurizer import (
    FEATURIZER_OUTPUT_DIM,
    SECTION_DIMS,
    SECTION_OFFSETS,
    mask_self_tichu_call,
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


def test_training_accepts_generator_iterable(tmp_path):
    """`train_one_call_epoch` must consume an `Iterable[CallExample]`,
    not require a `Sequence`. Generators are the canonical non-Sequence
    iterable; the parquet adapter is one. This guards Q7 (streaming)."""
    torch.manual_seed(0)
    net = GrandTichuCallNetwork(feature_dim=8, skill_dim=2, hidden=8)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    def gen():
        yield from SyntheticCallDataset(
            seed=0, n_examples=12, positive_rate=0.5, feature_dim=8,
        )

    log_path = tmp_path / "step.csv"
    # Should not raise — one epoch over the generator.
    train_one_call_epoch(net, gen(), optimizer, batch_size=4, log_path=log_path)


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


# ---------------------------------------------------------------------------
# Leak fix: `tichu_callers[seat]` must not survive into the call-classifier
# features, because at the seat's first-play snapshot it IS the label.
# ---------------------------------------------------------------------------


def test_mask_self_tichu_call_zeros_only_own_seat():
    """The mask must zero the calling seat's own bit and leave the other
    three seats' bits in the `tichu_callers` section untouched (those
    encode opponents' calls, which are legitimate public context)."""
    feat = np.ones(FEATURIZER_OUTPUT_DIM, dtype=np.float32)
    base = SECTION_OFFSETS["tichu_callers"]
    mask_self_tichu_call(feat, seat=1)
    assert feat[base + 0] == 1.0
    assert feat[base + 1] == 0.0  # the seat's own bit is gone
    assert feat[base + 2] == 1.0
    assert feat[base + 3] == 1.0
    # Adjacent sections must be untouched.
    assert feat[base - 1] == 1.0
    assert feat[base + SECTION_DIMS["tichu_callers"]] == 1.0


def test_mask_self_tichu_call_rejects_bad_seat():
    feat = np.zeros(FEATURIZER_OUTPUT_DIM, dtype=np.float32)
    for bad in (-1, 4, 7):
        try:
            mask_self_tichu_call(feat, seat=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for seat={bad}")


def test_section_offsets_match_section_dims():
    """`SECTION_OFFSETS` is a derived view of `SECTION_DIMS`; this guards
    against drift between the two if a section is ever reordered."""
    cursor = 0
    for name, dim in SECTION_DIMS.items():
        assert SECTION_OFFSETS[name] == cursor, name
        cursor += dim
    assert cursor == FEATURIZER_OUTPUT_DIM


# ---------------------------------------------------------------------------
# split_examples_by_game + evaluate_call_examples helpers.
# ---------------------------------------------------------------------------


def _ex(game_id: str, target: int = 0) -> CallExample:
    return CallExample(
        features=np.zeros(4, dtype=np.float32),
        target=target,
        skill_decile=0,
        sample_weight=1.0,
        game_id=game_id,
    )


def test_split_keeps_whole_games_on_one_side():
    """All examples sharing a game_id must land on the same side. With
    4 examples per game, an example-level split would put ~all games in
    both train and val — the test would not catch the bug."""
    examples = []
    for g in range(200):
        for _seat in range(4):
            examples.append(_ex(f"game_{g}"))
    train, val = split_examples_by_game(examples, val_frac=0.2, seed=0)
    assert len(train) + len(val) == len(examples)
    # No game appears on both sides.
    train_games = {e.game_id for e in train}
    val_games = {e.game_id for e in val}
    assert train_games.isdisjoint(val_games)
    # All 4 examples of any game land on the same side.
    from collections import Counter
    counts_train = Counter(e.game_id for e in train)
    counts_val = Counter(e.game_id for e in val)
    assert all(c == 4 for c in counts_train.values())
    assert all(c == 4 for c in counts_val.values())


def test_split_is_deterministic_under_seed():
    examples = [_ex(f"g{g}") for g in range(500)]
    t1, v1 = split_examples_by_game(examples, val_frac=0.1, seed=42)
    t2, v2 = split_examples_by_game(examples, val_frac=0.1, seed=42)
    assert [e.game_id for e in t1] == [e.game_id for e in t2]
    assert [e.game_id for e in v1] == [e.game_id for e in v2]


def test_split_val_frac_zero_returns_all_train():
    examples = [_ex(f"g{g}") for g in range(50)]
    train, val = split_examples_by_game(examples, val_frac=0.0)
    assert len(train) == 50 and val == []


def test_binary_auc_perfect_and_random():
    # Perfect separation → AUC 1.0
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert _binary_auc(probs, labels) == 1.0
    # Reversed → AUC 0.0
    assert _binary_auc(probs, labels[::-1]) == 0.0
    # Ties on probabilities collapse to average rank → AUC 0.5
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert abs(_binary_auc(probs, labels) - 0.5) < 1e-9
    # Single-class input is undefined → NaN
    assert np.isnan(_binary_auc(probs, np.zeros(4, dtype=np.int64)))


def test_evaluate_call_examples_smoke():
    """End-to-end smoke: evaluate returns the documented keys with finite
    values on a tiny synthetic set with both classes present."""
    net = GrandTichuCallNetwork(feature_dim=8, skill_dim=2, hidden=8)
    examples = list(SyntheticCallDataset(
        seed=0, n_examples=20, positive_rate=0.5, feature_dim=8,
    ))
    m = evaluate_call_examples(net, examples, batch_size=8)
    assert set(m.keys()) == {"n", "loss", "accuracy", "auc", "pos_frac"}
    assert m["n"] == 20.0
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["pos_frac"] <= 1.0
    assert np.isfinite(m["loss"])
    assert np.isfinite(m["auc"])  # both classes present

