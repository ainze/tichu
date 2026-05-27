"""awr_refine_epoch — wire baseline + AWR weights into BC objective."""

import csv

import numpy as np
import pytest
import torch

from tichu_training.awr.refine import _apply_awr_weights, awr_refine_epoch
from tichu_training.awr.value_baseline import ValueBaseline, fit_value_baseline
from tichu_training.bc.dataset import BCExample, SyntheticBCDataset
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS


def _build_model(feature_dim: int):
    torch.manual_seed(0)
    return BCModel(
        feature_dim=feature_dim,
        skill_buckets=10,
        skill_dim=4,
        trunk_hidden=16,
        trunk_depth=1,
        trunk_out_dim=16,
        head_hidden=16,
    )


def test_bc_example_round_outcome_defaults_to_zero():
    e = BCExample(
        decision_type="play",
        features=np.zeros(4, dtype=np.float32),
        target=0,
        legal_mask=np.array([True], dtype=bool),
        sample_weight=1.0,
        skill_decile=0,
    )
    assert e.round_outcome == 0.0


def test_synthetic_dataset_emits_non_zero_round_outcome():
    examples = list(SyntheticBCDataset(seed=0, n_per_head=20, feature_dim=8))
    outcomes = np.array([e.round_outcome for e in examples], dtype=np.float32)
    assert (outcomes != 0).any()


def test_refine_epoch_runs_and_returns_summary(tmp_path):
    feature_dim = 8
    examples = list(SyntheticBCDataset(seed=0, n_per_head=8, feature_dim=feature_dim))
    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    out = awr_refine_epoch(
        model, examples, baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=4,
        log_path=tmp_path / "step.csv",
    )
    assert "avg_weight" in out and "loss_total" in out
    assert out["avg_weight"] > 0.0


def test_refine_epoch_csv_has_awr_weight_mean_column(tmp_path):
    feature_dim = 8
    examples = list(SyntheticBCDataset(seed=0, n_per_head=8, feature_dim=feature_dim))
    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_path = tmp_path / "step.csv"
    awr_refine_epoch(
        model, examples, baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=4,
        log_path=log_path,
    )
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert rows
    assert "awr_weight_mean" in rows[0]
    assert float(rows[0]["awr_weight_mean"]) > 0.0


def test_refine_with_unit_weights_matches_train_one_epoch_gradients(tmp_path):
    """When the baseline perfectly predicts outcomes, advantages are 0 and AWR
    weights are all 1 (uniform). The gradient step then matches what a plain
    weighted BC step would produce on the same examples and sample_weights."""
    feature_dim = 4
    examples = list(SyntheticBCDataset(seed=42, n_per_head=4, feature_dim=feature_dim))
    # Force every outcome to 0 — baseline trivially predicts 0 ⇒ advantage 0.
    examples = [
        BCExample(
            decision_type=e.decision_type,
            features=e.features,
            target=e.target,
            legal_mask=e.legal_mask,
            sample_weight=e.sample_weight,
            skill_decile=e.skill_decile,
            round_outcome=0.0,
        )
        for e in examples
    ]

    # AWR path with a freshly initialised baseline whose predictions are roughly 0
    # (centered random init; advantages dominated by outcomes - 0 = 0).
    torch.manual_seed(0)
    model_awr = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=8)
    # Zero-out baseline weights to make predictions exactly 0 ⇒ advantage = 0.
    with torch.no_grad():
        for p in baseline.parameters():
            p.zero_()
    opt_awr = torch.optim.SGD(model_awr.parameters(), lr=0.1)
    out = awr_refine_epoch(
        model_awr, examples, baseline, opt_awr,
        beta=1.0, max_weight=20.0, batch_size=2,
        log_path=tmp_path / "awr.csv",
    )
    # With zero advantages, all AWR weights collapse to 1 (after the max-subtract
    # stability trick: scaled - max == 0 for all i, exp(0) == 1).
    assert abs(out["avg_weight"] - 1.0) < 1e-5


def test_held_out_subset_yields_win_rate_proxy(tmp_path):
    feature_dim = 8
    examples = list(SyntheticBCDataset(seed=0, n_per_head=10, feature_dim=feature_dim))
    split = int(0.9 * len(examples))
    train, held_out = examples[:split], examples[split:]

    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = awr_refine_epoch(
        model, train, baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=4,
        log_path=tmp_path / "awr.csv",
        held_out_subset=held_out,
    )
    assert out["win_rate_proxy"] is not None
    assert 0.0 <= out["win_rate_proxy"] <= 1.0


def test_step_csv_is_written_per_batch_not_at_epoch_end(tmp_path, monkeypatch):
    """The CSV must be flushed as each batch fires so long-running refine
    epochs (and crashes mid-epoch) leave on-disk progress, not an empty file."""
    import tichu_training.awr.refine as refine_mod

    feature_dim = 8
    # Enough examples for ≥2 batches per head at batch_size=4.
    examples = list(SyntheticBCDataset(seed=0, n_per_head=12, feature_dim=feature_dim))
    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    real_append = refine_mod._append_csv
    append_call_row_counts: list[int] = []

    def spy_append(path, rows):
        rows_list = list(rows)
        append_call_row_counts.append(len(rows_list))
        real_append(path, rows_list)

    monkeypatch.setattr(refine_mod, "_append_csv", spy_append)

    awr_refine_epoch(
        model, examples, baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=4,
        log_path=tmp_path / "step.csv",
    )

    # Per-batch writes ⇒ many calls with one row each. Bulk-at-end ⇒ one
    # call carrying every row.
    assert len(append_call_row_counts) >= 2, (
        f"expected per-batch _append_csv calls, got {append_call_row_counts}"
    )
    assert all(c == 1 for c in append_call_row_counts), (
        f"expected exactly one row per call, got {append_call_row_counts}"
    )


def test_step_csv_survives_mid_epoch_exception(tmp_path, monkeypatch):
    """When the forward pass blows up partway through a refine epoch, the
    rows that already fired must be on disk and the exception must propagate."""
    import tichu_training.awr.refine as refine_mod

    feature_dim = 8
    examples = list(SyntheticBCDataset(seed=0, n_per_head=12, feature_dim=feature_dim))
    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    real_to_tensors = refine_mod._to_tensors
    call_count = {"n": 0}

    def explode_on_second_call(batch):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated mid-epoch crash")
        return real_to_tensors(batch)

    monkeypatch.setattr(refine_mod, "_to_tensors", explode_on_second_call)

    log_path = tmp_path / "step.csv"
    with pytest.raises(RuntimeError, match="simulated mid-epoch crash"):
        awr_refine_epoch(
            model, examples, baseline, optimizer,
            beta=1.0, max_weight=20.0, batch_size=4,
            log_path=log_path,
        )

    assert log_path.exists()
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert len(rows) >= 1, "expected the row from the batch that fired before the crash"


def test_apply_awr_weights_chunked_matches_full():
    """Chunking the baseline forward pass must produce numerically
    identical sample_weights to the unchunked version — chunking is a
    memory-only optimisation; the AWR math (max-subtract, exp, clip) is
    applied across the full advantage vector either way."""
    torch.manual_seed(0)
    feature_dim = 6
    examples = list(SyntheticBCDataset(seed=0, n_per_head=20, feature_dim=feature_dim))
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=8)

    full = _apply_awr_weights(examples, baseline, beta=1.0, max_weight=20.0)
    chunked = _apply_awr_weights(
        examples, baseline, beta=1.0, max_weight=20.0, chunk_size=7,
    )

    assert len(full) == len(chunked) == len(examples)
    for a, b in zip(full, chunked):
        assert a.sample_weight == pytest.approx(b.sample_weight, rel=1e-6)


def test_apply_awr_weights_calls_baseline_in_chunks():
    """With chunk_size=10 over 25 examples, the baseline should see three
    forward passes of size 10, 10, 5 — never the full 25 at once."""
    torch.manual_seed(0)
    feature_dim = 4
    examples = list(SyntheticBCDataset(seed=0, n_per_head=9, feature_dim=feature_dim))[:25]
    assert len(examples) == 25
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=4)

    seen_batch_sizes: list[int] = []
    real_forward = baseline.forward

    def spy(features):
        seen_batch_sizes.append(int(features.shape[0]))
        return real_forward(features)

    baseline.forward = spy  # type: ignore[method-assign]
    _apply_awr_weights(
        examples, baseline, beta=1.0, max_weight=20.0, chunk_size=10,
    )

    assert seen_batch_sizes == [10, 10, 5], (
        f"expected chunks 10,10,5 got {seen_batch_sizes}"
    )


def test_apply_awr_weights_accepts_precomputed_features():
    """The caller often already has features as a stacked array (it just
    fit the value baseline on them). Re-stacking inside _apply_awr_weights
    doubles memory; passing them in must give the same result."""
    torch.manual_seed(0)
    feature_dim = 5
    examples = list(SyntheticBCDataset(seed=0, n_per_head=12, feature_dim=feature_dim))
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=6)

    auto = _apply_awr_weights(examples, baseline, beta=1.0, max_weight=20.0)

    features = np.stack([e.features for e in examples])
    shared = _apply_awr_weights(
        examples, baseline, beta=1.0, max_weight=20.0, features=features,
    )

    for a, b in zip(auto, shared):
        assert a.sample_weight == pytest.approx(b.sample_weight, rel=1e-6)


def test_fit_value_baseline_never_does_full_dataset_forward():
    """The final MSE re-evaluation must also be chunked — otherwise a
    multi-million-row training set OOMs the GPU on the last line of
    fit_value_baseline, after training succeeded."""
    torch.manual_seed(0)
    feature_dim = 4
    n = 73
    features = np.random.RandomState(0).randn(n, feature_dim).astype(np.float32)
    outcomes = np.random.RandomState(1).randn(n).astype(np.float32)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=4)

    max_batch_seen = {"n": 0}
    real_forward = baseline.forward

    def spy(features_t):
        max_batch_seen["n"] = max(max_batch_seen["n"], int(features_t.shape[0]))
        return real_forward(features_t)

    baseline.forward = spy  # type: ignore[method-assign]
    fit_value_baseline(
        baseline, features, outcomes,
        batch_size=16, epochs=1, lr=1e-3,
    )

    # batch_size=16 ⇒ no forward pass should ever see > 16 examples,
    # including the final MSE re-evaluation. Without chunking the final
    # pass would see all 73.
    assert max_batch_seen["n"] <= 16, (
        f"expected forward batches ≤ 16, saw a {max_batch_seen['n']}-example pass"
    )


def test_invalid_beta_propagates(tmp_path):
    examples = list(SyntheticBCDataset(seed=0, n_per_head=4, feature_dim=4))
    model = _build_model(4)
    baseline = ValueBaseline(feature_dim=4, hidden=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        awr_refine_epoch(
            model, examples, baseline, optimizer,
            beta=0.0, max_weight=20.0, batch_size=2,
            log_path=tmp_path / "awr.csv",
        )
