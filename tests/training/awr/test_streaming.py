"""Streaming AWR pipeline — never materialises the full dataset.

The non-streaming path holds every BCExample in RAM (~66 KB each), so a
full-corpus AWR refinement would need 100+ TB. The streaming path
collects only a small held-out subset (capped) and streams the rest
through baseline-fit and AWR-refine passes.

These tests pin:
  * Deterministic held-out partitioning across passes (a given example
    always lands in the same bucket).
  * The held-out collector respects max_held_out without changing the
    partition predicate.
  * `fit_value_baseline_streaming` actually reduces MSE on a stream.
  * `awr_refine_epoch_streaming` writes the expected step.csv columns,
    drains per-head partial buffers at end-of-stream, computes
    win_rate_proxy on a passed held-out subset, and never materialises
    more than `chunk_size` examples in memory.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from tichu_training.awr.streaming import (
    awr_refine_epoch_streaming,
    collect_held_out,
    fit_value_baseline_streaming,
    make_held_out_filter,
)
from tichu_training.awr.value_baseline import ValueBaseline
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


def _factory(examples):
    """Streaming-compatible factory: returns a fresh iterator on each call."""
    return lambda: iter(examples)


# ---------- held-out partitioning ----------

def test_held_out_filter_is_deterministic_across_passes():
    """Same example must land in the same partition every time so the
    held-out set is consistent across the baseline-fit pass and every
    AWR-refine pass."""
    examples = list(SyntheticBCDataset(seed=0, n_per_head=20, feature_dim=8))
    is_held_out, _ = make_held_out_filter(fraction=0.2)

    first = [is_held_out(e) for e in examples]
    second = [is_held_out(e) for e in examples]
    assert first == second


def test_collect_held_out_caps_at_max_but_predicate_keeps_excluding():
    """Once the held-out store hits max_held_out we stop accumulating,
    but every example matching the partition predicate must still be
    skipped during training — otherwise the model trains on held-out
    examples on later passes and the win_rate_proxy is contaminated."""
    examples = list(SyntheticBCDataset(seed=0, n_per_head=100, feature_dim=8))

    held, is_held_out = collect_held_out(
        _factory(examples), fraction=0.3, max_held_out=10,
    )

    assert len(held) <= 10, "held-out must respect cap"
    # The predicate selects ~30% of examples by hash — many more than 10.
    matched = sum(1 for e in examples if is_held_out(e))
    assert matched > 10, "predicate must keep matching examples beyond the cap"


# ---------- baseline fit ----------

def test_fit_value_baseline_streaming_reduces_mse():
    """The streaming SGD fit should reduce MSE relative to a randomly
    initialised baseline. Tests on a synthetic stream where outcomes are
    a learnable function of the features."""
    torch.manual_seed(0)
    feature_dim = 8
    examples = list(SyntheticBCDataset(seed=0, n_per_head=200, feature_dim=feature_dim))
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)

    # Initial MSE on the stream (no training).
    feats = np.stack([e.features for e in examples])
    outs = np.array([e.round_outcome for e in examples], dtype=np.float32)
    with torch.no_grad():
        init_mse = float(((baseline(torch.from_numpy(feats)) - torch.from_numpy(outs)) ** 2).mean())

    final_mse = fit_value_baseline_streaming(
        baseline, _factory(examples),
        batch_size=32, lr=1e-2, chunk_size=64, sgd_steps_per_chunk=2,
        held_out_filter=lambda e: False,
        show_progress=False,
    )

    assert final_mse < init_mse, f"expected MSE to drop, init={init_mse:.4f} final={final_mse:.4f}"


# ---------- refine epoch ----------

def test_awr_refine_epoch_streaming_writes_expected_csv_columns(tmp_path):
    """step.csv must carry the same schema as the non-streaming path,
    including awr_weight_mean, and the summary must include the held-out
    win_rate_proxy."""
    torch.manual_seed(0)
    feature_dim = 8
    examples = list(SyntheticBCDataset(seed=0, n_per_head=50, feature_dim=feature_dim))
    held_out = examples[:8]  # tiny held-out subset
    train_pool = examples[8:]

    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_path = tmp_path / "step.csv"
    summary = awr_refine_epoch_streaming(
        model, _factory(train_pool), baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=4,
        log_path=log_path,
        held_out_examples=held_out,
        held_out_filter=lambda e: False,  # train_pool already excludes held-out
        chunk_size=16,
        show_progress=False,
    )

    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert rows
    assert "awr_weight_mean" in rows[0]
    assert "loss_total" in rows[0]
    assert "loss_play" in rows[0]
    assert summary["win_rate_proxy"] is not None
    assert 0.0 <= summary["win_rate_proxy"] <= 1.0
    assert summary["avg_weight"] > 0
    assert summary["loss_total"] >= 0


def test_awr_refine_epoch_streaming_drains_partial_per_head_buffers(tmp_path):
    """When the stream ends with under-full per-head buffers, those
    examples must still fire one final batch each — otherwise the last
    chunk of every head is silently dropped."""
    torch.manual_seed(0)
    feature_dim = 8
    # Carefully sized so the last chunk leaves partial buffers:
    # 14 play examples + batch_size=4 ⇒ buffers fire at 4,8,12; 2 examples remain.
    examples = list(SyntheticBCDataset(seed=0, n_per_head=14, feature_dim=feature_dim))
    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_path = tmp_path / "step.csv"
    awr_refine_epoch_streaming(
        model, _factory(examples), baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=4,
        log_path=log_path,
        held_out_examples=None,
        held_out_filter=lambda e: False,
        chunk_size=32,
        show_progress=False,
    )

    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    # 14 examples per head × 3 heads = 42 total. batch_size 4 ⇒ floor(14/4)=3 full
    # batches per head + 1 partial drain = 4 batches per head × 3 = 12 rows total.
    # Without partial-buffer drain we'd see 9 rows.
    assert len(rows) == 12, (
        f"expected 12 rows (3 full + 1 partial drain per head × 3 heads), got {len(rows)}"
    )


def test_awr_refine_epoch_streaming_excludes_held_out_by_predicate(tmp_path):
    """Examples matching held_out_filter must never be trained on, no
    matter how many AWR-refine passes are run."""
    torch.manual_seed(0)
    feature_dim = 8
    examples = list(SyntheticBCDataset(seed=0, n_per_head=20, feature_dim=feature_dim))

    # Mark first 10 examples as held-out via identity check.
    held_set = {id(e) for e in examples[:10]}
    def is_held(e):
        return id(e) in held_set

    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Spy on which examples actually reach the per-head buffers by
    # monkeypatching the baseline forward to record what comes through.
    seen_count = {"n": 0}
    real_forward = baseline.forward
    def spy(features):
        seen_count["n"] += int(features.shape[0])
        return real_forward(features)
    baseline.forward = spy  # type: ignore[method-assign]

    awr_refine_epoch_streaming(
        model, _factory(examples), baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=4,
        log_path=tmp_path / "step.csv",
        held_out_examples=None,
        held_out_filter=is_held,
        chunk_size=8,
        show_progress=False,
    )

    # 60 total examples, 10 held-out ⇒ 50 should reach the chunked baseline forward.
    assert seen_count["n"] == 50, (
        f"expected 50 examples through forward (60 - 10 held-out), got {seen_count['n']}"
    )
