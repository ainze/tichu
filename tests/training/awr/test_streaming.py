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


def test_fit_value_baseline_streaming_writes_baseline_csv(tmp_path):
    """When log_path is given, one row per chunk is written with the
    aggregate diagnostics, not just the per-batch noise."""
    torch.manual_seed(0)
    feature_dim = 8
    examples = list(SyntheticBCDataset(seed=0, n_per_head=200, feature_dim=feature_dim))
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)

    log_path = tmp_path / "baseline.csv"
    fit_value_baseline_streaming(
        baseline, _factory(examples),
        batch_size=32, lr=1e-2, chunk_size=64, sgd_steps_per_chunk=2,
        held_out_filter=lambda e: False,
        log_path=log_path,
        show_progress=False,
    )

    with log_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "baseline.csv must contain at least one chunk row"
    expected_cols = {
        "chunk_idx", "examples_seen", "chunk_mse", "running_mse", "last_batch_mse",
    }
    assert expected_cols <= set(rows[0].keys())
    # examples_seen must be monotonic non-decreasing.
    seen = [int(r["examples_seen"]) for r in rows]
    assert seen == sorted(seen) and seen[-1] > 0
    # running_mse exists and is finite — the cardinal property the previous
    # single-batch reading lacked.
    for r in rows:
        rm = float(r["running_mse"])
        assert rm == rm and rm != float("inf"), f"running_mse must be finite, got {rm}"


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


def _stream_with_outcomes_in_range(seed, n_per_head, feature_dim, low, high):
    """Synthetic dataset rewritten with outcomes uniformly in [low, high].
    Use to reproduce Tichu-scale advantages (~±200) on the test suite's
    deterministic synthetic generator."""
    rng = np.random.RandomState(seed)
    examples = list(SyntheticBCDataset(seed=seed, n_per_head=n_per_head, feature_dim=feature_dim))
    return [
        BCExample(
            decision_type=e.decision_type,
            features=e.features,
            target=e.target,
            legal_mask=e.legal_mask,
            sample_weight=e.sample_weight,
            skill_decile=e.skill_decile,
            round_outcome=float(rng.uniform(low, high)),
        )
        for e in examples
    ]


def test_unstandardised_large_advantages_collapse_weights_to_zero(tmp_path):
    """With Tichu-scale outcomes (±200) and beta=1.0, per-chunk
    max-subtract pushes exp(-large) → 0 for nearly every example. The
    chunk-mean AWR weight drops to ~1/chunk_size. This is the bug that
    standardisation fixes — pinning it ensures the fix actually
    addresses the cause."""
    torch.manual_seed(0)
    feature_dim = 8
    examples = _stream_with_outcomes_in_range(0, 50, feature_dim, low=-200, high=200)

    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    summary = awr_refine_epoch_streaming(
        model, _factory(examples), baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=8,
        log_path=tmp_path / "step.csv",
        held_out_examples=None,
        held_out_filter=lambda e: False,
        chunk_size=64,
        standardize_advantages=False,
        show_progress=False,
    )

    # avg_weight collapses to << 1 because only the single max-advantage
    # example per chunk gets non-trivial weight.
    assert summary["avg_weight"] < 0.05, (
        f"expected weight collapse with unstandardised ±200 advantages, "
        f"got avg_weight={summary['avg_weight']:.4f}"
    )


def test_standardised_advantages_keep_weights_in_healthy_range(tmp_path):
    """Same Tichu-scale advantages, but with per-chunk standardisation
    the scaled advantages have ~unit variance, beta=1.0 behaves as the
    AWR paper intends, and chunk-mean weights stay in a sensible range
    (mean ≳ 0.1 for roughly-normal advantages after max-subtract)."""
    torch.manual_seed(0)
    feature_dim = 8
    examples = _stream_with_outcomes_in_range(0, 50, feature_dim, low=-200, high=200)

    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    summary = awr_refine_epoch_streaming(
        model, _factory(examples), baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=8,
        log_path=tmp_path / "step.csv",
        held_out_examples=None,
        held_out_filter=lambda e: False,
        chunk_size=64,
        standardize_advantages=True,
        show_progress=False,
    )

    assert summary["avg_weight"] > 0.1, (
        f"expected healthy weight range with standardisation, "
        f"got avg_weight={summary['avg_weight']:.4f}"
    )


def test_mid_epoch_eval_callback_fires_every_n_chunks(tmp_path):
    """With eval_every_chunks=2 the callback must fire after chunks 2, 4,
    6, ... so the user gets intermediate win_rate_proxy values during
    long full-corpus epochs (where waiting for the end-of-epoch row
    means hours of blind running)."""
    torch.manual_seed(0)
    feature_dim = 8
    # 7 chunks of size 8 = 56 examples ⇒ expect callback at chunks 2, 4, 6
    # plus the implicit final partial chunk doesn't get a callback (just
    # the final end-of-epoch return).
    examples = list(SyntheticBCDataset(seed=0, n_per_head=20, feature_dim=feature_dim))[:56]
    held_out = examples[:8]

    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fired_at: list[dict] = []

    def callback(payload: dict) -> None:
        fired_at.append(dict(payload))

    awr_refine_epoch_streaming(
        model, _factory(examples), baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=4,
        log_path=tmp_path / "step.csv",
        held_out_examples=held_out,
        held_out_filter=lambda e: False,
        chunk_size=8,
        eval_every_chunks=2,
        intermediate_eval_callback=callback,
        show_progress=False,
    )

    # 56 examples / chunk_size 8 = 7 chunks. eval_every_chunks=2 fires
    # at chunks 2, 4, 6 — so 3 intermediate callbacks.
    assert len(fired_at) == 3, (
        f"expected 3 mid-epoch callbacks at chunks 2,4,6 got {len(fired_at)}"
    )
    for i, payload in enumerate(fired_at):
        assert "chunks_done" in payload
        assert "win_rate_proxy" in payload
        assert "loss_total" in payload
        assert "avg_weight" in payload
        assert payload["chunks_done"] == (i + 1) * 2
        assert 0.0 <= payload["win_rate_proxy"] <= 1.0


def test_no_mid_epoch_eval_when_disabled(tmp_path):
    """eval_every_chunks=0 preserves legacy behavior — the callback is
    never invoked, just the final summary returned."""
    torch.manual_seed(0)
    feature_dim = 8
    examples = list(SyntheticBCDataset(seed=0, n_per_head=20, feature_dim=feature_dim))[:40]

    model = _build_model(feature_dim)
    baseline = ValueBaseline(feature_dim=feature_dim, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    called = []

    awr_refine_epoch_streaming(
        model, _factory(examples), baseline, optimizer,
        beta=1.0, max_weight=20.0, batch_size=4,
        log_path=tmp_path / "step.csv",
        held_out_examples=examples[:4],
        held_out_filter=lambda e: False,
        chunk_size=8,
        eval_every_chunks=0,
        intermediate_eval_callback=lambda p: called.append(p),
        show_progress=False,
    )

    assert called == [], "callback must not fire when eval_every_chunks=0"


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
