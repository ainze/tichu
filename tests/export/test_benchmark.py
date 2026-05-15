"""Latency benchmark utility + slow-marked p99 gate."""

import pytest
import torch

from tichu_export.benchmark import benchmark_p99


def _build_tiny_bc():
    from tichu_training.bc.heads import BCModel
    torch.manual_seed(0)
    return BCModel(
        feature_dim=16, skill_buckets=10, skill_dim=4,
        trunk_hidden=16, trunk_depth=1, trunk_out_dim=16, head_hidden=16,
    )


def test_benchmark_returns_four_percentile_floats():
    model = _build_tiny_bc()
    stats = benchmark_p99(
        model,
        torch.randn(1, 16),
        torch.tensor([0], dtype=torch.long),
        n=20,
    )
    for key in ("p50", "p95", "p99", "mean"):
        assert key in stats
        assert isinstance(stats[key], float)
        assert stats[key] > 0


def test_benchmark_p99_is_at_least_p50():
    model = _build_tiny_bc()
    stats = benchmark_p99(
        model,
        torch.randn(1, 16),
        torch.tensor([0], dtype=torch.long),
        n=50,
    )
    assert stats["p99"] >= stats["p50"]


def test_tiny_model_easily_within_budget():
    model = _build_tiny_bc()
    stats = benchmark_p99(
        model,
        torch.randn(1, 16),
        torch.tensor([0], dtype=torch.long),
        n=30,
    )
    assert stats["p99"] < 50.0


@pytest.mark.slow
def test_production_size_bc_under_500ms_p99():
    from tichu_training.bc.heads import BCModel
    torch.manual_seed(0)
    model = BCModel(
        feature_dim=512,  # smaller than 16568 for CI speed, but production-shape trunk
        skill_buckets=10, skill_dim=64,
        trunk_hidden=1024, trunk_depth=4, trunk_out_dim=512, head_hidden=256,
    ).eval()
    stats = benchmark_p99(
        model,
        torch.randn(1, 512),
        torch.tensor([0], dtype=torch.long),
        n=1000,
    )
    assert stats["p99"] < 500.0, stats
