"""End-to-end smoke for `train_bc --refine-from <bc> --config <awr-streaming>`.

Mirrors test_cli.test_awr_smoke_runs_and_writes_compatible_checkpoint
but exercises the streaming path (awr.streaming: true).
"""

import csv
from pathlib import Path

import pytest
import yaml

from tichu_training.bc.heads import BCModel
from tichu_training.bc.training import load_checkpoint
from tichu_training.cli.train_bc import main


_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def _streaming_awr_config(base_path: Path, tmp_path: Path) -> Path:
    """Take awr_smoke.yaml and add awr.streaming: true so we exercise
    the streaming path on the small synthetic dataset (no parquet
    setup required)."""
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg.setdefault("awr", {})
    cfg["awr"]["streaming"] = True
    cfg["awr"]["chunk_size"] = 16
    cfg["awr"]["max_held_out"] = 8
    cfg["awr"]["held_out_fraction"] = 0.1
    cfg["awr"]["baseline_sgd_steps_per_chunk"] = 2
    out = tmp_path / "awr_streaming.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def test_awr_streaming_smoke_e2e(tmp_path):
    # 1. BC pass to produce a checkpoint.
    bc_run = tmp_path / "bc"; bc_run.mkdir()
    rc = main([
        "--config", str(_CONFIGS / "bc_smoke.yaml"),
        "--run-dir", str(bc_run),
    ])
    assert rc == 0
    bc_ckpt = sorted((bc_run / "checkpoints").glob("*.bin"))[-1]

    # 2. Streaming AWR refine.
    awr_cfg = _streaming_awr_config(_CONFIGS / "awr_smoke.yaml", tmp_path)
    awr_run = tmp_path / "awr"; awr_run.mkdir()
    rc = main([
        "--config", str(awr_cfg),
        "--run-dir", str(awr_run),
        "--refine-from", str(bc_ckpt),
    ])
    assert rc == 0

    # step.csv has the AWR schema.
    step_csv = awr_run / "step.csv"
    assert step_csv.exists()
    rows = list(csv.DictReader(step_csv.open("r", encoding="utf-8")))
    assert rows
    assert "awr_weight_mean" in rows[0]
    for r in rows:
        loss = float(r["loss_total"])
        assert loss == loss  # not NaN
        assert loss >= 0

    # epoch.csv has the per-epoch summary, including (possibly empty)
    # win_rate_proxy.
    epoch_csv = awr_run / "epoch.csv"
    assert epoch_csv.exists()
    epoch_rows = list(csv.DictReader(epoch_csv.open("r", encoding="utf-8")))
    assert epoch_rows
    assert {"epoch", "loss_total", "avg_weight", "win_rate_proxy"} <= set(epoch_rows[0].keys())

    # Per-epoch checkpoint + final checkpoint were written, all loadable.
    ckpts = sorted((awr_run / "checkpoints").glob("*.bin"))
    assert ckpts, "streaming AWR should write at least one checkpoint"
    model = BCModel(
        feature_dim=32, skill_buckets=10, skill_dim=4,
        trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16,
    )
    step = load_checkpoint(ckpts[-1], model)
    assert step > 0
