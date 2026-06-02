"""Smoke test for `train_calls` CLI."""

import csv
from pathlib import Path

import yaml

import tichu_training.cli.train_calls as train_calls_mod
from tichu_training.cli.train_calls import main


_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def _synthetic_cfg(tmp_path: Path, *, shuffle: bool, epochs: int = 3) -> Path:
    cfg = {
        "dataset": "synthetic",
        "dataset_kwargs": {
            "seed": 0, "n_examples": 128, "positive_rate": 0.4, "feature_dim": 32,
        },
        "model": {"hidden": 16, "skill_dim": 4},
        "learning_rate": 5.0e-3, "batch_size": 16, "epochs": epochs,
        "checkpoint_every": 1000, "seed": 0, "shuffle": shuffle,
    }
    p = tmp_path / f"cfg_shuffle_{shuffle}.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_shuffle_permutes_training_order_each_epoch(tmp_path, monkeypatch):
    """Regression for the recency-ordering bug: `shuffle: true` must permute
    the training pass each epoch (and not lose/duplicate rows), while the
    default leaves the order fixed every epoch."""
    captured: list[tuple[int, ...]] = []
    real = train_calls_mod.train_one_call_epoch

    def spy(net, examples, optimizer, **kw):
        captured.append(tuple(id(e) for e in examples))
        return real(net, examples, optimizer, **kw)

    monkeypatch.setattr(train_calls_mod, "train_one_call_epoch", spy)

    # Default (shuffle off): identical order every epoch.
    captured.clear()
    main([
        "--config", str(_synthetic_cfg(tmp_path, shuffle=False)),
        "--run-dir", str(tmp_path / "off"), "--only", "grand",
    ])
    off = captured[:]
    assert len(off) == 3
    assert off[0] == off[1] == off[2]

    # shuffle: true — every epoch a different order, same multiset of rows.
    captured.clear()
    main([
        "--config", str(_synthetic_cfg(tmp_path, shuffle=True)),
        "--run-dir", str(tmp_path / "on"), "--only", "grand",
    ])
    on = captured[:]
    assert len(on) == 3
    # Same rows present every epoch (in-place shuffle reuses one list, so ids
    # are stable within the run) — nothing dropped or duplicated.
    assert sorted(on[0]) == sorted(on[1]) == sorted(on[2])
    assert len(on[0]) == len(off[0])
    # Genuinely re-permuted each epoch.
    assert on[0] != on[1]
    assert on[1] != on[2]


def test_smoke_runs_both_networks_and_loss_decreases(tmp_path):
    rc = main([
        "--config", str(_CONFIGS / "calls_smoke.yaml"),
        "--run-dir", str(tmp_path),
    ])
    assert rc == 0

    for tag in ("grand", "tichu"):
        log_path = tmp_path / f"{tag}_step.csv"
        assert log_path.exists()
        rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
        assert len(rows) > 1
        assert float(rows[-1]["loss"]) < float(rows[0]["loss"])
        # Final checkpoint exists.
        assert (tmp_path / "checkpoints" / f"{tag}_final.bin").exists()
        # Calling-rate CSV written per epoch.
        rate_files = list(tmp_path.glob(f"{tag}_calling_rate_by_decile_epoch*.csv"))
        assert rate_files
        rate_rows = list(csv.DictReader(rate_files[0].open("r", encoding="utf-8")))
        assert rate_rows and {"decile", "n", "calling_rate"} <= set(rate_rows[0].keys())

    # Config copied into run dir.
    assert (tmp_path / "calls_smoke.yaml").exists()


def test_only_flag_restricts_to_one_network(tmp_path):
    rc = main([
        "--config", str(_CONFIGS / "calls_smoke.yaml"),
        "--run-dir", str(tmp_path),
        "--only", "grand",
    ])
    assert rc == 0
    assert (tmp_path / "checkpoints" / "grand_final.bin").exists()
    assert not (tmp_path / "checkpoints" / "tichu_final.bin").exists()
