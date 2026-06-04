"""train_ppo CLI wiring (ADR-0029): warm-start from a BC Checkpoint, run the
self-play loop, and write a BC-byte-compatible Sharpened Checkpoint."""

import torch

from tichu_training.bc.heads import BCModel
from tichu_training.bc.training import load_checkpoint, save_checkpoint
from tichu_training.cli.train_ppo import run_ppo_training
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM

_MODEL_CFG = dict(
    skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16
)


def _write_bc_checkpoint(path):
    model = BCModel(FEATURIZER_OUTPUT_DIM, skill_buckets=10, **_MODEL_CFG)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(model, optimizer, step=0, path=path)


def test_run_ppo_training_warm_starts_and_writes_loadable_sharpened_checkpoint(tmp_path):
    torch.manual_seed(0)
    bc_path = tmp_path / "bc.bin"
    out_path = tmp_path / "sharpened.bin"
    _write_bc_checkpoint(bc_path)

    config = {
        "warm_start": str(bc_path),
        "out": str(out_path),
        "model": _MODEL_CFG,
        "critic": {"hidden": 16},
        "kl": {"coef": 1.0, "target": 0.02},
        "league": {"max_snapshots": 3, "snapshot_every": 1},
        "ppo": {
            "iterations": 2, "positions_per_iter": 2, "pool_seed": 0,
            "gamma": 1.0, "lam": 0.95, "clip_eps": 0.1, "vf_coef": 0.5,
            "ent_coef": 0.01, "ppo_epochs": 2, "policy_lr": 1e-4,
            "critic_lr": 1e-3, "skill_decile": 9, "learner_team": 0,
        },
    }

    result = run_ppo_training(config, progress=False)

    assert len(result["history"]) == 2
    assert out_path.exists()
    # Byte-compatible with a BC Checkpoint: a fresh same-shape BCModel loads it
    # (load_checkpoint raises on a version or shape mismatch).
    fresh = BCModel(FEATURIZER_OUTPUT_DIM, skill_buckets=10, **_MODEL_CFG)
    load_checkpoint(out_path, fresh)
    # The league grew via snapshots (frozen BC opponent + learner snapshots).
    assert result["league_size"] >= 2

    # Mid-run observability (the kill rule depends on it): per-iteration stats are
    # logged to CSV and an intermediate Checkpoint is persisted every snapshot_every.
    import csv

    log_rows = list(csv.DictReader(open(tmp_path / "ppo_log.csv", encoding="utf-8")))
    assert len(log_rows) == 2
    assert {"iter", "wall_s", "kl_to_bc", "entropy"} <= set(log_rows[0].keys())
    # snapshot_every=1 -> one persisted snapshot per iteration, each loadable.
    for it in (1, 2):
        snap = tmp_path / "snapshots" / f"iter_{it:05d}.bin"
        assert snap.exists()
    load_checkpoint(tmp_path / "snapshots" / "iter_00002.bin", BCModel(FEATURIZER_OUTPUT_DIM, skill_buckets=10, **_MODEL_CFG))


def test_run_ppo_training_with_critic_warmup_completes_and_writes_checkpoint(tmp_path):
    """Critic warm-up (ADR-0029 §value): fitting the critic under the frozen BC
    policy before the policy moves must run cleanly and still yield a loadable
    Sharpened Checkpoint."""
    torch.manual_seed(0)
    bc_path = tmp_path / "bc.bin"
    out_path = tmp_path / "sharpened.bin"
    _write_bc_checkpoint(bc_path)

    config = {
        "warm_start": str(bc_path),
        "out": str(out_path),
        "model": _MODEL_CFG,
        "critic": {"hidden": 16, "warmup_iters": 2, "warmup_epochs": 2},
        "kl": {"coef": 1.0, "target": 0.04},
        "league": {"max_snapshots": 3, "snapshot_every": 1},
        "ppo": {
            "iterations": 2, "positions_per_iter": 2, "pool_seed": 0,
            "gamma": 1.0, "lam": 0.95, "clip_eps": 0.1, "vf_coef": 0.5,
            "ent_coef": 0.03, "ppo_epochs": 2, "policy_lr": 1e-4,
            "critic_lr": 3e-3, "skill_decile": 9, "learner_team": 0,
        },
    }

    result = run_ppo_training(config, progress=False)

    assert len(result["history"]) == 2
    assert out_path.exists()
    load_checkpoint(out_path, BCModel(FEATURIZER_OUTPUT_DIM, skill_buckets=10, **_MODEL_CFG))
