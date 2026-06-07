"""train_cotrain CLI wiring (ADR-0034): warm-start the four nets from their BC
Checkpoints, run the co-training loop, write a Resume Bundle every iteration, and
resume losslessly from it on a re-run (the owner's headline stop/continue ask)."""

import csv
from pathlib import Path

import torch

from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.heads import BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.bc.training import save_checkpoint
from tichu_training.cli.train_cotrain import run_cotrain_training
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM as _D

_MODEL = dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16)


def _write(net, path):
    save_checkpoint(net, torch.optim.Adam(net.parameters()), step=0, path=str(path))


def _config(tmp_path):
    warm = {
        "play": tmp_path / "play.bin",
        "schupfen": tmp_path / "schupfen.bin",
        "tichu": tmp_path / "tichu.bin",
        "grand": tmp_path / "grand.bin",
    }
    _write(BCModel(_D, skill_buckets=10, **_MODEL), warm["play"])
    _write(SchupfenNetwork(_D, skill_dim=8, hidden=16), warm["schupfen"])
    _write(TichuCallNetwork(_D, skill_dim=8, hidden=16), warm["tichu"])
    _write(GrandTichuCallNetwork(_D, skill_dim=8, hidden=16), warm["grand"])
    return {
        "warm_start": {k: str(v) for k, v in warm.items()},
        "run_dir": str(tmp_path / "run"),
        "model": _MODEL,
        "schupfen_model": {"skill_dim": 8, "hidden": 16},
        "call_model": {"skill_dim": 8, "hidden": 16},
        "critic": {"hidden": 16},
        "perfect_info": True,
        "kl": {dt: {"coef": 0.5, "target": 0.02} for dt in ("play", "schupfen", "tichu", "grand")},
        "entropy": {dt: 0.01 for dt in ("play", "schupfen", "tichu", "grand")},
        "snapshot_every": 2,
        "ppo": {
            "iterations": 2, "positions_per_iter": 2, "pool_seed": 0,
            "gamma": 1.0, "lam": 0.95, "clip_eps": 0.1, "vf_coef": 0.5,
            "ppo_epochs": 1, "policy_lr": 1e-3, "critic_lr": 1e-3,
            "skill_decile": 9, "learner_team": 0,
        },
    }


def test_cotrain_cli_warm_starts_runs_and_writes_a_resume_bundle(tmp_path):
    torch.manual_seed(0)
    config = _config(tmp_path)

    result = run_cotrain_training(config, progress=False)

    assert result["start_iter"] == 0 and result["final_iter"] == 2
    assert len(result["history"]) == 2
    # The Resume Bundle was written, and per-net dials are logged for attribution.
    assert Path(result["bundle_path"]).exists()
    rows = list(csv.DictReader(open(Path(config["run_dir"]) / "ppo_log.csv", encoding="utf-8")))
    assert len(rows) == 2
    assert {"play_kl", "schupfen_kl", "tichu_kl_coef", "value_loss"} <= set(rows[0].keys())
    # Serving snapshots for every net at snapshot_every=2.
    for dt in ("play", "schupfen", "tichu", "grand"):
        assert (Path(config["run_dir"]) / "snapshots" / f"iter_00002_{dt}.bin").exists()


def test_cotrain_cli_resumes_from_the_bundle_on_rerun(tmp_path):
    torch.manual_seed(0)
    config = _config(tmp_path)
    run_cotrain_training(config, progress=False)  # iters 0,1 -> bundle at iteration 2

    # Re-run the SAME command with a higher target: it must auto-resume at iter 2
    # (not redo 0,1) and continue to 4 — the idempotent start/continue contract.
    config["ppo"]["iterations"] = 4
    result = run_cotrain_training(config, progress=False)

    assert result["start_iter"] == 2 and result["final_iter"] == 4
    assert len(result["history"]) == 2  # only the 2 NEW iterations ran
    rows = list(csv.DictReader(open(Path(config["run_dir"]) / "ppo_log.csv", encoding="utf-8")))
    assert [int(r["iter"]) for r in rows] == [0, 1, 2, 3]


def test_cotrain_cli_restart_ignores_the_bundle(tmp_path):
    torch.manual_seed(0)
    config = _config(tmp_path)
    run_cotrain_training(config, progress=False)

    # --restart starts fresh even though a bundle exists.
    result = run_cotrain_training(config, restart=True, progress=False)
    assert result["start_iter"] == 0 and result["final_iter"] == 2
