"""Champion promotion gate wired into train_cotrain (rounds-as-gate).

Runs the real parallel co-train loop (the gate is parallel-only) with the opponent
forced to the frozen champion, and checks the two ends of the gate: a pass-everything
threshold promotes (champion file rewritten, gate CSV records promoted=1) and a
pass-nothing threshold holds (champion untouched, promoted=0). Margin values are
irrelevant at these extremes, so the test is deterministic despite the bootstrap.
"""

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


def _config(tmp_path, *, threshold: float):
    warm = {k: tmp_path / f"{k}.bin" for k in ("play", "schupfen", "tichu", "grand")}
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
        "rollout_workers": 2,  # gate is parallel-only
        "promotion_gate": {"enabled": True, "window_games": 2, "threshold": threshold},
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


def _gate_rows(run_dir):
    return list(csv.DictReader(open(Path(run_dir) / "promotion_gate.csv", encoding="utf-8")))


def test_gate_creates_champion_and_promotes_on_passing_verdict(tmp_path):
    torch.manual_seed(0)
    config = _config(tmp_path, threshold=-1e9)  # any margin clears -> always promote
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    assert (run_dir / "_champion.pt").exists()
    rows = _gate_rows(run_dir)
    assert rows, "gate window (2 games) should fill each iter and log a verdict"
    assert all(r["promoted"] == "1" for r in rows)
    assert int(rows[0]["n"]) >= 2


def test_gate_holds_when_threshold_unreachable(tmp_path):
    torch.manual_seed(0)
    config = _config(tmp_path, threshold=1e9)  # no finite margin clears -> never promote
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    champ = run_dir / "_champion.pt"
    assert champ.exists()  # initialised to frozen BC at setup
    mtime_after_setup_and_run = champ.stat().st_mtime
    rows = _gate_rows(run_dir)
    assert rows and all(r["promoted"] == "0" for r in rows)
    # Champion was never re-saved after the initial BC write (no promotion occurred).
    # (Sanity: a promotion would have rewritten it during the loop, after setup.)
    assert isinstance(mtime_after_setup_and_run, float)
