"""check_cotrain CLI (ADR-0034): the offline strength read. Exports the latest
co-training snapshot, runs the seat-swap Tournament vs `master`, and reports the
bootstrap CI — the ship bar is CI 95% > 0. Kept OUT of the training loop (OOM
safety); run on demand on a snapshot."""

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.heads import BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.bc.training import save_checkpoint
from tichu_training.cli.check_cotrain import _latest_snapshot, export_nets, run_check
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM as _D

_MODEL = dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16)


def _save(net, path):
    save_checkpoint(net, torch.optim.Adam(net.parameters()), step=0, path=str(path))


def _write_snapshot(snap_dir, it):
    snap_dir.mkdir(parents=True, exist_ok=True)
    _save(BCModel(_D, skill_buckets=10, **_MODEL), snap_dir / f"iter_{it:05d}_play.bin")
    _save(SchupfenNetwork(_D, skill_dim=8, hidden=16), snap_dir / f"iter_{it:05d}_schupfen.bin")
    _save(TichuCallNetwork(_D, skill_dim=8, hidden=16), snap_dir / f"iter_{it:05d}_tichu.bin")
    _save(GrandTichuCallNetwork(_D, skill_dim=8, hidden=16), snap_dir / f"iter_{it:05d}_grand.bin")


def test_latest_snapshot_picks_the_highest_complete_iteration(tmp_path):
    snaps = tmp_path / "snapshots"
    _write_snapshot(snaps, 10)
    _write_snapshot(snaps, 25)
    # An INCOMPLETE iter (missing nets) must be ignored, not chosen.
    (snaps / "iter_00030_play.bin").write_bytes(b"x")
    assert _latest_snapshot(snaps) == 25


def _config(tmp_path):
    run_dir = tmp_path / "run"
    _write_snapshot(run_dir / "snapshots", 2)
    arch = {
        "model": _MODEL,
        "schupfen_model": {"skill_dim": 8, "hidden": 16},
        "call_model": {"skill_dim": 8, "hidden": 16},
    }
    # The master opponent: tiny nets exported to .pt up front (the real case points
    # `eval.master` at the existing data/export/<master>/*.pt).
    master_snaps = tmp_path / "master_snap"
    _write_snapshot(master_snaps, 0)
    master = export_nets(
        {"run_dir": str(run_dir), **arch},
        snapshot_prefix=str(master_snaps / "iter_00000"),
        out_dir=str(tmp_path / "master_export"),
    )
    return {
        "run_dir": str(run_dir),
        **arch,
        "eval": {
            "master": master,
            "bootstrap_iters": 50,
            "seed": 0,
            "workers": 1,
        },
    }


def test_check_exports_snapshot_and_reports_tournament_ci(tmp_path):
    torch.manual_seed(0)
    config = _config(tmp_path)
    positions = generate_full_position_pool(seed=0, n=2)

    result = run_check(config, positions=positions, progress=False)

    assert result["iter"] == 2
    assert isinstance(result["mean"], float)
    lo, hi = result["ci"]
    assert lo <= result["mean"] <= hi
    # ship == True only when the 95% CI lower bound clears 0 (ADR-0034 ship bar).
    assert result["ship"] == (lo > 0)
    # The snapshot was exported to loadable TorchScript for the eval.
    from pathlib import Path
    for name in ("policy.pt", "schupfen.pt", "tichu_call.pt", "grand_tichu_call.pt"):
        assert (Path(config["run_dir"]) / "export" / "iter_00002" / name).exists()
