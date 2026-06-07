"""check_cotrain — offline strength read for a co-training run (ADR-0034).

Exports the latest (or a chosen) co-training snapshot to TorchScript, runs the
seat-swap **Tournament** vs `master` over the Full-strength Pool, and reports the
bootstrap CI. The **ship bar is CI 95% > 0** (any positive); Move-Prediction-Eval
is reported (when a held-out dir is given) and *flagged*, not gating (ADR-0034).

Deliberately a SEPARATE command from training: the Tournament is the heaviest job
you own and the play-only perfect-info run OOM-died at iter 163 — keeping it out of
the multi-day loop is what stops a strength read from killing the run.

    py -m tichu_training.cli.check_cotrain --config configs/cotrain_v5.yaml
    py -m tichu_training.cli.check_cotrain --config configs/cotrain_v5.yaml --iter 1500
"""

import argparse
import sys
from functools import partial
from pathlib import Path

import torch

from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.tournament import run_full_tournament
from tichu_export.torchscript import export_torchscript
from tichu_ml.registry import build_agent
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.training import load_checkpoint
from tichu_training.cli.train_cotrain import _build_models
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION

# Side-effect import: registers the `ml` agent factory (and makes it resolvable in
# spawned tournament workers, mirroring eval_matrix).
import tichu_inference.ml_agent  # noqa: F401,E402

# snapshot suffix -> exported TorchScript filename (the names MLAgent expects).
_EXPORT_NAMES = {
    "play": "policy.pt", "schupfen": "schupfen.pt",
    "tichu": "tichu_call.pt", "grand": "grand_tichu_call.pt",
}


def _build_agent(factory, **kwargs):
    return build_agent(factory, **kwargs)


def _latest_snapshot(snapshots_dir) -> int | None:
    """The highest iteration N for which ALL FOUR `iter_{N:05d}_{net}.bin` exist
    (a half-written snapshot from a kill mid-save is skipped, not chosen)."""
    snapshots_dir = Path(snapshots_dir)
    iters: dict[int, set[str]] = {}
    for path in snapshots_dir.glob("iter_*_*.bin"):
        stem = path.stem  # iter_00025_play
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        iters.setdefault(int(parts[1]), set()).add(parts[2])
    complete = [n for n, nets in iters.items() if nets >= {"play", "schupfen", "tichu", "grand"}]
    return max(complete) if complete else None


def export_nets(config, *, snapshot_prefix: str, out_dir: str, progress: bool = False) -> dict:
    """Export a snapshot's four nets (`{snapshot_prefix}_{net}.bin`) to TorchScript
    under `out_dir`, returning the MLAgent kwargs `{checkpoint_path, schupfen_path,
    tichu_call_path, grand_call_path}`. Reuses `train_cotrain._build_models` so the
    arch matches the run's config exactly. With `progress`, prints each source
    checkpoint loaded and its export destination (so you can confirm the right
    snapshot was picked up)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = _build_models(config)
    example = (torch.randn(1, FEATURIZER_OUTPUT_DIM), torch.tensor([0], dtype=torch.long))
    paths: dict[str, str] = {}
    for net_key, model in models.items():
        src = f"{snapshot_prefix}_{net_key}.bin"
        load_checkpoint(src, model)
        dest = out / _EXPORT_NAMES[net_key]
        export_torchscript(
            model, example_inputs=example,
            featurizer_version=FEATURIZER_VERSION,
            # Only the play policy consumes the play Action Space; standalone nets
            # carry an empty stamp (matches export_model).
            action_space_version=ACTION_SPACE_VERSION if net_key == "play" else "",
            output_path=dest,
        )
        if progress:
            print(f"  export {net_key:8s} {src}  ->  {dest}", flush=True)
        paths[net_key] = str(dest)
    return {
        "checkpoint_path": paths["play"],
        "schupfen_path": paths["schupfen"],
        "tichu_call_path": paths["tichu"],
        "grand_call_path": paths["grand"],
    }


def run_check(config, *, iteration: int | None = None, positions=None, progress: bool = True,
              seed: int | None = None, n_deals: int | None = None) -> dict:
    """Export the chosen (default: latest) snapshot and tournament it vs `master`.
    Returns `{iter, mean, ci, n, ship, matrix_path, seed}` — `ship` is True iff the
    95% CI lower bound clears 0 (the ADR-0034 ship bar). `seed` / `n_deals` override
    the config for robustness sweeps (a marginal CI should hold across eval seeds)."""
    run_dir = Path(config["run_dir"])
    eval_cfg = config["eval"]
    skill_decile = int(config.get("ppo", {}).get("skill_decile", 9))
    eval_seed = int(eval_cfg.get("seed", 0) if seed is None else seed)

    it = iteration if iteration is not None else _latest_snapshot(run_dir / "snapshots")
    if it is None:
        raise FileNotFoundError(f"no complete snapshot under {run_dir / 'snapshots'}")

    if progress:
        chosen = "explicit --iter" if iteration is not None else "latest complete"
        print(f"check: snapshot iter {it} ({chosen}) from {run_dir / 'snapshots'}", flush=True)

    cotrain = export_nets(
        config,
        snapshot_prefix=str(run_dir / "snapshots" / f"iter_{it:05d}"),
        out_dir=str(run_dir / "export" / f"iter_{it:05d}"),
        progress=progress,
    )
    name = f"cotrain_{it:05d}"
    agent_builders = {
        "master": partial(_build_agent, "ml", skill_decile=skill_decile, **eval_cfg["master"]),
        name: partial(_build_agent, "ml", skill_decile=skill_decile, **cotrain),
    }

    if progress:
        m = eval_cfg["master"]
        for label, key in (("policy", "checkpoint_path"), ("schupfen", "schupfen_path"),
                           ("tichu", "tichu_call_path"), ("grand", "grand_call_path")):
            print(f"  master {label:8s} {m[key]}", flush=True)
        n = len(positions) if positions is not None else eval_cfg.get("n_deals", "all")
        print(f"  tournament: {name} vs master ({n} deals, "
              f"{int(eval_cfg.get('workers', 1))} workers)", flush=True)

    if positions is None:
        positions = load_full_position_pool(Path(eval_cfg["starting_position_pool"]))
        cap = eval_cfg.get("n_deals") if n_deals is None else n_deals
        if cap is not None:
            positions = positions[: int(cap)]

    result = run_full_tournament(
        agent_builders, positions,
        bootstrap_iters=int(eval_cfg.get("bootstrap_iters", 1000)),
        seed=eval_seed,
        workers=int(eval_cfg.get("workers", 1)),
        progress=None,
    )

    mean = float(result.mean(name, "master"))
    lo, hi = (float(x) for x in result.ci(name, "master"))
    ship = lo > 0.0

    check_dir = run_dir / "check"
    check_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = check_dir / f"iter_{it:05d}_tournament.parquet"
    from tichu_training.cli.eval_matrix import _write_matrix
    _write_matrix(result, matrix_path)

    if progress:
        verdict = "*** SHIP: CI clears 0 ***" if ship else "not yet (CI spans 0)"
        print(
            f"check iter {it} (seed {eval_seed}): {name} vs master  mean={mean:+.2f}  "
            f"CI=[{lo:+.2f}, {hi:+.2f}]  n={result.n(name, 'master')}  -> {verdict}",
            flush=True,
        )
    return {
        "iter": it, "mean": mean, "ci": (lo, hi),
        "n": int(result.n(name, "master")), "ship": ship,
        "matrix_path": str(matrix_path), "seed": eval_seed,
    }


def main(argv=None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description="Offline strength read for co-training (ADR-0034)")
    parser.add_argument("--config", required=True, help="The co-training config YAML (run_dir + arch + eval).")
    parser.add_argument("--iter", type=int, default=None, help="Snapshot iteration to check (default: latest).")
    parser.add_argument("--seed", type=int, default=None, help="Override eval seed (for a robustness sweep of a marginal CI).")
    parser.add_argument("--n-deals", type=int, default=None, help="Override eval n_deals (tighten a marginal CI).")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    result = run_check(config, iteration=args.iter, seed=args.seed, n_deals=args.n_deals)
    return 0 if result["ship"] else 1


if __name__ == "__main__":
    sys.exit(main())
