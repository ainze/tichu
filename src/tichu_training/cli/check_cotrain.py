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
              seed: int | None = None, n_deals: int | None = None,
              held_out: str | None = None, tournament: bool = True) -> dict:
    """Export the chosen (default: latest) snapshot, tournament it vs `master`, and
    (if `held_out` given) run Move-Prediction-Eval. Returns `{iter, mean, ci, n,
    ship, matrix_path, seed, move_pred}` — `ship` is True iff the 95% CI lower bound
    clears 0 (the ADR-0034 ship bar). `seed`/`n_deals` override the config for
    robustness sweeps. `move_pred` (when run) reports play top-1 for cotrain vs
    master and **flags** a crater (>~3pp drop) — reported, NOT gating (Q9).
    `tournament=False` skips the heavy tournament (quick plausibility check only)."""
    run_dir = Path(config["run_dir"])
    eval_cfg = config["eval"]
    skill_decile = int(config.get("ppo", {}).get("skill_decile", 9))
    eval_seed = int(eval_cfg.get("seed", 0) if seed is None else seed)
    held_out = held_out if held_out is not None else eval_cfg.get("held_out")

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
        if tournament:
            n = len(positions) if positions is not None else eval_cfg.get("n_deals", "all")
            print(f"  tournament: {name} vs master ({n} deals, "
                  f"{int(eval_cfg.get('workers', 1))} workers)", flush=True)

    mean = lo = hi = n = None
    ship = False
    matrix_path = None
    if tournament:
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
        n = int(result.n(name, "master"))
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
                f"CI=[{lo:+.2f}, {hi:+.2f}]  n={n}  -> {verdict}", flush=True,
            )

    move_pred = None
    if held_out:
        move_pred = _move_prediction(
            agent_builders, name, held_out,
            max_decisions=eval_cfg.get("move_pred_max"),
            max_games=int(eval_cfg.get("move_pred_games", 20)),
            progress=progress,
        )

    return {
        "iter": it, "mean": mean, "ci": (lo, hi) if mean is not None else None,
        "n": n, "ship": ship, "matrix_path": str(matrix_path) if matrix_path else None,
        "seed": eval_seed, "move_pred": move_pred,
    }


# Human-plausibility leash (Q9): a co-trained policy that beats master but has
# drifted far from human play (cratered play top-1) is a degenerate winner. Flag,
# don't gate.
_MOVE_PRED_CRATER_PP = 0.03  # > 3pp play top-1 drop vs master -> flagged


def _load_eval_games(held_out: str, max_games: int):
    """Load games for Move-Prediction-Eval from EITHER a directory of `.tch` files
    OR the zstd `.tch` archive (`data/archive.zst`, streamed via `iter_archive` — no
    extraction). For the archive, take a deterministic strided sample of `max_games`
    so the games spread across the corpus (NB: there is no game-level held-out — the
    corpus held-out is a per-example hash; this measures cotrain's *drift vs master*
    on identical decisions, which is valid even on training games)."""
    from tichu_training.bsw.parser import parse_tch

    path = Path(held_out)
    if path.suffix == ".zst":
        from tichu_training.bsw.archive import iter_archive, list_game_ids
        ids = list_game_ids(path)
        if max_games and len(ids) > max_games:
            step = max(1, len(ids) // max_games)
            ids = ids[::step][:max_games]
        wanted = set(ids)
        return [parse_tch(text, game_id=gid) for gid, text in iter_archive(path, game_ids=wanted)]
    games = sorted(path.glob("*.tch"))
    if max_games:
        games = games[:max_games]
    return [parse_tch(p.read_text(encoding="utf-8"), game_id=p.stem) for p in games]


def _move_prediction(agent_builders, name, held_out, *, max_decisions, max_games, progress):
    """Move-Prediction-Eval: top-1/top-5 play-prediction accuracy of the cotrain
    snapshot vs `master` on held-out human games, and a crater flag on the play
    head. Reuses the validated `evaluate_move_prediction` (ADR-0025)."""
    from tichu_engine.legality import legal_actions_for
    from tichu_eval.move_prediction import decisions_from_game, evaluate_move_prediction

    games = _load_eval_games(held_out, max_games)
    # Q9's human-plausibility metric is the PLAY head; restrict to play decisions
    # that have a legal action (call/deal-time states fed to `act` hit the no-legal
    # fallback, and calls aren't an `act` decision anyway).
    decisions = [
        d for g in games for d in decisions_from_game(g)
        if d.decision_type == "play" and legal_actions_for(d.private_state)
    ]
    cap = None if max_decisions is None else int(max_decisions)

    out = {}
    for agent_name in ("master", name):
        agent = agent_builders[agent_name]()
        out[agent_name] = evaluate_move_prediction(agent, decisions, max_decisions=cap)

    master_play = out["master"].get("play", {}).get("top1")
    cotrain_play = out[name].get("play", {}).get("top1")
    drop = None if (master_play is None or cotrain_play is None) else master_play - cotrain_play
    flagged = bool(drop is not None and drop > _MOVE_PRED_CRATER_PP)

    if progress:
        print(f"  move-prediction ({len(games)} games, {len(decisions)} decisions):", flush=True)
        for agent_name in ("master", name):
            play = out[agent_name].get("play", {})
            t5 = play.get("top5")
            print(f"    {agent_name:>14s} play top1={play.get('top1', float('nan')):.3f}"
                  f"  top5={'n/a' if t5 is None else f'{t5:.3f}'}  n={play.get('n', 0)}", flush=True)
        if drop is not None:
            tag = "  *** FLAG: play top-1 cratered (>3pp) ***" if flagged else "  (within 3pp — ok)"
            print(f"    play top-1 drop vs master: {drop:+.3f}{tag}", flush=True)
    return {"by_agent": out, "play_top1_drop": drop, "flagged": flagged}


def main(argv=None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description="Offline strength read for co-training (ADR-0034)")
    parser.add_argument("--config", required=True, help="The co-training config YAML (run_dir + arch + eval).")
    parser.add_argument("--iter", type=int, default=None, help="Snapshot iteration to check (default: latest).")
    parser.add_argument("--seed", type=int, default=None, help="Override eval seed (for a robustness sweep of a marginal CI).")
    parser.add_argument("--n-deals", type=int, default=None, help="Override eval n_deals (tighten a marginal CI).")
    parser.add_argument("--held-out", default=None,
                        help="Dir of held-out .tch games -> run Move-Prediction-Eval (reported, not gating).")
    parser.add_argument("--move-pred-only", action="store_true",
                        help="Skip the tournament; only export + Move-Prediction-Eval (quick human-plausibility check).")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    result = run_check(
        config, iteration=args.iter, seed=args.seed, n_deals=args.n_deals,
        held_out=args.held_out, tournament=not args.move_pred_only,
    )
    return 0 if result["ship"] else 1


if __name__ == "__main__":
    sys.exit(main())
