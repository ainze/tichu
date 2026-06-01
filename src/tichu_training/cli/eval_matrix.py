"""`eval_matrix` CLI — config-driven tournament + paired-difference matrix.

Two modes:
  * `tournament` (default): all-vs-all self-play matrix with bootstrap CIs.
  * `move_prediction`: top-1 / top-5 accuracy vs held-out human games.
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tichu_eval.starting_position_pool import load_starting_position_pool
from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.move_prediction import (
    decisions_from_game,
    evaluate_move_prediction,
)
from tichu_eval.tournament import MatrixResult, run_full_tournament, run_tournament
from tichu_ml.registry import build_agent
from tichu_training.bsw.parser import parse_tch

# Side-effect import: registers the torch-backed `ml` agent (MLAgent) in the
# agent registry so configs can reference `factory: ml`. Kept here rather than
# in `tichu_ml.__init__` so the registry stays torch-free for engine-only
# callers; the eval CLI is already an ML entry point.
import tichu_inference.ml_agent  # noqa: F401,E402


log = logging.getLogger("eval_matrix")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run an all-vs-all tournament or held-out move-prediction eval.",
    )
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.add_argument(
        "--mode", choices=("tournament", "move_prediction"), default="tournament",
        help="Evaluation mode (default: tournament)",
    )
    p.add_argument(
        "--held-out", metavar="DIR",
        help="Directory of .tch files (required for --mode move_prediction)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_copy = output_path.parent / config_path.name
    if config_copy.resolve() != config_path.resolve():
        shutil.copy(config_path, config_copy)

    agents = {
        spec["name"]: build_agent(spec["factory"], **spec.get("kwargs", {}))
        for spec in config["agents"]
    }

    if args.mode == "tournament":
        return _run_tournament_mode(agents, config, output_path)
    if args.mode == "move_prediction":
        if not args.held_out:
            p.error("--mode move_prediction requires --held-out DIR")
        return _run_move_prediction_mode(
            agents, Path(args.held_out), config, output_path
        )
    raise AssertionError(f"unknown mode: {args.mode}")


def _run_tournament_mode(agents, config, output_path: Path) -> int:
    variant = config.get("variant", "full_strength")
    pool_path = Path(config["starting_position_pool"])
    n_cap = config.get("n_deals")
    bootstrap_iters = int(config.get("bootstrap_iters", 1000))
    seed = int(config.get("seed", 0))

    if variant == "full_strength":
        positions = load_full_position_pool(pool_path)
        positions = positions[: int(n_cap)] if n_cap is not None else positions
        result = run_full_tournament(
            agents, positions, bootstrap_iters=bootstrap_iters, seed=seed
        )
    elif variant == "play_strength":
        deals = load_starting_position_pool(pool_path)
        deals = deals[: int(n_cap)] if n_cap is not None else deals
        result = run_tournament(
            agents, deals, bootstrap_iters=bootstrap_iters, seed=seed
        )
    else:
        raise ValueError(
            f"unknown tournament variant {variant!r} "
            "(expected 'full_strength' or 'play_strength')"
        )

    _write_matrix(result, output_path)
    _pretty_print(result, sorted(agents.keys()))
    return 0


def _run_move_prediction_mode(agents, held_out_dir: Path, config, output_path: Path) -> int:
    games = []
    for path in sorted(held_out_dir.glob("*.tch")):
        games.append(parse_tch(path.read_text(encoding="utf-8"), game_id=path.stem))
    if not games:
        log.warning("no .tch files found in %s", held_out_dir)

    max_decisions = config.get("max_decisions")
    if max_decisions is not None:
        max_decisions = int(max_decisions)

    rows: list[dict] = []
    for name, agent in agents.items():
        decisions: list = []
        for game in games:
            decisions.extend(decisions_from_game(game))
        out = evaluate_move_prediction(agent, decisions, max_decisions=max_decisions)
        for decision_type, stats in sorted(out.items()):
            rows.append({
                "agent": name,
                "decision_type": decision_type,
                "top1": stats["top1"],
                "top5": stats["top5"],
                "n": stats["n"],
            })
            top5_str = "n/a" if stats["top5"] is None else f"{stats['top5']:.3f}"
            print(
                f"{name:>10s}  {decision_type:>12s}  "
                f"top1={stats['top1']:.3f}  top5={top5_str}  n={stats['n']}"
            )

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["agent", "decision_type", "top1", "top5", "n"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return 0


def _write_matrix(result: MatrixResult, path: Path) -> None:
    rows = result.rows
    table = pa.table({
        "agent_a": pa.array([r["agent_a"] for r in rows], type=pa.string()),
        "agent_b": pa.array([r["agent_b"] for r in rows], type=pa.string()),
        "mean": pa.array([r["mean"] for r in rows], type=pa.float64()),
        "ci_lower": pa.array([r["ci_lower"] for r in rows], type=pa.float64()),
        "ci_upper": pa.array([r["ci_upper"] for r in rows], type=pa.float64()),
        "n": pa.array([r["n"] for r in rows], type=pa.int32()),
        "call_bonus_mean": pa.array(
            [r.get("call_bonus_mean", 0.0) for r in rows], type=pa.float64()
        ),
    })
    pq.write_table(table, path)


def _pretty_print(result: MatrixResult, names: list[str]) -> None:
    width = max(8, max(len(n) for n in names) + 2)
    header = "".rjust(width) + "".join(name.rjust(width) for name in names)
    print(header)
    for a in names:
        row = a.rjust(width)
        for b in names:
            row += f"{result.mean(a, b):+.1f}".rjust(width)
        print(row)
    print()
    print("95% bootstrap CI (lower, upper):")
    for a in names:
        for b in names:
            if a == b:
                continue
            lo, hi = result.ci(a, b)
            print(f"  {a} vs {b}: mean={result.mean(a, b):+.2f}  CI=[{lo:+.2f}, {hi:+.2f}]  n={result.n(a, b)}")


if __name__ == "__main__":
    sys.exit(main())
