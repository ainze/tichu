"""`eval_matrix` CLI — config-driven tournament + paired-difference matrix.

Two modes:
  * `tournament` (default): all-vs-all self-play matrix with bootstrap CIs.
  * `move_prediction`: top-1 / top-5 accuracy vs held-out human games.
"""

import argparse
import csv
import itertools
import logging
import shutil
import sys
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from tichu_eval.starting_position_pool import load_starting_position_pool
from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.behavioral import run_behavioral_profiles
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

# Side-effect import: registers the `search` factory (SearchAgent, Phase-2 PIMC —
# ADR-0030) in the registry so configs can reference `factory: search` and spawned
# workers resolve it. Torch-free at import (SearchAgent lazy-loads MLAgent).
import tichu_training.search.agent  # noqa: F401,E402
# Registers `forced_press` + `forced_bomb` (ADR-0031 premise tests).
import tichu_training.search.forced_press  # noqa: F401,E402


log = logging.getLogger("eval_matrix")


def _build_agent(factory: str, **kwargs):
    """Agent builder threaded to parallel workers (see `run_full_tournament`).

    Wrapping `build_agent` in a function that lives *in this module* is what makes
    the `ml` factory resolvable inside a spawned worker: when the worker unpickles
    a `partial(_build_agent, ...)` it imports this module, whose top-level
    `import tichu_inference.ml_agent` registers `MLAgent`. (The baselines
    self-register on the `tichu_ml` package import.) Routing through
    `partial(build_agent, ...)` directly would only import `tichu_ml.registry` in
    the worker, leaving `ml` unregistered — `build_agent` would then `KeyError`
    inside the Pool initializer and deadlock the run."""
    return build_agent(factory, **kwargs)


def _slice_positions(positions, n_cap, n_offset=0):
    """Select the ``[n_offset, n_offset + n_cap)`` window of the Pool. ``n_offset``
    lets parallel jobs cover disjoint Position ranges — behavioral mode is serial,
    so we fan out across processes by slice and pool the counts afterwards."""
    if n_offset:
        positions = positions[int(n_offset):]
    return positions[: int(n_cap)] if n_cap is not None else positions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run an all-vs-all tournament or held-out move-prediction eval.",
    )
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.add_argument(
        "--mode", choices=("tournament", "move_prediction", "behavioral"),
        default="tournament",
        help="Evaluation mode (default: tournament). 'behavioral' profiles each "
             "agent's play style (bomb / call / trick-win rates) in self-play.",
    )
    p.add_argument(
        "--held-out", metavar="DIR",
        help="Directory of .tch files (required for --mode move_prediction)",
    )
    p.add_argument(
        "--n-offset", type=int, default=0,
        help="skip the first N positions before applying n_deals — lets parallel "
             "jobs cover disjoint Pool slices (mainly for serial behavioral runs)",
    )
    p.add_argument(
        "--output", default=None,
        help="override the config's output path (so parallel slices write to "
             "distinct files for later pooling)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True,
        help="Show a tqdm progress bar over the tournament "
             "(auto-disabled when stderr is not a TTY). Use --no-progress for "
             "clean piped/cron logs.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    output_path = Path(args.output or config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_copy = output_path.parent / config_path.name
    if config_copy.resolve() != config_path.resolve():
        shutil.copy(config_path, config_copy)

    # Builder callables (name -> zero-arg factory). `partial` captures each
    # agent's factory + kwargs so a parallel worker can rebuild it from scratch
    # (torch models do not pickle/fork cleanly). The serial path just invokes
    # them once. `build_agent` is module-level, so the partials are picklable
    # for spawn-based multiprocessing on Windows.
    agent_builders = {
        spec["name"]: partial(_build_agent, spec["factory"], **spec.get("kwargs", {}))
        for spec in config["agents"]
    }

    if args.mode == "tournament":
        return _run_tournament_mode(
            agent_builders, config, output_path,
            show_progress=args.progress, n_offset=args.n_offset,
        )
    if args.mode == "move_prediction":
        if not args.held_out:
            p.error("--mode move_prediction requires --held-out DIR")
        return _run_move_prediction_mode(
            agent_builders, Path(args.held_out), config, output_path
        )
    if args.mode == "behavioral":
        return _run_behavioral_mode(
            agent_builders, config, output_path, n_offset=args.n_offset
        )
    raise AssertionError(f"unknown mode: {args.mode}")


def _run_tournament_mode(
    agent_builders, config, output_path: Path, *,
    show_progress: bool = True, n_offset: int = 0,
) -> int:
    variant = config.get("variant", "full_strength")
    pool_path = Path(config["starting_position_pool"])
    n_cap = config.get("n_deals")
    bootstrap_iters = int(config.get("bootstrap_iters", 1000))
    seed = int(config.get("seed", 0))
    workers = int(config.get("workers", 1))

    if variant == "full_strength":
        positions = _slice_positions(load_full_position_pool(pool_path), n_cap, n_offset)
        # One matrix-wide bar over every unit of work: each unordered agent pair
        # plays every Position once. The tournament reports progress through the
        # `progress` hook (one tick per Position); `logging_redirect_tqdm` keeps
        # the surviving INFO lines from garbling the bar's \r line. `--no-progress`
        # (or a non-TTY stderr) skips the bar entirely and passes progress=None.
        n_pairs = len(list(itertools.combinations(sorted(agent_builders), 2)))
        total_units = n_pairs * len(positions)
        # disable=None lets tqdm auto-suppress the bar on a non-TTY stderr
        # (piped/cron logs stay clean); --no-progress forces it off everywhere.
        with logging_redirect_tqdm(), tqdm(
            total=total_units, unit="pos", desc="tournament",
            dynamic_ncols=True, disable=(True if not show_progress else None),
        ) as bar:
            result = run_full_tournament(
                agent_builders, positions,
                bootstrap_iters=bootstrap_iters, seed=seed, workers=workers,
                progress=(None if not show_progress else bar.update),
            )
    elif variant == "play_strength":
        # Play-strength is serial-only; build the agents once up front.
        agents = {name: build() for name, build in agent_builders.items()}
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
    _pretty_print(result, sorted(agent_builders.keys()))
    return 0


def _run_move_prediction_mode(agent_builders, held_out_dir: Path, config, output_path: Path) -> int:
    agents = {name: build() for name, build in agent_builders.items()}
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


_BEHAVIORAL_COLS = [
    "bomb_per_round", "bomb_when_legal_rate", "tichu_call_rate",
    "tichu_success_rate", "grand_call_rate", "grand_success_rate",
    "trick_win_rate", "out_first_rate", "slam_rate", "caller_passivity_rate",
    "caller_bomb_passivity_rate",
]


def _run_behavioral_mode(agent_builders, config, output_path: Path, *, n_offset: int = 0) -> int:
    """Per-agent behavioral profile (self-play) over the Full-strength Pool."""
    pool_path = Path(config["starting_position_pool"])
    n_cap = config.get("n_deals")
    positions = _slice_positions(load_full_position_pool(pool_path), n_cap, n_offset)

    names = sorted(agent_builders)
    log.info("behavioral profile: %d agents x %d positions (self-play, 4 seats)",
             len(names), len(positions))
    builders = {name: agent_builders[name] for name in names}
    profiles = run_behavioral_profiles(builders, positions)

    rows = []
    for name in names:
        row = {"agent": name, **profiles[name].as_row()}
        rows.append(row)

    # Console table: one agent per row, the headline rates as columns.
    w = max(8, max(len(n) for n in names) + 1)
    print("agent".rjust(w) + "".join(c.replace("_rate", "").replace("_", " ")[:11].rjust(13)
                                     for c in _BEHAVIORAL_COLS))
    for name in names:
        p = profiles[name]
        line = name.rjust(w)
        for col in _BEHAVIORAL_COLS:
            line += f"{getattr(p, col):.3f}".rjust(13)
        print(line)
    print(f"\n(seat-rounds per agent: {profiles[names[0]].seat_rounds}; "
          "trick_win/out_first parity = 0.250)")

    fieldnames = ["agent", "seat_rounds", "bomb_per_round", "bomb_when_legal_rate",
                  "bombs_played", "bomb_legal_decisions", "grand_call_rate",
                  "grand_success_rate", "tichu_call_rate", "tichu_success_rate",
                  "trick_win_rate", "out_first_rate", "slam_rate",
                  "caller_passivity_rate", "caller_pass_opportunities",
                  "caller_bomb_passivity_rate", "caller_pass_bomb_opportunities"]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
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
