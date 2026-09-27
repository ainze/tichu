"""Behavioral Drift Benchmark CLI — how a Subject Agent plays vs its BC.

Plays the Fixed-Opponent arms (or, with `--aa`, the A/A Null Run), saves the raw
per-Decision / per-Round log as Parquet, and writes the tidy `summary.csv` and a
self-contained `report.html`. `--from-log` re-summarises an existing log without
replaying a Round (new metric, different floor, more bootstrap draws).

  py -3.14 -m tichu_training.cli.drift_benchmark --config configs/drift_v7_iter15360.yaml
  py -3.14 -m tichu_training.cli.drift_benchmark --config ... --aa        # gate first
  py -3.14 -m tichu_training.cli.drift_benchmark --config ... --from-log

Config keys: `subject` / `bc` (agent specs as in `eval_matrix`: name, factory,
kwargs), `pool_seed`, `n_deals`, `workers`, `n_boot`, `floor`, `output_dir`, and an
optional `human` block — `source` (the zstd `.tch` archive, or a directory of
`.tch` files), `ratings` (for `min_decile`) and `groups` (each `name`, `n_rounds`,
optional `min_decile`, and `sample_games`: how many archive Games to spread the
sample over — enough to yield `n_rounds` qualifying Rounds). Human logs do not depend on the Subject: each is cached
under `<output_dir>/human/<group>/` and reused until deleted.
See CONTEXT.md §"Behavioral Drift Benchmark" / §"A/A Null Run".
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import yaml

import re

from tichu_eval.drift_arms import load_drift_log, run_aa_null, run_drift_arms, save_drift_log
from tichu_eval.drift_human import human_log
from tichu_eval.drift_metrics import summarise
from tichu_eval.drift_report import render_report
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.cli.eval_matrix import _build_agent

log = logging.getLogger("drift_benchmark")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Behavioral Drift Benchmark: Subject vs its BC.")
    p.add_argument("--config", required=True)
    p.add_argument("--aa", action="store_true",
                   help="run the A/A Null Run (BC vs BC on disjoint deals) into <output_dir>/aa")
    p.add_argument("--from-log", action="store_true",
                   help="re-summarise the saved log in the output dir; play nothing")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%H:%M:%S")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = Path(config["output_dir"]) / ("aa" if args.aa else "")
    out.mkdir(parents=True, exist_ok=True)
    if (out / config_path.name).resolve() != config_path.resolve():
        shutil.copy(config_path, out / config_path.name)

    n, seed, workers = int(config["n_deals"]), int(config.get("pool_seed", 0)), int(config.get("workers", 1))
    subject, bc = config["subject"], config["bc"]
    if args.from_log:
        drift_log = load_drift_log(out)
    else:
        positions = generate_full_position_pool(seed=seed, n=n)
        bc_builder = _builder(bc)
        if args.aa:
            # Pool seed s deals `s + i`: the second, disjoint set starts at s + n.
            log.info("A/A Null Run: %s on deals [%d, %d) vs [%d, %d)", bc["name"], seed, seed + n,
                     seed + n, seed + 2 * n)
            drift_log = run_aa_null(bc_builder, positions,
                                    generate_full_position_pool(seed=seed + n, n=n), workers=workers)
        else:
            log.info("drift arms: %s vs %s over %d deals, %d workers", subject["name"], bc["name"], n, workers)
            drift_log = run_drift_arms(_builder(subject), bc_builder, positions, workers=workers)
        save_drift_log(drift_log, out)

    references = [] if args.aa else _human_references(config.get("human"), Path(config["output_dir"]))
    panel = summarise(drift_log, n_boot=int(config.get("n_boot", 2000)), seed=seed,
                      floor=int(config.get("floor", 200)), references=references)
    panel.to_csv(out / "summary.csv", index=False)
    provenance = {
        "mode": "A/A Null Run (BC vs BC, disjoint deals)" if args.aa else "Fixed-Opponent",
        "subject": _describe(bc if args.aa else subject),
        "bc": _describe(bc),
        "deals": f"{n} (pool seed {seed}{f'; A/A second set seed {seed + n}' if args.aa else ''})",
        "bootstrap draws": config.get("n_boot", 2000),
        "opportunity floor": config.get("floor", 200),
        "git": _git_sha(),
    }
    (out / "report.html").write_text(
        render_report(panel, provenance,
                      title="A/A Null Run" if args.aa else f"Behavioral Drift: {subject['name']} vs BC"),
        encoding="utf-8")
    survivors = int(panel.bh_survives.sum())
    log.info("wrote %s (%d Δ survive BH)", out, survivors)
    return 0


def _human_references(spec: dict | None, output_dir: Path) -> list:
    """One reference log per configured human group, replayed once and cached."""
    if not spec:
        return []
    refs = []
    for group in spec["groups"]:
        cache = output_dir / "human" / re.sub(r"[^a-z0-9]+", "-", group["name"].lower()).strip("-")
        if (cache / "rounds.parquet").exists():
            log.info("human reference %r: cached at %s", group["name"], cache)
            ref = load_drift_log(cache)
        else:
            min_decile = group.get("min_decile")
            decile_of = _deciles(spec["ratings"]) if min_decile is not None else None
            log.info("human reference %r: replaying up to %d Rounds from %s", group["name"],
                     group["n_rounds"], spec["source"])
            games = _human_games(Path(spec["source"]), int(group.get("sample_games", 20_000)))
            ref = human_log(games, name=group["name"],
                            n_rounds=int(group["n_rounds"]), decile_of=decile_of, min_decile=min_decile)
            save_drift_log(ref, cache)
        refs.append(ref)
    return refs


def _human_games(source: Path, sample_games: int):
    """Parsed BSW games, lazily: every `.tch` in a directory, or a deterministic
    strided sample of `sample_games` Games spread evenly over the whole zstd
    archive (read in archive order; the caller stops once it has enough Rounds,
    so `sample_games` should be sized so that happens near the end)."""
    from tichu_training.bsw.archive import iter_archive, list_game_ids
    from tichu_training.bsw.parser import parse_tch

    if source.is_dir():
        texts = ((f.stem, f.read_text(encoding="utf-8", errors="replace"))
                 for f in sorted(source.glob("*.tch")))
    else:
        ids = list_game_ids(source)
        texts = iter_archive(source, game_ids=_sample_ids(ids, sample_games))
    for game_id, text in texts:
        try:
            yield parse_tch(text, game_id=game_id)
        except Exception:   # a malformed log is skipped, as materialisation does
            continue


def _sample_ids(ids: list[str], sample_games: int) -> set[str]:
    """`sample_games` ids at an even stride across the whole archive, so the human
    reference covers all of BSW history rather than one stretch of it."""
    stride = max(1, len(ids) // sample_games)
    return set(ids[::stride][:sample_games])


def _deciles(ratings_path) -> dict:
    import pandas as pd

    r = pd.read_parquet(ratings_path, columns=["player_handle", "skill_decile"])
    return dict(zip(r.player_handle, r.skill_decile))


def _builder(spec: dict):
    return partial(_build_agent, spec["factory"], **spec.get("kwargs", {}))


def _describe(spec: dict) -> str:
    kwargs = spec.get("kwargs", {})
    detail = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    return f"{spec['name']} ({spec['factory']}{': ' + detail if detail else ''})"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
