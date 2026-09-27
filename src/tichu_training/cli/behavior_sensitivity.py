"""Behavior Sensitivity Probe CLI — which behaviors, nudged, change strength?

Three steps, in pre-registered order
(docs/notes/2026-09-27-behavior-sensitivity-preregistration.md):

  calibrate  baseline play on a separate deal set, logging every lever's logit
             gap; freezes δ± per lever into <out>/calibration/deltas.yaml. No EV.
  run        the reference arm (unbiased champion vs opponent), the δ = 0
             identity check, then every treatment arm (lever × ±δ). Resumable:
             an arm whose log already exists is not replayed.
  summarise  pairs each arm with the reference -> results.csv, the per-arm drift
             panels, and verdict.md (Holm across the arms).

  py -3.14 -m tichu_training.cli.behavior_sensitivity calibrate --config configs/behavior_sensitivity_v7_iter15360.yaml
  py -3.14 -m tichu_training.cli.behavior_sensitivity run       --config ...
  py -3.14 -m tichu_training.cli.behavior_sensitivity summarise --config ...

Every arm is played with `workers > 1` — an ML agent's choice between concrete
realisations of one Intent follows the hand's iteration order, which pickling a
Position to a worker rebuilds, so serial and parallel runs are not the same games
(parallel runs all are). The identity check guards the pairing.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tichu_eval.behavior_sensitivity import arm_effect, holm, pair_arms, verdicts
from tichu_eval.drift_arms import load_drift_log, run_fixed_opponent_arm, save_drift_log
from tichu_eval.drift_metrics import summarise
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.cli.eval_matrix import _build_agent
from tichu_training.search.behavior_bias import choose_delta
from tichu_training.search.behavior_bias_agent import collect_lever_gaps

log = logging.getLogger("behavior_sensitivity")

SIGNS = {"plus": "+", "minus": "-"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Behavior Sensitivity Probe.")
    p.add_argument("step", choices=("calibrate", "run", "summarise"))
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true",
                   help="calibrate: overwrite frozen deltas (only before `run` has started)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%H:%M:%S")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / config_path.name).resolve() != config_path.resolve():
        shutil.copy(config_path, out / config_path.name)
    return {"calibrate": calibrate, "run": run, "summarise": summarise_step}[args.step](
        config, out, force=args.force)


# --- builders -------------------------------------------------------------------

def _spec_builder(spec: dict, **extra):
    factory = spec["factory"]
    if extra:          # a variant of the champion: same nets, a different factory
        factory = extra.pop("factory")
    return partial(_build_agent, factory, **{**spec.get("kwargs", {}), **extra})


def _treatment_builder(config, lever: str, delta: float):
    return _spec_builder(config["champion"], factory="ml_biased", lever=lever, delta=delta)


def _progress(label: str, total: int):
    done = [0]
    step = max(1, total // 10)

    def tick(n):
        before = done[0] // step
        done[0] += n
        if done[0] // step > before or done[0] == total:
            log.info("%s: %d / %d deals", label, done[0], total)
    return tick


# --- calibrate --------------------------------------------------------------------

def calibrate(config, out: Path, *, force: bool = False) -> int:
    cal = config["calibration"]
    cal_dir = out / "calibration"
    deltas_path = cal_dir / "deltas.yaml"
    if deltas_path.exists() and not force:
        log.info("deltas already frozen at %s (use --force before `run` to redo)", deltas_path)
        return 0
    if force and (out / "arms").exists():
        raise SystemExit("refusing to re-calibrate: treatment arms have already been played")
    cal_dir.mkdir(parents=True, exist_ok=True)
    n = int(cal["n_deals"])
    positions = generate_full_position_pool(seed=int(cal["pool_seed"]), n=n)
    log.info("calibration: %d deals (pool seed %d), %d workers", n, cal["pool_seed"],
             config["workers"])
    gaps = collect_lever_gaps(_spec_builder(config["champion"], factory="ml_gaps"),
                              _spec_builder(config["opponent"]), positions,
                              workers=int(config["workers"]))
    pd.DataFrame([(lever, g) for lever, gs in gaps.items() for g in gs],
                 columns=["lever", "gap"]).to_parquet(cal_dir / "gaps.parquet", index=False)
    frozen = {}
    for lever in config["levers"]:
        d = choose_delta(gaps[lever], share=float(cal["share"]), cap=float(cal["cap"]))
        g = np.asarray(gaps[lever])
        frozen[lever] = {
            "plus": round(float(d["plus"]), 4), "minus": round(float(d["minus"]), 4),
            "predicted_plus_share": round(float(d["plus_share"]), 4),
            "predicted_minus_share": round(float(d["minus_share"]), 4),
            "situations": int(len(g)),
            "baseline_rate": round(float(np.mean(g > 0)), 4) if len(g) else None,
        }
        log.info("%s: %s", lever, frozen[lever])
    deltas_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    log.info("froze deltas at %s", deltas_path)
    return 0


def _deltas(out: Path) -> dict:
    path = out / "calibration" / "deltas.yaml"
    if not path.exists():
        raise SystemExit(f"no frozen deltas at {path}: run `calibrate` first")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _arms(config, out: Path):
    deltas = _deltas(out)
    for lever in config["levers"]:
        for sign in SIGNS:
            yield lever, sign, float(deltas[lever][sign])


# --- run --------------------------------------------------------------------------

def run(config, out: Path, *, force: bool = False) -> int:
    arms = list(_arms(config, out))
    n, workers = int(config["n_deals"]), int(config["workers"])
    if workers <= 1:
        raise SystemExit("every arm must run with workers > 1 (see the module docstring)")
    positions = generate_full_position_pool(seed=int(config.get("pool_seed", 0)), n=n)
    opponent = _spec_builder(config["opponent"])

    ref_dir = out / "reference"
    if (ref_dir / "rounds.parquet").exists():
        log.info("reference arm: cached at %s", ref_dir)
    else:
        ref = run_fixed_opponent_arm(_spec_builder(config["champion"]), opponent, positions,
                                     arm="reference", workers=workers,
                                     progress=_progress("reference", n))
        save_drift_log(ref, ref_dir)

    k = int(config.get("identity_check_deals", 200))
    ident = run_fixed_opponent_arm(_treatment_builder(config, config["levers"][0], 0.0),
                                   opponent, positions[:k], arm="reference", workers=workers)
    ref_rounds = load_drift_log(ref_dir).rounds
    expected = ref_rounds[ref_rounds.deal < k].reset_index(drop=True)
    got = ident.rounds.reset_index(drop=True)
    cols = ["deal", "half", "total_0", "total_1", "out_order", "tichu_callers", "grand_callers"]
    if not got[cols].astype(str).equals(expected[cols].astype(str)):
        raise SystemExit("identity check FAILED: δ = 0 does not reproduce the reference arm")
    log.info("identity check passed: δ = 0 reproduces the reference on %d deals", k)

    for lever, sign, delta in arms:
        arm_dir = out / "arms" / f"{lever}_{sign}"
        if (arm_dir / "rounds.parquet").exists():
            log.info("%s %s (δ=%+.3f): cached", lever, SIGNS[sign], delta)
            continue
        log.info("%s %s: δ = %+.3f over %d deals", lever, SIGNS[sign], delta, n)
        t = run_fixed_opponent_arm(_treatment_builder(config, lever, delta), opponent, positions,
                                   arm=f"{lever}_{sign}", workers=workers,
                                   progress=_progress(f"{lever} {SIGNS[sign]}", n))
        save_drift_log(t, arm_dir)
    return 0


# --- summarise --------------------------------------------------------------------

def summarise_step(config, out: Path, *, force: bool = False) -> int:
    n_boot, seed = int(config.get("n_boot", 10000)), int(config.get("pool_seed", 0))
    floor = int(config.get("floor", 200))
    ref = load_drift_log(out / "reference")
    rows = []
    for lever, sign, delta in _arms(config, out):
        arm_dir = out / "arms" / f"{lever}_{sign}"
        if not (arm_dir / "rounds.parquet").exists():
            log.warning("%s %s: not played yet, skipped", lever, SIGNS[sign])
            continue
        treat = load_drift_log(arm_dir)
        e = arm_effect(ref, treat, lever, n_boot=n_boot, seed=seed, floor=floor)
        rows.append({"lever": lever, "sign": SIGNS[sign], "delta": delta, **e})
        # The whole drift panel for this arm: what else the nudge moved.
        summarise(pair_arms(ref, treat), n_boot=min(n_boot, 2000), seed=seed,
                  floor=floor).to_csv(arm_dir / "panel.csv", index=False)
    table = pd.DataFrame(rows)
    table["holm"] = holm(table.p.to_numpy())
    table.to_csv(out / "results.csv", index=False)
    (out / "verdict.md").write_text(_verdict_md(table, n_deals=int(config["n_deals"])),
                                    encoding="utf-8")
    log.info("wrote %s and %s", out / "results.csv", out / "verdict.md")
    print(_verdict_md(table, n_deals=int(config["n_deals"])))
    return 0


def _verdict_md(table: pd.DataFrame, *, n_deals: int) -> str:
    v = verdicts(table)
    lines = [f"# Behavior Sensitivity Probe — {n_deals} deals per arm", "",
             "| lever | arm | δ | lever rate ref → treat | Δrate | ΔEV/Round [95% CI] | "
             "card play | slope (pts per rate-pt) | Holm |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(
            f"| {r.lever} | {r.sign} | {r.delta:+.2f} | {100 * r.rate_ref:.1f}% → "
            f"{100 * r.rate_treat:.1f}% | {100 * r.d_rate:+.1f} pt | {r.d_ev:+.2f} "
            f"[{r.d_ev_lo:+.2f}, {r.d_ev_hi:+.2f}] | {r.d_card_play:+.2f} | {r.slope:+.3f} | "
            f"{'yes' if r.holm else 'no'} |")
    lines += ["", "## Verdicts (pre-registered rules)", ""]
    lines += [f"- **{lever}**: {verdict}" for lever, verdict in v.levers.items()]
    if v.failed_manipulation:
        lines.append("- failed manipulation (|Δrate| < 5 pt): "
                     + ", ".join(f"{a} {b}" for a, b in v.failed_manipulation))
    lines.append(f"- global null: **{'yes' if v.global_null else 'no'}**")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
