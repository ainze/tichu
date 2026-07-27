"""Does the seat-swap cluster bootstrap unjam the promotion gate?

The greedy gate records `2 * n_deals` margins per window — the two seat arrangements
of each deal, adjacent — then bootstraps them FLAT. Card luck enters the two
arrangements with opposite sign, so it cancels within the pair; resampling flat
re-counts uncertainty the seat swap already removed, and the CI comes out too wide.
A gate that only resolves large margins ratchets during the early easy gains and then
jams permanently (live run: last promotion iter 10368, 129 held windows since).

This replays ONE real gate window offline and reports the verdict both ways, so the
`promotion_gate.paired` flag can be flipped on evidence rather than on the argument.

Known from the 40k wishfix-vs-cpfix3328 ship check (a DISTANT cross-lineage pair):
rho = -0.42, shrink 0.735 at 4096 deals. The gate's own champion comparison is a
NEAR-TWIN pair (128 iterations apart, agreeing on most decisions), where the luck
cancels harder and rho should be more negative — that is what this measures.

Residual after pairing is dominated by CALL-bonus variance (sd 86 of 118 on the ship
check): each agent picks its own Tichu/Grand, so +/-100/200 does NOT cancel under the
swap. Hence the `--pool-windows` column: pairing narrows one window, pooling across
windows is what actually reaches the ~1.6 band.

  py -m scripts.gate_pairing_check --config configs/cotrain_v6_pbrs_resid_wish_gated_wishfix.yaml

RUN IT WHEN TRAINING IS STOPPED (or with few workers) — the persistent rollout pool
plus this tournament oversubscribes the box (OOM, ADR-0034).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.tournament import _bootstrap_ci, pair_cluster


def _fmt(mean, lo, hi, thr=0.0):
    verdict = "PROMOTE" if lo > thr else "hold   "
    return f"{verdict}  mean {mean:+7.3f}  CI [{lo:+7.2f}, {hi:+7.2f}]  +/-{(hi - lo) / 2:6.3f}"


def _report(name, margins, *, iters, seed, thr, pool_windows):
    margins = np.asarray(margins, dtype=float)
    t0, t1 = margins[0::2], margins[1::2]
    rho = float(np.corrcoef(t0, t1)[0, 1])
    pairs = pair_cluster(margins)
    exact = float(np.mean(pairs == 0.0))

    mf, lof, hif = _bootstrap_ci(margins, iters, np.random.default_rng(seed))
    mp, lop, hip = _bootstrap_ci(margins, iters, np.random.default_rng(seed), paired=True)

    print(f"\n=== {name} ===")
    print(f"  deals {pairs.size}   rho(arr1, arr2) {rho:+.4f}   "
          f"exactly-cancelled deals {100 * exact:.1f}%")
    print(f"  sd single {margins.std(ddof=1):7.2f}   sd pair-mean {pairs.std(ddof=1):7.2f}")
    print(f"  CURRENT (flat)  {_fmt(mf, lof, hif, thr)}")
    print(f"  PAIRED          {_fmt(mp, lop, hip, thr)}")
    hw_f, hw_p = (hif - lof) / 2, (hip - lop) / 2
    if hw_f > 0:
        print(f"  shrink {hw_p / hw_f:.3f}   "
              f"(equivalent to {100 * (1 - (hw_p / hw_f) ** 2):.0f}% fewer deals)")
    # What the SAME per-deal edge would resolve to if pooled over N windows -- the
    # lever that matters once pairing alone leaves the bar above the observed margin.
    for w in pool_windows:
        scaled = hw_p / np.sqrt(w)
        flag = "PROMOTE" if (mp - scaled) > thr else "hold"
        print(f"    pooled x{w:<3d} -> +/-{scaled:6.3f}  ci_lo {mp - scaled:+7.3f}  {flag}")
    return {"opponent": name, "rho": rho, "mean": mf,
            "hw_flat": hw_f, "hw_paired": hw_p, "n_deals": int(pairs.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--iter", type=int, default=None,
                    help="learner snapshot iteration (default: latest complete)")
    ap.add_argument("--deals", type=int, default=None,
                    help="deals per opponent (default: the config's greedy_n_deals)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--pool-windows", type=int, nargs="*", default=[5, 10, 20],
                    help="also report the CI this margin would reach pooled over N windows")
    ap.add_argument("--out", default=None, help="optional .npz dump of the raw margins")
    args = ap.parse_args(argv)

    from tichu_training.cli.check_cotrain import _latest_snapshot
    from tichu_training.cli.train_cotrain import _arch_cfg, _build_models
    from tichu_training.bc.training import load_checkpoint
    from tichu_training.ppo.greedy_gate import (
        export_live_models, export_opponent, greedy_pair_margins,
    )

    config = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(config["run_dir"])
    gate_cfg = config.get("promotion_gate", {})
    n_deals = args.deals or int(gate_cfg.get("greedy_n_deals", 2048))
    skill_decile = int(config.get("ppo", {}).get("skill_decile", 9))
    thr = float(gate_cfg.get("threshold", 0.0))

    it = args.iter if args.iter is not None else _latest_snapshot(run_dir / "snapshots")
    if it is None:
        print(f"no complete snapshot under {run_dir / 'snapshots'}", file=sys.stderr)
        return 1

    # The exact opponent set the live gate scores: rolling champion, frozen BC, and
    # any extra_opponents (including observe-only ones).
    opponents = [("champion", str(run_dir / "_champion.pt"))]
    bc_path = run_dir / "_bc_opponent.pt"
    if bool(gate_cfg.get("also_beat_bc", False)) and bc_path.exists():
        opponents.append(("bc", str(bc_path)))
    for spec in gate_cfg.get("extra_opponents", []) or []:
        if Path(spec["path"]).exists():
            opponents.append((spec["name"], spec["path"]))

    print(f"gate pairing check: learner snapshot iter {it}, {n_deals} deals/opponent, "
          f"{args.workers} workers")
    print(f"  opponents: {[n for n, _ in opponents]}")

    work = run_dir / "_pairing_check"
    models = _build_models(config)
    for net_key, model in models.items():
        load_checkpoint(str(run_dir / "snapshots" / f"iter_{it:05d}_{net_key}.bin"), model)
    learner_kwargs = export_live_models(models, work / "learner")

    # Fresh deals, seeded far from the rollout (pool_seed + iter*M), the gate eval
    # (1e9 + ...) and the ship check (1.8e8) streams, so nothing is replayed.
    positions = generate_full_position_pool(seed=2_000_000_000, n=n_deals)

    rows, dump = [], {}
    arch = _arch_cfg(config)
    for name, path in opponents:
        print(f"  playing vs {name} ...", flush=True)
        opp_kwargs = export_opponent(path, arch, work / f"opp_{name}")
        margins = greedy_pair_margins(
            learner_kwargs, opp_kwargs, positions,
            skill_decile=skill_decile, workers=args.workers,
        )
        dump[name] = margins
        rows.append(_report(name, margins, iters=args.bootstrap_iters, seed=0,
                            thr=thr, pool_windows=args.pool_windows))

    required = [r for r in rows
                if r["opponent"] not in {s["name"] for s in gate_cfg.get("extra_opponents", []) or []
                                         if not s.get("require", True)}]
    print("\n=== summary (required opponents gate the promotion) ===")
    for r in rows:
        req = "required" if r in required else "observe "
        print(f"  {r['opponent']:12s} {req}  rho {r['rho']:+.3f}  "
              f"+/-{r['hw_flat']:6.3f} -> +/-{r['hw_paired']:6.3f}")

    if args.out:
        np.savez(args.out, **dump)
        print(f"\nraw margins -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
