"""Stratified tier-2 verification over an existing tier-1 hindsight sweep.

`mine_blunders`' built-in tier 2 takes the top `--tier2-top` candidates by hindsight
delta. That screens the extreme TAIL only — on the iter-27008 run all 200 came from
the [400,inf) band (delta +610..+1010) and the [15,50), [50,150), [150,400) bands
were never touched. Perfectly reasonable for "show me the worst mistakes", useless
for "what is the blunder RATE", and not comparable to a stratified run: the two
screen disjoint parts of the distribution.

This is the v1 protocol instead — a fixed sample from each delta band plus an
EV-neutral |delta| <= 5 CONTROL, so survival can be extrapolated back to the band
populations AND the false-positive rate is measured rather than assumed. (The cpfix
baseline scored 0/150 on its control; without that number a handful of tier-2 hits
means nothing.)

Generalised from the one-off `driver_strat_verify.py` left in the cpfix run dir,
which hardcoded its export path, output dir and a since-deleted worktree on sys.path.

Tier 1 is the expensive half and is reused as-is, so this only re-runs verification.

  py -m scripts.mine_blunders_stratified \
      --tier1 data/runs/blunder_mining_iter27008/tier1_hindsight.parquet \
      --export-dir data/runs/<run>/_ship_check_export \
      --out data/runs/blunder_mining_iter27008/tier2_stratified.parquet

IMPORTANT: `round_idx` indexes a REGENERATED position pool, so `--pool-seed` and
`--rounds` must match the tier-1 sweep that produced the parquet, or every verified
decision is replayed against the wrong deal. `mine_blunders` defaults are 777000 /
1000; the runs here used 777000 / 1500. A max(round_idx) >= rounds mismatch is caught
below, but a wrong SEED is silent — check it yourself.
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd

_BANDS = ((15, 50), (50, 150), (150, 400))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier1", required=True, help="tier1_hindsight.parquet to verify from")
    ap.add_argument("--export-dir", required=True, help="four-net TorchScript export dir")
    ap.add_argument("--out", required=True, help="destination .parquet for the verified rows")
    ap.add_argument("--pool-seed", type=int, default=777000,
                    help="MUST match the tier-1 sweep (mine_blunders default 777000)")
    ap.add_argument("--rounds", type=int, default=1500,
                    help="MUST match the tier-1 sweep")
    ap.add_argument("--per-band", type=int, default=150)
    ap.add_argument("--control-n", type=int, default=150,
                    help="|delta| <= 5 false-positive control; 0 disables (do not)")
    ap.add_argument("--worlds", type=int, default=24)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--win", type=float, default=0.7, help="robust iff alt_win_rate >= this")
    ap.add_argument("--delta", type=float, default=15.0, help="robust iff mean_delta >= this")
    ap.add_argument("--skill-decile", type=int, default=9)
    args = ap.parse_args(argv)

    from scripts.mine_blunders import _init_worker, _tier2_task, cluster_report
    from tichu_eval.full_position_pool import generate_full_position_pool

    df = pd.read_parquet(args.tier1)
    hi_idx = int(df["round_idx"].max())
    if hi_idx >= args.rounds:
        raise SystemExit(
            f"tier-1 references round_idx up to {hi_idx} but --rounds is {args.rounds}; "
            "the pool would not cover it — these are not the same sweep")

    # One row per DECISION (tier 1 emits a row per alternative), keeping the best
    # alternative, so band membership is a property of the decision not the branch.
    best = df.sort_values("delta", ascending=False).drop_duplicates(
        subset=["round_idx", "turn"], keep="first")

    samples, plan = [], []
    for i, (lo, hi) in enumerate(_BANDS):
        band = best[(best["delta"] >= lo) & (best["delta"] < hi)]
        take = band.sample(n=min(args.per_band, len(band)), random_state=100 + i).copy()
        take["band"] = f"[{lo},{hi})"
        plan.append((f"[{lo},{hi})", len(band), len(take)))
        samples.append(take)
    if args.control_n:
        ctrl = best[best["delta"].abs() <= 5]
        take = ctrl.sample(n=min(args.control_n, len(ctrl)), random_state=99).copy()
        take["band"] = "control|d|<=5"
        plan.append(("control|d|<=5", len(ctrl), len(take)))
        samples.append(take)

    for name, pop, n in plan:
        print(f"band {name:14s} {pop:>7,} decisions, sampling {n}", flush=True)
    cands = pd.concat(samples, ignore_index=True)

    positions = generate_full_position_pool(seed=args.pool_seed, n=args.rounds)
    by_round: dict[int, list[dict]] = {}
    for row in cands.to_dict("records"):
        by_round.setdefault(int(row["round_idx"]), []).append(row)
    tasks = [(ri, positions[ri], rows, args.worlds) for ri, rows in by_round.items()]
    print(f"verifying {len(cands)} candidates across {len(tasks)} rounds "
          f"at {args.worlds} worlds, {args.workers} workers", flush=True)

    ctx = mp.get_context("spawn")
    verified: list[dict] = []
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.export_dir, args.skill_decile)) as pool:
        for chunk in pool.imap_unordered(_tier2_task, tasks, chunksize=1):
            verified.extend(chunk)
            if len(verified) % 50 < len(chunk):
                print(f"  scored {len(verified)}/{len(cands)}", flush=True)

    v = pd.DataFrame(verified)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    v.to_parquet(out, index=False)

    v["robust"] = (v["alt_win_rate"] >= args.win) & (v["mean_delta"] >= args.delta)
    print(f"\n=== survival by band (win>={args.win}, mean_delta>={args.delta}) ===", flush=True)
    surv_tbl = v.groupby("band").agg(n=("robust", "size"), robust=("robust", "sum"),
                                     mean_win=("alt_win_rate", "mean"))
    print(surv_tbl.to_string(), flush=True)

    # Extrapolate each band's survival back to its population -> a per-round rate
    # that IS comparable across runs (the control is excluded; it is the FP check).
    pops = {name: pop for name, pop, _ in plan}
    total = 0.0
    print("\n=== stratified rate ===", flush=True)
    for name, row in surv_tbl.iterrows():
        if str(name).startswith("control"):
            fp = row["robust"] / row["n"] if row["n"] else 0.0
            print(f"  {name:14s} FALSE-POSITIVE control: {int(row['robust'])}/{int(row['n'])}"
                  f" = {fp:.2%}", flush=True)
            continue
        rate = row["robust"] / row["n"] if row["n"] else 0.0
        est = rate * pops.get(str(name), 0)
        total += est
        print(f"  {name:14s} survival {rate:6.2%} x {pops.get(str(name), 0):>7,} "
              f"= {est:8.1f} robust", flush=True)
    print(f"  TOTAL {total:,.0f} robust blunders over {args.rounds} rounds "
          f"= {total / args.rounds:.3f} per round", flush=True)

    hits = v[v["robust"]]
    if len(hits):
        print("\n=== robust clusters ===", flush=True)
        print(cluster_report(hits).head(15).to_string(index=False), flush=True)
    print(f"\nrows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
