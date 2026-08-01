"""Build a Correction Corpus from an existing tier-1 sweep (ADR-0042).

Exhaustively verifies every tier-1 Decision falling in the three Delta Bands and
emits **both** verdict classes — Verified Corrections and Verified
Non-Corrections — plus a drift set drawn from Decisions the policy actually chose
between. This is the corpus `Preference Correction` trains on.

Distinct from the two existing miners:
  * `mine_blunders`         — tier 1 + a top-N tail screen. Answers "show me the
                              worst mistakes"; no bands, no false-positive control.
  * `mine_blunders_stratified` — 150 per band + a Control Band. Answers "what is
                              the blunder RATE"; samples, so it is not a corpus.
  * this                    — exhaustive over the bands, keeps the failures, and
                              featurizes. Answers "what do we train on".

Tier 1 is the expensive half and is reused as-is, so this only runs verification.

Paths are ABSOLUTE in the example on purpose: `data/` lives in the main checkout
only, so relative paths resolve to nothing when this runs from a git worktree.

  py -m scripts.mine_corrections \
      --tier1 C:/workbench/tichu/data/runs/blunder_mining_iter27008/tier1_hindsight.parquet \
      --export-dir C:/workbench/tichu/data/runs/<run>/_ship_check_export \
      --out-dir C:/workbench/tichu/data/runs/correction_corpus_iter27008

IMPORTANT: `round_idx` indexes a REGENERATED Starting-Position Pool, so
`--pool-seed` and `--rounds` MUST match the tier-1 sweep or every Decision is
verified against the wrong deal. A `max(round_idx) >= rounds` mismatch is caught
below; a wrong SEED is silent — check it yourself.
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Resolve THIS tree's src ahead of the editable install, which points at the main
# checkout. Without it a worktree run pairs this `scripts/` with another tree's
# `src/` — a missing module if you are lucky, a silently stale `blunder_miner`
# (salted tier-2 seeds) if you are not. Derived from __file__, never hardcoded:
# the run-dir drivers hardcoded absolute worktree paths and rotted.
# `spawn` workers inherit sys.path through the preparation data, so this covers
# the pool as well as the parent.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _task(task):
    from tichu_training.correction.corpus import drift_rows, verify_and_featurize
    from tichu_training.search.blunder_miner import record_round

    from scripts.mine_blunders import _WORKER

    round_idx, position, candidates, worlds, drift_n, reverify = task
    agents = _WORKER["agents"]
    corpus = verify_and_featurize(agents, position, round_idx=round_idx,
                                  candidates=candidates, worlds=worlds,
                                  reverify_worlds=reverify)
    _, decisions = record_round(agents, position)
    # `len(candidates)` is returned separately because rows are NOT 1:1 with
    # candidates — a Decision whose legal set escapes the Action Space (e.g.
    # `Single(phoenix, as_rank=1.5)`) yields no row. Driving the bar off rows would
    # leave it short of 100% forever, and the drop would go unreported.
    return corpus, drift_rows(decisions, round_idx=round_idx, n=drift_n), len(candidates)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier1", required=True, help="tier1_hindsight.parquet to verify from")
    ap.add_argument("--export-dir", required=True, help="four-net TorchScript export dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pool-seed", type=int, default=777000,
                    help="MUST match the tier-1 sweep")
    ap.add_argument("--rounds", type=int, default=1500, help="MUST match the tier-1 sweep")
    ap.add_argument("--worlds", type=int, default=24, help="screening worlds")
    ap.add_argument("--reverify-worlds", type=int, default=96,
                    help="re-measure MOVABLE survivors (cross-world spread > 0) on "
                         "this many fresh worlds and relabel from that reading. "
                         "65%% of survivors are invariant and are skipped, so this "
                         "costs ~+8%% wall, not 4x. 0 disables.")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--drift-per-round", type=int, default=1,
                    help="non-forced Decisions sampled per Round for the drift set")
    ap.add_argument("--skill-decile", type=int, default=9)
    args = ap.parse_args(argv)

    from tichu_eval.full_position_pool import generate_full_position_pool

    from scripts.mine_blunders import _init_worker
    from tichu_training.correction.corpus import band_candidates, progress_stats

    df = pd.read_parquet(args.tier1)
    hi_idx = int(df["round_idx"].max())
    if hi_idx >= args.rounds:
        raise SystemExit(
            f"tier-1 references round_idx up to {hi_idx} but --rounds is {args.rounds}; "
            "the pool would not cover it — these are not the same sweep")

    cands = band_candidates(df)
    print(f"{len(df):,} tier-1 rows -> {len(cands):,} in-band decisions", flush=True)
    for band, n in cands["band"].value_counts().sort_index().items():
        print(f"  {band:12s} {n:>7,}", flush=True)

    positions = generate_full_position_pool(seed=args.pool_seed, n=args.rounds)
    by_round: dict[int, list[dict]] = {}
    for row in cands.to_dict("records"):
        by_round.setdefault(int(row["round_idx"]), []).append(row)
    reverify = args.reverify_worlds or None
    tasks = [(ri, positions[ri], rows, args.worlds, args.drift_per_round, reverify)
             for ri, rows in by_round.items()]
    print(f"verifying {len(cands):,} candidates across {len(tasks):,} rounds at "
          f"{args.worlds} worlds, {args.workers} workers", flush=True)
    if reverify:
        print(f"  movable survivors re-measured on {reverify} fresh worlds", flush=True)

    # Progress is measured in CANDIDATES, not rounds: a round carries anywhere from
    # one to dozens of them, so a rounds-based ETA misleads badly early on.
    ctx = mp.get_context("spawn")
    corpus: list[dict] = []
    drift: list[dict] = []
    done = 0
    tty = sys.stderr.isatty()
    bar = tqdm(total=len(cands), unit="cand", unit_scale=False, dynamic_ncols=True,
               desc="verify", smoothing=0.05, disable=not tty)
    try:
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(args.export_dir, args.skill_decile)) as pool:
            for rows, dr, attempted in pool.imap_unordered(_task, tasks, chunksize=2):
                corpus.extend(rows)
                drift.extend(dr)
                done += 1
                bar.update(attempted)
                st = progress_stats(corpus)
                if tty:
                    bar.set_postfix(corr=st["corrections"], rev=st["reverified"],
                                    regr=st["regressed"], refresh=False)
                elif done % 50 == 0:
                    # Detached runs redirect stdout to a log; a \r bar would render
                    # as one unreadable line, so emit periodic lines instead.
                    el = bar.format_dict["elapsed"]
                    frac = st["verified"] / max(len(cands), 1)
                    eta = el / frac - el if frac > 0 else 0.0
                    print(f"  {st['verified']:,}/{len(cands):,} cand ({frac:.1%}) "
                          f"rounds {done}/{len(tasks)} "
                          f"corr={st['corrections']:,} rev={st['reverified']:,} "
                          f"regr={st['regressed']:,} "
                          f"elapsed {el / 3600:.1f}h eta {eta / 3600:.1f}h", flush=True)
    finally:
        bar.close()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(corpus)
    frame.to_parquet(out / "corpus.parquet", index=False)
    pd.DataFrame(drift).to_parquet(out / "drift.parquet", index=False)

    st = progress_stats(corpus)
    n_pos, n_neg = st["corrections"], st["verified"] - st["corrections"]
    skipped = len(cands) - st["verified"]
    if skipped:
        # Never a silent cap: a Decision whose legal set escapes the Action Space
        # is absent from BOTH classes, so it is a coverage gap in the corpus, not
        # merely a smaller corpus.
        print(f"\nskipped {skipped:,}/{len(cands):,} candidates "
              f"({skipped / len(cands):.1%}) — legal set outside the Action Space "
              f"(e.g. Single(phoenix, as_rank=1.5)); absent from both classes",
              flush=True)
    if st["reverified"]:
        print(f"\nre-verified {st['reverified']:,} movable survivors at {reverify} "
              f"worlds: {st['regressed']:,} regressed to Verified Non-Correction",
              flush=True)
    print("\n=== corpus ===", flush=True)
    print(frame.groupby(["band", "is_correction"]).size().to_string(), flush=True)
    print(f"\n{n_pos:,} Verified Corrections / {n_neg:,} Verified Non-Corrections "
          f"({n_neg / max(n_pos, 1):.1f}:1); {len(drift):,} drift rows", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
