"""Blunder-mining runner: tier-1 hindsight sweep -> tier-2 robustness filter -> clusters.

Self-plays pool rounds with the SHIPPED agent (deterministic MLAgent over an export
dir, partner-trick guard on), then runs the two-tier counterfactual-replay funnel
from `tichu_training.search.blunder_miner`. Output: parquet of tier-1 rows, parquet
of tier-2-verified blunders, and a ranked cluster report (hypotheses for the
forced-action probe harness — NOT ship verdicts).

  py -m scripts.mine_blunders --export-dir <run>/export/iter_06225 \
      --rounds 1000 --workers 10 --out-dir data/runs/blunder_mining_v1

Tier-2 keeps a candidate only if the alternative beats the chosen action in at
least --tier2-win of --tier2-worlds determinized worlds AND the mean delta clears
--tier2-delta — "the agent should have known better without seeing hidden cards".
"""

import argparse
import multiprocessing as mp
import random
import sys
from pathlib import Path

import pandas as pd

_WORKER: dict = {}


def _init_worker(export_dir: str, skill_decile: int) -> None:
    import torch

    torch.set_num_threads(1)
    from tichu_inference.ml_agent import MLAgent

    export = Path(export_dir)
    agent = MLAgent(
        export / "policy.pt",
        skill_decile=skill_decile,
        schupfen_path=export / "schupfen.pt",
        tichu_call_path=export / "tichu_call.pt",
        grand_call_path=export / "grand_tichu_call.pt",
    )
    _WORKER["agents"] = [agent] * 4  # stateless -> one instance serves all seats


def _tier1_task(task):
    from tichu_training.search.blunder_miner import mine_round

    round_idx, position, top_k = task
    return mine_round(_WORKER["agents"], position, round_idx=round_idx, top_k=top_k)


def _tier2_task(task):
    """Replay one round, then verify each of its candidates across worlds."""
    from tichu_engine.legality import legal_actions_for
    from tichu_training.search.blunder_miner import (
        candidate_seed,
        record_round,
        verify_candidate,
    )

    round_idx, position, candidates, worlds = task
    _, decisions = record_round(_WORKER["agents"], position)
    by_turn = {d.turn: d for d in decisions}
    out = []
    for cand in candidates:
        d = by_turn.get(cand["turn"])
        if d is None or repr(d.chosen) != cand["chosen"]:
            continue  # replay drifted (shouldn't happen with deterministic agents)
        alt = next(
            (a for a in legal_actions_for(d.state.private_view(d.seat))
             if repr(a) == cand["alt"]), None,
        )
        if alt is None:
            continue
        rng = random.Random(candidate_seed(round_idx, cand["turn"], cand["alt"]))
        verdict = verify_candidate(_WORKER["agents"], d, alt, worlds=worlds, rng=rng)
        out.append({**cand, **verdict})
    return out


def select_tier2_candidates(df: pd.DataFrame, *, min_delta: float, top: int) -> pd.DataFrame:
    """Tier-1 rows worth the tier-2 spend: hindsight delta clears `min_delta`,
    best alternative per decision only, capped at `top` by delta."""
    hits = df[df["delta"] >= min_delta]
    best = hits.sort_values("delta", ascending=False).drop_duplicates(
        subset=["round_idx", "turn"], keep="first"
    )
    return best.head(top)


def cluster_report(verified: pd.DataFrame) -> pd.DataFrame:
    """Group surviving blunders into hypothesis clusters, ranked by frequency
    then mean robust delta."""
    keys = ["role", "phase", "caller_ctx", "partner_leads", "chosen_kind", "alt_kind"]
    grouped = (
        verified.groupby(keys)
        .agg(n=("mean_delta", "size"), mean_delta=("mean_delta", "mean"),
             mean_win=("alt_win_rate", "mean"))
        .reset_index()
        .sort_values(["n", "mean_delta"], ascending=False)
    )
    return grouped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Two-tier counterfactual blunder miner")
    parser.add_argument("--export-dir", required=True,
                        help="Export dir with policy/schupfen/tichu_call/grand_tichu_call .pt")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--pool-seed", type=int, default=777000,
                        help="Position pool seed (disjoint from the eval pool).")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--skill-decile", type=int, default=9)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tier2-top", type=int, default=200)
    parser.add_argument("--tier2-worlds", type=int, default=20)
    parser.add_argument("--tier2-win", type=float, default=0.7)
    parser.add_argument("--tier2-delta", type=float, default=15.0,
                        help="Tier-1 hindsight delta needed to enter tier 2.")
    parser.add_argument("--tier1-only", action="store_true")
    args = parser.parse_args(argv)

    from tichu_eval.full_position_pool import generate_full_position_pool

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = generate_full_position_pool(seed=args.pool_seed, n=args.rounds)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.export_dir, args.skill_decile)) as pool:
        # ---- Tier 1: hindsight sweep -------------------------------------
        tasks = [(i, pos, args.top_k) for i, pos in enumerate(positions)]
        rows: list[dict] = []
        done = 0
        for chunk in pool.imap_unordered(_tier1_task, tasks, chunksize=4):
            rows.extend(chunk)
            done += 1
            if done % 50 == 0:
                print(f"tier1: {done}/{len(tasks)} rounds, {len(rows)} rows", flush=True)
        df = pd.DataFrame(rows)
        t1_path = out_dir / "tier1_hindsight.parquet"
        df.to_parquet(t1_path, index=False)
        print(f"tier1 DONE: {len(df)} rows over {len(tasks)} rounds -> {t1_path}", flush=True)
        if len(df):
            print(f"  hindsight delta>0: {(df['delta'] > 0).mean():.1%}  "
                  f">=+{args.tier2_delta:g}: {(df['delta'] >= args.tier2_delta).mean():.2%}  "
                  f"max: {df['delta'].max():+.0f}", flush=True)
        if args.tier1_only or not len(df):
            return 0

        # ---- Tier 2: robustness across determinized worlds ---------------
        cands = select_tier2_candidates(df, min_delta=args.tier2_delta, top=args.tier2_top)
        print(f"tier2: verifying {len(cands)} candidates x {args.tier2_worlds} worlds",
              flush=True)
        by_round: dict[int, list[dict]] = {}
        for row in cands.to_dict("records"):
            by_round.setdefault(int(row["round_idx"]), []).append(row)
        t2_tasks = [
            (ri, positions[ri], cand_rows, args.tier2_worlds)
            for ri, cand_rows in by_round.items()
        ]
        verified_rows: list[dict] = []
        for chunk in pool.imap_unordered(_tier2_task, t2_tasks, chunksize=1):
            verified_rows.extend(chunk)
            print(f"tier2: {len(verified_rows)} candidates scored", flush=True)

    verified = pd.DataFrame(verified_rows)
    t2_path = out_dir / "tier2_verified.parquet"
    if len(verified):
        verified.to_parquet(t2_path, index=False)
    survivors = verified[
        (verified["alt_win_rate"] >= args.tier2_win)
        & (verified["mean_delta"] >= args.tier2_delta)
    ] if len(verified) else verified
    print(f"tier2 DONE: {len(verified)} scored, {len(survivors)} robust blunders "
          f"(win>={args.tier2_win:.0%}, mean_delta>=+{args.tier2_delta:g}) -> {t2_path}",
          flush=True)

    if len(survivors):
        survivors.to_parquet(out_dir / "blunders.parquet", index=False)
        report = cluster_report(survivors)
        report.to_csv(out_dir / "clusters.csv", index=False)
        print("\n=== ROBUST BLUNDER CLUSTERS (hypotheses for the probe harness) ===",
              flush=True)
        print(report.head(20).to_string(index=False), flush=True)
    else:
        print("no robust blunders survived tier 2", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
