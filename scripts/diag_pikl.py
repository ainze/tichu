"""Diagnose the piKL lam=0.1 regression (-58/round) -- per-decision, no tournament.

H1 (asked_tichu phantom calls) came back 0% flips on call-live in the first pass
-> refuted (the call net declines on depleted mid-round hands, so no phantom calls
fire). This pass targets H4: at lam=0.1, piKL ~= argmax over a NOISY N=10 rollout Q,
so it overrides the BC pick on plays the rollout can't actually rank at N=10.

Per decision, with the faithful asked_tichu, build the [k x Nmax] return matrix
ONCE and read several things off it (cheap, nested sub-sampling):

  override rate : how often piKL's lam=0.1 pick != the BC argmax (top tau).
  Q-argmax stability : argmax(Q@10) vs argmax(Q@40) -- low = N=10 is noise.
  override quality : when piKL overrides BC, is its pick's Q@40 (the better
      estimate) higher or LOWER than BC's? Negative mean = the overrides are
      EV-negative even by the rollout's own less-noisy read -> the smoking gun.
  guard rate : piKL picks the partner-trick-guard would suppress.
"""

import random
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

EXPORT = "C:/workbench/tichu/data/runs/cotrain_wish_v5/export/iter_06225"
POOL = "C:/workbench/tichu/data/full_position_pool_s0_n20000.parquet"


def main(n_deals: int = 20, k: int = 5, nmax: int = 30, max_dec: int = 100,
         seed: int = 0) -> None:
    from tichu_engine.legality import legal_actions_for
    from tichu_training.search.heuristic_probes import suppress_partner_trick_bomb
    from tichu_eval.full_position_pool import load_full_position_pool
    from tichu_training.ppo.pmcpa import build_export_agent
    from tichu_training.search.blunder_miner import playout_from, record_round, team_relative
    from tichu_training.search.determinize import sample_determinized_world
    from tichu_training.search.pikl import anchored_softmax, standardize_q

    anchor = build_export_agent(export_dir=EXPORT, skill_decile=9)
    field = [anchor] * 4
    pool = load_full_position_pool(Path(POOL))[:n_deals]
    rng = random.Random(seed)

    LAMS = [0.1, 0.3, 1.0, 3.0]
    n = stable = guard = 0
    over = {lam: [] for lam in LAMS}   # per-lam: Q_full[pikl]-Q_full[bc] on overrides
    spreads = []
    for pos in pool:
        _, decisions = record_round(field, pos)
        for d in decisions:
            view = d.state.private_view(d.seat)
            if len(list(legal_actions_for(view))) <= 1:
                continue
            scored = anchor.play_action_scores(view)[:k]
            cands = [a for a, _ in scored]
            tau = np.array([p for _, p in scored], dtype=np.float64)
            if len(cands) <= 1:
                continue
            wr = random.Random(rng.randrange(2**31))
            worlds = [sample_determinized_world(view, None, wr) for _ in range(nmax)]
            # [k x nmax] team-relative returns, faithful asked_tichu. Built ONCE;
            # the lam-sweep below is pure analysis on it (zero extra playouts).
            R = np.array([[team_relative(
                playout_from(field, w, forced_action=c, asked_tichu=d.asked_tichu,
                             initial_scores=d.initial_scores), view.player)
                for w in worlds] for c in cands])
            q10, qfull = R[:, :10].mean(1), R.mean(1)   # piKL sees N=10; qfull = best est
            n += 1
            spreads.append(float(q10.std()))
            if int(np.argmax(q10)) == int(np.argmax(qfull)):
                stable += 1
            qstd = standardize_q(q10, scale=22.0)
            for lam in LAMS:
                pikl = int(np.argmax(anchored_softmax(tau, qstd, lam)))
                if pikl != 0:  # 0 = BC argmax (highest tau)
                    over[lam].append(float(qfull[pikl] - qfull[0]))
            if suppress_partner_trick_bomb(view, cands[int(np.argmax(
                    anchored_softmax(tau, qstd, 0.1)))], list(legal_actions_for(view))):
                guard += 1
            if n % 5 == 0:
                print(f"  [{n} dec] stable={stable} guard={guard}", flush=True)
            if n >= max_dec:
                break
        if n >= max_dec:
            break

    print("\n=== piKL diagnosis: lam sweep on cached rollouts ===")
    print(f"decisions: {n}   k={k}   piKL sees N=10, quality judged at N={nmax}")
    print(f"argmax-Q stable (N10==N{nmax}) : {stable}/{n} = {stable/max(n,1):.0%}  "
          f"(low => N=10 argmax is noise-dominated)")
    print(f"mean within-decision Q10 spread: {np.mean(spreads):.1f} pts (signal) vs "
          f"per-world SE at N=10 ~ {55/(10**0.5):.0f} pts (noise)")
    print(f"guard would-fire (lam=0.1 pick): {guard}/{n} = {guard/max(n,1):.0%}\n")
    print(f"{'lam':>5} | {'override rate':>13} | {'mean dQ_full on overrides':>26} | %neg")
    for lam in LAMS:
        dd = np.array(over[lam])
        rate = len(dd) / max(n, 1)
        if len(dd):
            print(f"{lam:>5} | {len(dd):>4}/{n} = {rate:>4.0%} | "
                  f"{dd.mean():>+10.1f} pts (n={len(dd):>2}) {'':>6} | {(dd<0).mean():>3.0%}")
        else:
            print(f"{lam:>5} | {0:>4}/{n} = {0:>4.0%} | {'(no overrides)':>26} |   -")
    print("\nRead: dQ_full<0 => piKL's overrides are EV-negative even by the better "
          f"N={nmax} estimate. If it stays <0 as lam grows (overrides shrink to the "
          "'confident' ones), the N=10 Q can't beat BC at any lam -> need lower-variance "
          "Q (more worlds or the PIC leaf-eval), not just a softer leash.")


if __name__ == "__main__":
    main()
