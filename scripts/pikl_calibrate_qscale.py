r"""Calibrate piKL's `q_scale` (ADR-0037 D) — the fixed point-scale that
`standardize_q` divides the raw paired-rollout advantages by, so `Q/λ` is
dimensionless and the paper's λ grid {0.03, 0.1, 0.3, 1.0} transfers.

The natural scale is the typical *spread* of the candidates' raw Q at a decision.
We play real rounds with the frozen anchor τ on all four seats, and at each
multi-legal Play Decision compute the top-k candidates' paired Q over N shared
worlds and record `std(Q)`. The median per-decision std is the recommended
`q_scale`.

  py -m scripts.pikl_calibrate_qscale
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


def main(n_deals: int = 25, k: int = 8, worlds: int = 10,
         max_decisions: int = 60, seed: int = 0) -> None:
    from tichu_eval.full_position_pool import load_full_position_pool
    from tichu_engine.legality import legal_actions_for
    from tichu_training.ppo.pmcpa import build_export_agent
    from tichu_training.search.blunder_miner import record_round
    from tichu_training.search.determinize import sample_determinized_world
    from tichu_training.search.pikl import pikl_q

    anchor = build_export_agent(export_dir=EXPORT, skill_decile=9)
    field = [anchor] * 4
    pool = load_full_position_pool(Path(POOL))[:n_deals]
    rng = random.Random(seed)

    stds = []
    for pos in pool:
        _, decisions = record_round(field, pos)
        for d in decisions:
            view = d.state.private_view(d.seat)
            if len(list(legal_actions_for(view))) <= 1:
                continue
            cands = [a for a, _ in anchor.play_action_scores(view)[:k]]
            if len(cands) <= 1:
                continue
            wr = random.Random(rng.randrange(2**31))
            ws = [sample_determinized_world(view, None, wr) for _ in range(worlds)]
            stds.append(float(np.std(pikl_q(field, view, cands, ws))))
            if len(stds) >= max_decisions:
                break
        if len(stds) >= max_decisions:
            break

    a = np.array(stds)
    print(f"decisions sampled: {len(a)}   k={k} worlds={worlds}")
    print(f"per-decision std(Q_raw)  median={np.median(a):.1f}  mean={a.mean():.1f}  "
          f"p25={np.percentile(a, 25):.1f}  p75={np.percentile(a, 75):.1f}  "
          f"max={a.max():.1f}")
    print(f"\nrecommended q_scale ~= {np.median(a):.0f}  (median per-decision spread)")


if __name__ == "__main__":
    main()
