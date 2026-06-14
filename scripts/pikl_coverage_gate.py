"""piKL coverage pre-gate (ADR-0037 E) — the cheap go/no-go before any agent.

piKL's support is the anchor's: it can only re-rank an Intent the BC head already
proposes in its top-k. So before building the agent, ask: of the 663 mined
corrections, what fraction have the BETTER Intent (`alt_idx`) inside the anchor's
top-k? Pre-registered: >=~60% in-support at the chosen k -> proceed; mostly
out-of-support -> piKL is blind to the located signal, stop.

Faithfulness check: under the served skill decile the agent played argmax, so the
agent's own `chosen_idx` should rank 0 on almost every row. If it does not, the
forward is not reproducing the mining-time anchor and the coverage numbers are
suspect.
"""

import sys

import numpy as np
import pandas as pd
import torch

from tichu_inference.ml_agent import load_policy_module
from tichu_training.search.pikl import intent_rank

CORRECTIONS = "C:/workbench/tichu/data/runs/blunder_mining_v1/corrections.parquet"
POLICY = "C:/workbench/tichu/data/runs/cotrain_wish_v5/export/iter_06225/policy.pt"
KS = (4, 8, 16, 32)


def main(decile: int = 9) -> None:
    df = pd.read_parquet(CORRECTIONS)
    feats = torch.from_numpy(
        np.stack([np.asarray(f, dtype=np.float32) for f in df["features"]])
    )
    skill = torch.full((len(df),), decile, dtype=torch.long)
    mod = load_policy_module(POLICY)
    with torch.no_grad():
        logits = mod(feats, skill)["play"].cpu().numpy()

    chosen_ranks, alt_ranks = [], []
    for i, row in df.reset_index(drop=True).iterrows():
        legal = list(row["legal_idx"])
        chosen_ranks.append(intent_rank(logits[i], legal, int(row["chosen_idx"])))
        alt_ranks.append(intent_rank(logits[i], legal, int(row["alt_idx"])))
    chosen_ranks = np.array(chosen_ranks)
    alt_ranks = np.array(alt_ranks)

    n = len(df)
    print(f"corrections: {n}   skill_decile: {decile}")
    print(f"FAITHFULNESS  chosen_idx rank==0: {(chosen_ranks == 0).mean():.1%} "
          f"(should be ~100% — the agent played argmax)")
    print("\nCOVERAGE  fraction of corrections whose BETTER Intent is in anchor top-k:")
    for k in KS:
        frac = (alt_ranks < k).mean()
        gate = "PASS" if frac >= 0.60 else "fail"
        print(f"  top-{k:<2}: {frac:6.1%}   [{gate} vs 60% bar]")
    pct = np.percentile(alt_ranks, [50, 75, 90, 95])
    print(f"\nalt-rank percentiles  p50={pct[0]:.0f} p75={pct[1]:.0f} "
          f"p90={pct[2]:.0f} p95={pct[3]:.0f}  max={alt_ranks.max()}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
