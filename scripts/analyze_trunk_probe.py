"""Trunk-capacity probe analysis (pre-registered, 2026-07-03).

Reads step.csv from the baseline BC run and each probe arm, compares
trailing-window play-head NLL (loss_play), and applies the pre-committed bar:

  1. trailing-50k-batch mean loss_play >= 1% relative improvement, AND
  2. the gap holds in EVERY per-10k-step bucket over the final 200k steps.

All runs share seed/shuffle so rows at the same step index saw the same
examples — deltas are paired. See
docs/notes/2026-07-03-trunk-capacity-probe-v6-preregistration.md

Usage:
  python scripts/analyze_trunk_probe.py [--runs-root C:\\workbench\\tichu\\data\\runs]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "bc_full_corpus_v6_wishfix_memmap"
ARMS = [
    "bc_full_corpus_v6_wishfix_trunk_1024x8_memmap",
    "bc_full_corpus_v6_wishfix_trunk_2048x4_memmap",
    "bc_full_corpus_v6_wishfix_trunk_funnel_memmap",
    "bc_full_corpus_v6_wishfix_trunk_2048x8_memmap",
]
TRAIL_WINDOW = 50_000       # play batches in the primary trailing mean
BUCKET = 10_000             # bucket width for the curve-holds check
CURVE_SPAN = 200_000        # final span covered by the bucket check
REL_BAR = 0.01              # pre-registered: >= 1% relative NLL improvement


def load_play_loss(run_dir: Path) -> pd.Series:
    """loss_play indexed by step, play-head rows only, step-aligned."""
    df = pd.read_csv(run_dir / "step.csv", usecols=["step", "loss_play"])
    s = df.dropna(subset=["loss_play"]).set_index("step")["loss_play"]
    if s.empty:
        raise SystemExit(f"{run_dir}: no loss_play rows in step.csv")
    return s


def trailing_mean(s: pd.Series, n: int) -> float:
    return float(s.iloc[-n:].mean())


def bucket_means(s: pd.Series, span: int, bucket: int) -> pd.Series:
    tail = s.iloc[-span:]
    groups = np.arange(len(tail)) // bucket
    return tail.groupby(groups).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path(r"C:\workbench\tichu\data\runs"))
    args = ap.parse_args()

    base = load_play_loss(args.runs_root / BASELINE)
    base_trail = trailing_mean(base, TRAIL_WINDOW)
    base_buckets = bucket_means(base, CURVE_SPAN, BUCKET)
    print(f"baseline {BASELINE}")
    print(f"  rows={len(base):,}  trailing-{TRAIL_WINDOW // 1000}k "
          f"loss_play={base_trail:.5f}\n")

    any_clear = False
    for arm in ARMS:
        run_dir = args.runs_root / arm
        if not (run_dir / "step.csv").exists():
            print(f"{arm}: step.csv not found — skipped\n")
            continue
        s = load_play_loss(run_dir)
        # Paired comparison only over steps both runs completed.
        n = min(len(s), len(base))
        trail = trailing_mean(s.iloc[:n], TRAIL_WINDOW)
        b_trail = trailing_mean(base.iloc[:n], TRAIL_WINDOW)
        rel = (b_trail - trail) / b_trail
        buckets = bucket_means(s.iloc[:n], CURVE_SPAN, BUCKET)
        b_buckets = bucket_means(base.iloc[:n], CURVE_SPAN, BUCKET)
        k = min(len(buckets), len(b_buckets))
        gaps = (b_buckets.values[:k] - buckets.values[:k]) / b_buckets.values[:k]
        holds = bool((gaps > 0).all())
        clears = rel >= REL_BAR and holds
        any_clear |= clears

        print(f"{arm}")
        print(f"  rows={len(s):,} (paired over {n:,})")
        print(f"  trailing loss_play={trail:.5f}  rel-improvement={rel * 100:+.3f}%"
              f"  (bar {REL_BAR * 100:.0f}%)")
        print(f"  per-{BUCKET // 1000}k-bucket gaps over final "
              f"{CURVE_SPAN // 1000}k: min={gaps.min() * 100:+.3f}% "
              f"max={gaps.max() * 100:+.3f}%  all-positive={holds}")
        print(f"  --> {'CLEARS the pre-registered bar' if clears else 'does NOT clear'}\n")

    if not any_clear:
        print("VERDICT: no arm clears — per the pre-registration closing rule, "
              "the trunk-capacity question closes PERMANENTLY.")
    else:
        print("VERDICT: at least one arm clears — tie-break on lowest inference "
              "FLOPs (funnel < depth ~= width); run Move Prediction secondary "
              "before adopting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
