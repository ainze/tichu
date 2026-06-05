r"""Pre-train the search leaf value on the real corpus (ADR-0031 fix).

The value-ceiling sweeps showed `materialised_full_v5/bc` (1.25B play rows, each with the
224-dim v5 feature + a `round_outcome` label) yields a value at **test R^2 ≈ 0.44** — vs the
≈0 the starved loop produced. This trains a `ValueBaseline(224, hidden=512)` on the decile-9
slice and persists it via `save_critic`, so `SearchAgent(critic_path=...)` gets a real-signal
leaf without any self-play. It is the critic for both the cheap frozen-search passivity dial
and (if that passes) the decoupled-value loop re-run.

Usage (PowerShell):
  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\pretrain_value_materialised.py --sample 18000000 --max-rows 2500000 ^
      --epochs 20 --out C:\workbench\tichu\data\export\search_critic_materialised_d9_v5\critic.bin
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tichu_training.awr.value_baseline import ValueBaseline, fit_value_baseline
from tichu_training.bc.materialised import MemmapBCDataset
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.search.critic import save_critic

BUNDLE = r"C:\workbench\tichu\data\materialised_full_v5\bc"


def _r2(pred, y) -> float:
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _reconstruct(ds, idx, chunk=200_000) -> np.ndarray:
    bits, cont = ds._feat_bits["play"], ds._feat_cont["play"]
    out = np.empty((len(idx), FEATURIZER_OUTPUT_DIM), dtype=np.float32)
    for s in range(0, len(idx), chunk):
        sl = idx[s : s + chunk]
        out[s : s + len(sl)] = ds._reconstruct_features(bits[sl], cont[sl])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=18_000_000)
    ap.add_argument("--max-rows", type=int, default=2_500_000)
    ap.add_argument("--decile", type=int, default=9)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = MemmapBCDataset(BUNDLE)
    n_play = ds.counts["play"]
    meta = ds._meta["play"]
    print(f"sampling {args.sample:,}/{n_play:,} play rows for decile {args.decile}...")
    rng = np.random.default_rng(0)
    cand = np.unique(rng.integers(0, n_play, size=args.sample))
    keep = cand[meta[cand]["skill_decile"] == args.decile]
    if len(keep) > args.max_rows:
        keep = rng.choice(keep, size=args.max_rows, replace=False)
    keep = np.sort(keep)
    z = meta[keep]["round_outcome"].astype(np.float32) / 100.0  # raw score -> value scale
    print(f"decile {args.decile}: {len(keep):,} rows | z mean {z.mean():+.3f} std {z.std():.3f}")

    t0 = time.perf_counter()
    feats = _reconstruct(ds, keep)
    print(f"reconstructed {feats.shape} in {time.perf_counter()-t0:.0f}s")

    # Small held-out slice for an honest R^2 readout (sparse sample -> no round leakage).
    n = len(feats)
    perm = rng.permutation(n)
    n_te = max(1, n // 20)
    te, tr = perm[:n_te], perm[n_te:]

    baseline = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=args.hidden)
    mse = fit_value_baseline(baseline, feats[tr], z[tr], batch_size=8192,
                             epochs=args.epochs, lr=args.lr, show_progress=False)
    baseline.eval()
    with torch.no_grad():
        pte = baseline(torch.from_numpy(feats[te])).cpu().numpy()
    r2 = _r2(pte, z[te])
    print(f"fit done: train mse={mse:.4f} | held-out R^2={r2:.3f}")

    save_critic(baseline, args.out, feature_dim=FEATURIZER_OUTPUT_DIM, hidden=args.hidden,
                note=f"materialised decile-{args.decile} round_outcome/100, "
                     f"{len(tr)} train rows, held-out R2={r2:.3f}")
    print(f"saved critic -> {args.out}")


if __name__ == "__main__":
    main()
