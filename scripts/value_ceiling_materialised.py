r"""Value-ceiling sweep on the REAL corpus (ADR-0031 Phase-1 follow-up).

The self-play sweep (`value_ceiling_sweep.py`) showed value R^2 still climbing at 68k
samples (0.23). But `materialised_full_v5/bc` already holds 1.25B play rows, each with the
224-dim v5 feature AND a `round_outcome` label, from real BSW games — ~15,000x more value
data, no self-play needed. This maps the R^2 ceiling at scale.

Filters to one `skill_decile` (default 9, the master's tier): skill is a policy input, NOT a
feature column, so mixing deciles injects irreducible target noise — conditioning removes it.

Split is per-row on a sparse random draw from 1.25B rows (expected rows-per-round in the
sample << 1), so same-round leakage is negligible — no round id is needed.

Usage (PowerShell):
  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\value_ceiling_materialised.py --sample 6000000 --max-rows 2000000
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

BUNDLE = r"C:\workbench\tichu\data\materialised_full_v5\bc"


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _mlp_r2(xtr, ytr, xte, yte, *, hidden, epochs, lr) -> tuple[float, float]:
    model = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=hidden)
    fit_value_baseline(model, xtr, ytr, batch_size=8192, epochs=epochs, lr=lr,
                       show_progress=False)
    model.eval()
    with torch.no_grad():
        ptr = model(torch.from_numpy(xtr)).cpu().numpy()
        pte = model(torch.from_numpy(xte)).cpu().numpy()
    return _r2(ptr, ytr), _r2(pte, yte)


def _reconstruct(ds, idx, chunk=200_000) -> np.ndarray:
    """Rebuild dense (len(idx), 224) features for sorted play-row indices, chunked."""
    bits_mm = ds._feat_bits["play"]
    cont_mm = ds._feat_cont["play"]
    out = np.empty((len(idx), FEATURIZER_OUTPUT_DIM), dtype=np.float32)
    for s in range(0, len(idx), chunk):
        sl = idx[s : s + chunk]
        out[s : s + len(sl)] = ds._reconstruct_features(bits_mm[sl], cont_mm[sl])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=6_000_000,
                    help="random play rows to scan for the decile filter")
    ap.add_argument("--max-rows", type=int, default=2_000_000,
                    help="cap on decile-filtered rows actually featurized")
    ap.add_argument("--decile", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    ds = MemmapBCDataset(BUNDLE)
    n_play = ds.counts["play"]
    meta = ds._meta["play"]
    print(f"bundle: {n_play:,} play rows; sampling {args.sample:,} to filter decile "
          f"{args.decile}...")

    rng = np.random.default_rng(0)
    cand = np.unique(rng.integers(0, n_play, size=args.sample))
    m = meta[cand]
    deciles = m["skill_decile"]
    # report the decile mix on the random sample
    vals, counts = np.unique(deciles, return_counts=True)
    print("decile histogram (sampled): " +
          " ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts)))

    keep = cand[deciles == args.decile]
    if len(keep) > args.max_rows:
        keep = rng.choice(keep, size=args.max_rows, replace=False)
    keep = np.sort(keep)
    # round_outcome is stored in RAW score units (std ~225); /100 matches the self-play
    # convention and the value-baseline target scale so a fixed-lr MSE fit is well-behaved.
    z = meta[keep]["round_outcome"].astype(np.float32) / 100.0
    print(f"decile {args.decile}: {len(keep):,} rows | z/100 mean {z.mean():+.3f} "
          f"std {z.std():.3f} Var {z.var():.3f}")

    t0 = time.perf_counter()
    feats = _reconstruct(ds, keep)
    print(f"reconstructed features {feats.shape} in {time.perf_counter()-t0:.0f}s\n")

    n = len(feats)
    perm = rng.permutation(n)
    n_te = n // 5
    te, tr = perm[:n_te], perm[n_te:]
    xtr, ytr, xte, yte = feats[tr], z[tr], feats[te], z[te]
    print(f"{len(ytr):,} train / {len(yte):,} test  (Var(z_test)={yte.var():.3f})\n")

    print(f"{'config':<26}{'train R^2':>12}{'test R^2':>12}")
    print("-" * 50)
    # data-scale sweep at hidden=512
    for frac in (0.1, 0.3, 1.0):
        k = max(1, int(len(xtr) * frac))
        tr_r2, te_r2 = _mlp_r2(xtr[:k], ytr[:k], xte, yte,
                               hidden=512, epochs=args.epochs, lr=args.lr)
        print(f"{'h=512 data='+str(int(frac*100))+'%':<26}{tr_r2:>12.3f}{te_r2:>12.3f}"
              f"  ({k:,})")
    # capacity at full data
    for hidden in (1024, 2048):
        tr_r2, te_r2 = _mlp_r2(xtr, ytr, xte, yte, hidden=hidden,
                               epochs=args.epochs, lr=args.lr)
        print(f"{'h='+str(hidden)+' data=100%':<26}{tr_r2:>12.3f}{te_r2:>12.3f}")


if __name__ == "__main__":
    main()
