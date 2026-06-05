r"""Value-ceiling diagnostic (ADR-0031 Phase-1 follow-up).

The search+learning run failed with value R^2 ~ 0 — the leaf value carried no signal,
so PIMC just re-expressed the passive prior. Before any more loop compute, answer the
prerequisite: *can any value model predict round_outcome from the v5 features well enough
to drive search?*

Generates master self-play `(features, z, round_id)` once (z = team-relative round_outcome
/100, the exact value-learning task), then sweeps:
  - a LINEAR baseline (closed-form) — the signal floor;
  - CAPACITY: ValueBaseline width in {128, 512, 2048};
  - DATA: 25 / 50 / 100 % of the training rounds at width 512;
reporting TRAIN and HELD-OUT R^2 under a **round-level** split (states from one round share
z + correlated features, so a per-sample split would leak and inflate test R^2).

Read:
  * test R^2 rises with data    -> more data is the lever.
  * test R^2 rises with width   -> the net was underpowered.
  * train R^2 high, test R^2 ~0 -> overfitting; more data/regularisation.
  * train R^2 ~0 even big+ample -> MC target is irreducibly noisy from these features
                                   -> need a TD/bootstrap target or richer features.

Usage (PowerShell):
  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\value_ceiling_sweep.py --rounds 1500
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_inference.ml_agent import MLAgent
from tichu_training.awr.value_baseline import ValueBaseline, fit_value_baseline
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, featurize

EXPORT = r"C:\workbench\tichu\data\export\bc_full_corpus_v5_memmap_unshuffled"
POOL = r"C:\workbench\tichu\data\full_position_pool_s0_n2000.parquet"


def _master() -> MLAgent:
    return MLAgent(EXPORT + r"\policy.pt", skill_decile=9,
                   schupfen_path=EXPORT + r"\schupfen.pt",
                   tichu_call_path=EXPORT + r"\tichu_call.pt",
                   grand_call_path=EXPORT + r"\grand_tichu_call.pt")


def collect(master, positions):
    """Master self-play (features, z, round_id) at every Play Decision."""
    feats, targets, rounds = [], [], []
    for rid, pos in enumerate(positions):
        rows = []
        master.last_fallback_used  # noqa: B018 - touch attr; observer below records

        def observer(seat, private_state, action, _rows=rows):
            _rows.append((seat, featurize(private_state)))

        result = play_full_round((master, master, master, master), pos.state,
                                 pos.grand_prefixes, observer=observer)
        total = result.total
        for seat, f in rows:
            t = seat % 2
            feats.append(f)
            targets.append((total[t] - total[1 - t]) / 100.0)
            rounds.append(rid)
    return (np.asarray(feats, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            np.asarray(rounds, dtype=np.int32))


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _linear_r2(xtr, ytr, xte, yte) -> tuple[float, float]:
    # Ridge via normal equations (small lambda for conditioning); the signal floor.
    x1 = np.concatenate([xtr, np.ones((len(xtr), 1), np.float32)], axis=1)
    lam = 1.0 * np.eye(x1.shape[1], dtype=np.float64)
    w = np.linalg.solve(x1.T @ x1 + lam, x1.T @ ytr)
    pred_tr = x1 @ w
    xte1 = np.concatenate([xte, np.ones((len(xte), 1), np.float32)], axis=1)
    return _r2(pred_tr, ytr), _r2(xte1 @ w, yte)


def _mlp_r2(xtr, ytr, xte, yte, *, hidden, epochs, lr) -> tuple[float, float]:
    model = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=hidden)
    fit_value_baseline(model, xtr, ytr, batch_size=4096, epochs=epochs, lr=lr,
                       show_progress=False)
    model.eval()
    with torch.no_grad():
        ptr = model(torch.from_numpy(xtr)).cpu().numpy()
        pte = model(torch.from_numpy(xte)).cpu().numpy()
    return _r2(ptr, ytr), _r2(pte, yte)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cache", default=r"C:\workbench\tichu\data\tmp\value_ceiling_data.npz")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    if cache.exists():
        d = np.load(cache)
        feats, z, rounds = d["feats"], d["z"], d["rounds"]
        print(f"loaded cached data: {len(z):,} samples from cache")
    else:
        print(f"generating master self-play: {args.rounds} rounds...")
        master = _master()
        positions = load_full_position_pool(pathlib.Path(POOL))[: args.rounds]
        t0 = time.perf_counter()
        feats, z, rounds = collect(master, positions)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, feats=feats, z=z, rounds=rounds)
        print(f"collected {len(z):,} samples in {time.perf_counter()-t0:.0f}s "
              f"(z mean {z.mean():+.3f} std {z.std():.3f}, Var {z.var():.3f})")

    # Round-level 80/20 split (no leakage across the shared-z, correlated states of a round).
    uniq = np.unique(rounds)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_test_rounds = max(1, len(uniq) // 5)
    test_rounds = set(uniq[:n_test_rounds].tolist())
    is_test = np.array([r in test_rounds for r in rounds])
    xtr, ytr = feats[~is_test], z[~is_test]
    xte, yte = feats[is_test], z[is_test]
    print(f"split: {len(ytr):,} train / {len(yte):,} test samples "
          f"({len(uniq)-n_test_rounds}/{n_test_rounds} rounds); "
          f"baseline Var(z_test)={yte.var():.3f}\n")

    print(f"{'model':<22}{'train R^2':>12}{'test R^2':>12}")
    print("-" * 46)

    tr, te = _linear_r2(xtr, ytr, xte, yte)
    print(f"{'linear (ridge)':<22}{tr:>12.3f}{te:>12.3f}")

    for hidden in (128, 512, 2048):
        tr, te = _mlp_r2(xtr, ytr, xte, yte, hidden=hidden, epochs=args.epochs, lr=args.lr)
        print(f"{'MLP hidden='+str(hidden):<22}{tr:>12.3f}{te:>12.3f}")

    print("-" * 46)
    # Data-scale sweep at width 512 (fraction of TRAIN rounds).
    tr_rounds = np.array(sorted(set(rounds[~is_test].tolist())))
    for frac in (0.25, 0.5, 1.0):
        k = max(1, int(len(tr_rounds) * frac))
        keep = set(tr_rounds[:k].tolist())
        m = np.array([r in keep for r in rounds[~is_test]])
        tr, te = _mlp_r2(xtr[m], ytr[m], xte, yte, hidden=512, epochs=args.epochs, lr=args.lr)
        print(f"{'data='+str(int(frac*100))+'% h=512':<22}{tr:>12.3f}{te:>12.3f}  "
              f"({m.sum():,} train)")


if __name__ == "__main__":
    main()
