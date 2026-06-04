r"""Fit the Phase-2 search critic (ADR-0030 Phase B).

Generates master-vs-master self-play, records (featurize(view), round_outcome) at
every Play Decision (the same state distribution PIMC evaluates at its leaves), and
fits a `ValueBaseline` on the team-relative round_outcome (normalized /100, call
bonus included). Persists via `save_critic` for `SearchAgent(critic_path=...)`.

Usage (PowerShell):
  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\fit_search_critic.py --positions 600 --epochs 8 ^
      --out C:\workbench\tichu\data\export\search_critic_master_selfplay_v5\critic.bin
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tichu_eval.full_position_pool import load_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_inference.ml_agent import MLAgent
from tichu_training.awr.value_baseline import ValueBaseline, fit_value_baseline
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, featurize
from tichu_training.search.critic import save_critic

EXPORT = r"C:\workbench\tichu\data\export\bc_full_corpus_v5_memmap_unshuffled"
POOL = r"C:\workbench\tichu\data\full_position_pool_s0_n2000.parquet"


def _master() -> MLAgent:
    return MLAgent(
        checkpoint_path=EXPORT + r"\policy.pt",
        schupfen_path=EXPORT + r"\schupfen.pt",
        tichu_call_path=EXPORT + r"\tichu_call.pt",
        grand_call_path=EXPORT + r"\grand_tichu_call.pt",
        skill_decile=9,
    )


def collect(master: MLAgent, positions) -> tuple[np.ndarray, np.ndarray]:
    feats: list[np.ndarray] = []
    targets: list[float] = []
    for pos in positions:
        round_rows: list[tuple[int, np.ndarray]] = []

        def observer(seat, private_state, action, _rows=round_rows):
            _rows.append((seat, featurize(private_state)))

        result = play_full_round(
            (master, master, master, master),
            pos.state,
            pos.grand_prefixes,
            observer=observer,
        )
        total = result.total  # (team0_delta, team1_delta), call bonus included
        for seat, f in round_rows:
            t = seat % 2
            feats.append(f)
            targets.append((total[t] - total[1 - t]) / 100.0)
    return np.asarray(feats, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"loading master + {args.positions} positions...")
    master = _master()
    positions = load_full_position_pool(pathlib.Path(POOL))[: args.positions]

    t0 = time.perf_counter()
    feats, targets = collect(master, positions)
    print(f"collected {len(targets):,} samples from {len(positions)} rounds "
          f"in {time.perf_counter() - t0:.0f}s  (dim={feats.shape[1]}, "
          f"target mean={targets.mean():+.3f} std={targets.std():.3f})")

    baseline = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=args.hidden)
    mse = fit_value_baseline(
        baseline, feats, targets,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        show_progress=False,
    )
    print(f"fit done: final MSE={mse:.4f}")

    save_critic(
        baseline, args.out,
        feature_dim=FEATURIZER_OUTPUT_DIM, hidden=args.hidden,
        note=f"master self-play round_outcome/100, {len(targets)} samples, mse={mse:.4f}",
    )
    print(f"saved critic -> {args.out}")


if __name__ == "__main__":
    main()
