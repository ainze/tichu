"""Belief floor + (optional) trained-model eval over a materialised bundle.

The NAIVE floor is the max-entropy predictor that knows ONLY opponent hand sizes:
for an unseen card, P(opponent_i holds it) = hand_size_i / total_unseen. A trained
Belief Model that learned *which* cards opponents hold must beat it.

Three comparable metrics over masked positions:
  * masked BCE          — comparable to train_belief epoch.csv `loss`
  * masked acc@0.5      — comparable to epoch.csv `accuracy` (weak: ~0.70 by
                          predicting "no" everywhere)
  * which-opponent top-1 — for each unseen card, argmax over 3 opponents vs the
                          true holder (chance 0.333). The discriminating metric.

With --checkpoint it also scores a trained Belief Checkpoint on the SAME
examples, so naive vs trained lines up apples-to-apples.

Usage:
  py -3.14 scripts/belief_naive_baseline.py --bundle <belief_dir> [--max-examples N]
  py -3.14 scripts/belief_naive_baseline.py --bundle <belief_dir> `
      --checkpoint <run>/checkpoints/belief_final.bin
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import numpy as np

from tichu_training.belief.belief_materialised import MemmapBeliefDataset

_EPS = 1e-7


# Round-progress buckets by cards_played (= 56 - sum(hand_sizes)). Belief is
# near-useless in the opening (cards ~uniform) and most informative late.
_BUCKETS = [(0, 14, "open"), (14, 28, "mid"), (28, 42, "late"), (42, 57, "end")]


def _bucket(cards_played: int) -> int:
    for i, (lo, hi, _) in enumerate(_BUCKETS):
        if lo <= cards_played < hi:
            return i
    return len(_BUCKETS) - 1


class _Metrics:
    def __init__(self) -> None:
        self.bce = 0.0
        self.pos = 0       # masked (opp, card) positions
        self.acc = 0
        self.top1 = 0
        self.cards = 0     # unseen cards
        self.b_top1 = [0] * len(_BUCKETS)   # per round-progress bucket
        self.b_cards = [0] * len(_BUCKETS)

    def add(self, probs, labels, card_mask, cards_played: int) -> None:
        p = np.clip(probs[:, card_mask], _EPS, 1.0 - _EPS)   # (3, U)
        y = labels[:, card_mask]                              # (3, U)
        self.bce += float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
        self.pos += y.size
        self.acc += int(((p > 0.5).astype(y.dtype) == y).sum())
        hits = int((p.argmax(axis=0) == y.argmax(axis=0)).sum())
        u = int(card_mask.sum())
        self.top1 += hits
        self.cards += u
        bi = _bucket(cards_played)
        self.b_top1[bi] += hits
        self.b_cards[bi] += u

    def report(self, name: str) -> None:
        by = "  ".join(
            f"{_BUCKETS[i][2]}={self.b_top1[i] / max(1, self.b_cards[i]):.3f}"
            for i in range(len(_BUCKETS))
        )
        print(f"{name:>10}  BCE={self.bce / max(1, self.pos):.4f}  "
              f"acc@0.5={self.acc / max(1, self.pos):.4f}  "
              f"top1={self.top1 / max(1, self.cards):.4f}   [by round: {by}]")


def _load_model(checkpoint: Path):
    import torch

    from tichu_training.belief.model import BeliefModel
    from tichu_training.checkpoint import Checkpoint
    from tichu_training.featurizer import FEATURIZER_VERSION

    ckpt = Checkpoint.load(checkpoint, expected_featurizer_version=FEATURIZER_VERSION)
    state = torch.load(io.BytesIO(ckpt.payload), weights_only=True)["model"]
    # Both dims come off the checkpoint's first layer — no tier argument needed
    # now that the belief input is just the policy Feature Vector (ADR-0038).
    hidden, feature_dim = (int(d) for d in state["fc1.weight"].shape)
    model = BeliefModel(feature_dim=feature_dim, hidden=hidden)
    model.load_state_dict(state)
    model.eval()
    return model, feature_dim


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--max-examples", type=int, default=500_000)
    p.add_argument("--checkpoint", type=Path, default=None)
    args = p.parse_args()

    model = None
    if args.checkpoint is not None:
        model, input_dim = _load_model(args.checkpoint)

    ds = MemmapBeliefDataset(args.bundle)
    naive = _Metrics()
    trained = _Metrics() if model is not None else None

    import torch
    n = 0
    for ex in ds:
        if n >= args.max_examples:
            break
        n += 1
        labels = np.asarray(ex.labels)
        card_mask = np.asarray(ex.mask)[0].astype(bool)
        unseen = int(card_mask.sum())
        if unseen == 0:
            continue
        # naive: per-opponent hand-size proportion, broadcast over the 56 cards.
        p_i = (labels.sum(axis=1) / unseen)[:, None] * np.ones((1, 56), dtype=np.float32)
        naive.add(p_i, labels, card_mask, ex.cards_played)
        if model is not None:
            with torch.no_grad():
                feats = torch.from_numpy(ex.features[:input_dim]).unsqueeze(0)
                probs = torch.sigmoid(model(feats))[0].numpy()
            trained.add(probs, labels, card_mask, ex.cards_played)

    print(f"examples scanned: {n}  (chance top1 ~0.333)")
    naive.report("naive")
    if trained is not None:
        trained.report("trained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
