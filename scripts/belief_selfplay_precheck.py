r"""ADR-0041 pre-check — is a Belief Model learnable on the champion's own
self-play distribution, and by how much over the card-counting floor?

The gate's headline (`Belief EV Value` = `D_on - D_off`) is zero by construction
if a Belief Model cannot beat a predictor that knows only `hand_sizes`. This
script answers that for a few CPU-hours, before any tournament spend.

Trains on **champion self-play** rather than the BSW corpus: both on-disk belief
checkpoints are v5 (unloadable against Featurizer v6), the BSW route needs a full
re-parse, and self-play is the distribution the **Belief-Optimal Chooser**
actually consumes — removing the human-corpus skew at the source.

Metric is which-opponent top-1 over **Unseen Cards** (chance ~1/3). Reference:
the v5 B-core model scored 0.505 aggregate on the BSW distribution.

  py -3.14 scripts/belief_selfplay_precheck.py `
    --export-dir C:\workbench\tichu\data\runs\cotrain_v6_pbrs_resid_wish_gated_cpfix\export\iter_03328 `
    --rounds 2000 --workers 10 --out C:\workbench\tichu\data\runs\belief_selfplay_v6\belief_final.bin

Launch long runs detached (`Start-Process`), not as a foreground call.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

_WORKER: dict = {}
_BUCKETS = [(0, 14, "open"), (14, 28, "mid"), (28, 42, "late"), (42, 57, "end")]


def _bucket(cards_played: int) -> int:
    for i, (lo, hi, _name) in enumerate(_BUCKETS):
        if lo <= cards_played < hi:
            return i
    return len(_BUCKETS) - 1


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


def _emit_task(task):
    from tichu_training.belief.selfplay_emit import belief_examples_for_selfplay_round

    round_idx, position, rich = task
    return belief_examples_for_selfplay_round(
        _WORKER["agents"], position, round_id=round_idx, rich_history=rich,
    )


def _report(name: str, per_bucket: dict[int, tuple[int, int]]) -> str:
    hits = sum(h for h, _ in per_bucket.values())
    cards = sum(c for _, c in per_bucket.values())
    by = "  ".join(
        f"{_BUCKETS[i][2]}={per_bucket[i][0] / max(1, per_bucket[i][1]):.4f}"
        for i in sorted(per_bucket)
    )
    return f"{name:8s} top1={hits / max(1, cards):.4f}   [{by}]"


def _score_by_bucket(examples, probs_for) -> dict[int, tuple[int, int]]:
    from tichu_training.belief.selfplay_emit import which_opponent_top1

    out: dict[int, tuple[int, int]] = {}
    for row, ex in enumerate(examples):
        h, c = which_opponent_top1(probs_for(row, ex), ex)
        if c == 0:
            continue
        b = _bucket(ex.cards_played)
        bh, bc = out.get(b, (0, 0))
        out[b] = (bh + h, bc + c)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ADR-0041 self-play belief pre-check")
    p.add_argument("--export-dir", required=True,
                   help="champion export dir (policy/schupfen/tichu_call/grand_tichu_call .pt)")
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--pool-seed", type=int, default=424000)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--skill-decile", type=int, default=9)
    p.add_argument("--holdout-frac", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=4,
                   help="residual blocks (only with --residual)")
    p.add_argument("--trunk-out-dim", type=int, default=None,
                   help="trunk output width (only with --residual); "
                        "--residual --hidden 1024 --depth 4 --trunk-out-dim 512 "
                        "is the play Policy Network's own trunk")
    p.add_argument("--residual", action="store_true",
                   help="use the policy's TichuTrunk instead of the 2-layer MLP")
    p.add_argument("--rich-history", action="store_true",
                   help="append RichHistory's recovered channels (decline context, "
                        "combo length, declined bombs, trick stakes, wish voids) — "
                        "v6's B-core block keeps only 27 numbers per Round")
    p.add_argument("--holder-loss", action="store_true",
                   help="per-card softmax over the 3 opponents instead of "
                        "independent sigmoids (matches what the world sampler draws)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", type=Path, default=None,
                   help="npz of emitted examples; reused if present, written if not "
                        "(makes capacity sweeps cheap — emit is the expensive part)")
    p.add_argument("--out", type=Path, default=None,
                   help="write the fitted Belief Checkpoint here (for the chooser)")
    args = p.parse_args(argv)

    from tichu_eval.full_position_pool import generate_full_position_pool
    from tichu_training.belief.selfplay_emit import (
        floor_marginals_for_example, load_examples, masked_bce,
        belief_probabilities, holder_nll, round_level_split,
        save_belief_checkpoint, save_examples, train_selfplay_belief,
    )

    examples = load_examples(args.cache) if args.cache else None
    if examples is None:
        positions = generate_full_position_pool(seed=args.pool_seed, n=args.rounds)
        tasks = [(i, pos, args.rich_history) for i, pos in enumerate(positions)]
        examples = []
        with mp.Pool(
            processes=args.workers, initializer=_init_worker,
            initargs=(str(args.export_dir), args.skill_decile),
        ) as pool:
            for i, rows in enumerate(
                pool.imap_unordered(_emit_task, tasks, chunksize=4), 1
            ):
                examples.extend(rows)
                if i % 100 == 0:
                    print(f"  emitted {i}/{len(tasks)} rounds "
                          f"({len(examples)} examples)", flush=True)
        if args.cache:
            save_examples(args.cache, examples)
            print(f"cached {len(examples)} examples -> {args.cache}", flush=True)
    else:
        print(f"loaded {len(examples)} cached examples from {args.cache}", flush=True)

    train, holdout = round_level_split(
        examples, holdout_frac=args.holdout_frac, seed=args.seed,
    )
    print(f"{len(examples)} examples "
          f"(train {len(train)} / holdout {len(holdout)})", flush=True)

    # Early stopping is mandatory: the hidden-hand target is round-unique, so an
    # unstopped fit memorises deals (measured: train top-1 0.998, holdout below
    # the floor) — the ADR-0033 round-level-split failure mode.
    model, history = train_selfplay_belief(
        train, holdout=holdout, hidden=args.hidden, depth=args.depth,
        trunk_out_dim=args.trunk_out_dim, residual=args.residual,
        holder_loss=args.holder_loss, epochs=args.epochs, seed=args.seed,
    )
    n_params = sum(p_.numel() for p_ in model.parameters())
    best = max(history, key=lambda row: row["holdout_top1"])
    print(f"\nhidden={args.hidden}  best epoch {best['epoch']} of {args.epochs} "
          f"(holdout top1 {best['holdout_top1']:.4f})")
    print("  per-epoch holdout top1: "
          + " ".join(f"{row['holdout_top1']:.4f}" for row in history))

    import torch

    def _predict(rows):
        feats = torch.from_numpy(np.stack([ex.features for ex in rows]))
        return belief_probabilities(model, feats, holder_loss=args.holder_loss)

    def _agg(rows, metric, probs_for) -> float:
        total = n = 0.0
        for row, ex in enumerate(rows):
            t, k = metric(probs_for(row, ex), ex)
            total += t
            n += k
        return total / max(1.0, n)

    def _bce(rows, probs_for) -> float:
        return _agg(rows, masked_bce, probs_for)

    # Train-split scores answer the underfit question: if the fit cannot beat the
    # floor on data it was trained on, a flat holdout is an optimisation artifact
    # rather than a statement about information.
    for split_name, rows in (("holdout", holdout), ("train", train)):
        probs = _predict(rows)
        floor_buckets = _score_by_bucket(
            rows, lambda _row, ex: floor_marginals_for_example(ex)
        )
        belief_buckets = _score_by_bucket(rows, lambda row, _ex: probs[row])
        floor_top1 = sum(h for h, _ in floor_buckets.values()) / max(
            1, sum(c for _, c in floor_buckets.values())
        )
        belief_top1 = sum(h for h, _ in belief_buckets.values()) / max(
            1, sum(c for _, c in belief_buckets.values())
        )
        floor_bce = _bce(rows, lambda _row, ex: floor_marginals_for_example(ex))
        belief_bce = _bce(rows, lambda row, _ex: probs[row])
        floor_nll = _agg(rows, holder_nll, lambda _row, ex: floor_marginals_for_example(ex))
        belief_nll = _agg(rows, holder_nll, lambda row, _ex: probs[row])

        print(f"\n== {split_name} ({len(rows)} examples)")
        print(_report("floor", floor_buckets))
        print(_report("belief", belief_buckets))
        print(f"top1  delta = {belief_top1 - floor_top1:+.4f}")
        print(f"BCE   floor={floor_bce:.4f}  belief={belief_bce:.4f}  "
              f"delta = {belief_bce - floor_bce:+.4f}  (lower is better)")
        print(f"NLL   floor={floor_nll:.4f}  belief={belief_nll:.4f}  "
              f"delta = {belief_nll - floor_nll:+.4f}  (holder NLL, ln3=1.0986)")

    feats = torch.from_numpy(np.stack([ex.features for ex in holdout]))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_belief_checkpoint(args.out, model, holder_loss=args.holder_loss)
        print(f"wrote {args.out}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
