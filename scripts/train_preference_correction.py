"""Run one Preference Correction arm over a Correction Corpus (ADR-0042).

  py -m scripts.train_preference_correction --arm pair \
      --corpus C:/workbench/tichu/data/runs/correction_corpus_iter27008/corpus.parquet \
      --drift  C:/workbench/tichu/data/runs/correction_corpus_iter27008/drift.parquet \
      --base-export C:/workbench/tichu/data/runs/cotrain_v6_pbrs_resid_wish_gated_wishfix/_ship_check_export \
      --config configs/cotrain_v6_pbrs_resid_wish_gated_wishfix.yaml \
      --out-export C:/workbench/tichu/data/runs/correction_corpus_iter27008/export_pair

`--arm ce` runs the 2026-06-11 recipe instead. That control arm is MANDATORY
before reading the pair arm: if CE also comes out fine on this corpus then the
June kill was about lineage or corpus size, not the delivery mechanism.

Output is a full four-net export dir — the fine-tuned play net plus the base
export's schupfen / tichu_call / grand_tichu_call copied verbatim, since only
play is trained. That makes it directly loadable by an MLAgent in a Tournament.

Every number printed here is a SELECTION dial. ADR-0040 convicted correction
fix-rate as a strength proxy; the verdict instrument is the Tournament, at the
n committed in ADR-0042 §Realised corpus (CE 8k, PAIR 40k).
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

# Resolve THIS tree's src ahead of the editable install (which points at the main
# checkout) — see scripts/mine_corrections.py for the full rationale.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_OTHER_NETS = ("schupfen.pt", "tichu_call.pt", "grand_tichu_call.pt")


def main(argv=None) -> int:
    import yaml

    from tichu_training.bc.training import load_checkpoint
    from tichu_training.cli.train_cotrain import _build_models
    from tichu_training.correction.train import (
        corpus_dials,
        rows_from_frame,
        split_by_round,
        train_arm,
    )

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("pair", "ce"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--drift", required=True)
    ap.add_argument("--base-export", required=True,
                    help="four-net export the corpus was MINED against")
    ap.add_argument("--config", required=True, help="cotrain config (model arch block)")
    ap.add_argument("--out-export", required=True)
    ap.add_argument("--skill-decile", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--anchor-coef", type=float, default=10.0)
    ap.add_argument("--delta-scale", type=float, default=50.0,
                    help="pairwise margin = mean_delta / this")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-provenance-mismatch", action="store_true",
                    help="skip the argmax_is_chosen guard (you almost never want this)")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    model = _build_models(config)["play"]
    base = Path(args.base_export)
    play = base / "policy.pt"
    if play.suffix == ".pt":
        model.load_state_dict(torch.jit.load(play, map_location="cpu").state_dict())
    else:
        load_checkpoint(str(play), model)
    model.to(args.device)

    corpus = pd.read_parquet(args.corpus)
    drift = pd.read_parquet(args.drift)
    train_df, held_df = split_by_round(corpus)
    train_rows = rows_from_frame(train_df, skill_decile=args.skill_decile)
    held_rows = rows_from_frame(held_df, skill_decile=args.skill_decile)
    drift_rows = rows_from_frame(drift, skill_decile=args.skill_decile)
    print(f"corpus {len(corpus):,} rows -> {len(train_rows):,} train / "
          f"{len(held_rows):,} held-out (disjoint Rounds); drift {len(drift_rows):,}",
          flush=True)

    before_t = corpus_dials(_logits(model, train_rows), train_rows)
    before_h = corpus_dials(_logits(model, held_rows), held_rows)
    print(f"  train    corrections {before_t['n_corrections']:,} / "
          f"non {before_t['n_non_corrections']:,}", flush=True)
    print(f"  held-out corrections {before_h['n_corrections']:,} / "
          f"non {before_h['n_non_corrections']:,}", flush=True)

    # PROVENANCE GUARD. Every `chosen` is the mining policy's legal-masked argmax,
    # so this reads ~1.0 iff these are the weights the corpus was mined against.
    prov = before_t["argmax_is_chosen"]
    print(f"\nprovenance: argmax_is_chosen = {prov:.4f} "
          f"(fix_rate {before_t['fix_rate']:.4f}, preservation "
          f"{before_t['preservation']:.4f})", flush=True)
    if prov < 0.99 and not args.allow_provenance_mismatch:
        raise SystemExit(
            f"argmax_is_chosen is {prov:.4f}, expected ~1.0. These weights are not "
            f"the ones {args.corpus} was mined against, so fine-tuning them answers "
            "a different question and nothing downstream would notice. Check "
            "--base-export, or pass --allow-provenance-mismatch if deliberate.")

    print(f"\narm={args.arm} epochs={args.epochs} lr={args.lr} "
          f"anchor={args.anchor_coef} delta_scale={args.delta_scale} "
          f"device={args.device}", flush=True)
    from tqdm import tqdm
    bar = tqdm(total=args.epochs, unit="ep", desc=f"train:{args.arm}",
               dynamic_ncols=True, disable=not sys.stderr.isatty())
    train_arm(model, train_rows, drift_rows, arm=args.arm, epochs=args.epochs,
              lr=args.lr, anchor_coef=args.anchor_coef,
              delta_scale=args.delta_scale,
              log=lambda ep, ls: (bar.update(1), bar.set_postfix(loss=f"{ls:.4f}",
                                                                 refresh=False)))
    bar.close()

    after_t = corpus_dials(_logits(model, train_rows), train_rows)
    after_h = corpus_dials(_logits(model, held_rows), held_rows)
    print("\n=== selection dials (NOT a verdict — ADR-0040 convicted fix-rate) ===",
          flush=True)
    hdr = f"{'':10s} {'fix_rate':>10s} {'preservation':>13s}"
    print(hdr, flush=True)
    for name, b, a in (("train", before_t, after_t), ("held-out", before_h, after_h)):
        print(f"{name:10s} {b['fix_rate']:.3f}->{a['fix_rate']:.3f}   "
              f"{b['preservation']:.3f}->{a['preservation']:.3f}", flush=True)
    print(f"\nheld-out TRANSFER {after_h['fix_rate'] - before_h['fix_rate']:+.3f}   "
          f"held-out DAMAGE {before_h['preservation'] - after_h['preservation']:+.3f}",
          flush=True)

    out = Path(args.out_export)
    out.mkdir(parents=True, exist_ok=True)
    _export(model, out / "policy.pt")
    for net in _OTHER_NETS:
        shutil.copyfile(base / net, out / net)
    print(f"\nfour-net export -> {out}  (play fine-tuned; {', '.join(_OTHER_NETS)} "
          "copied from the base — only play is trained)", flush=True)
    return 0


def _logits(model, rows):
    from tichu_training.correction.train import _batched_logits

    return _batched_logits(model, rows).detach().cpu()


def _export(model, path: Path) -> None:
    from tichu_export.torchscript import export_torchscript
    from tichu_training.action_space import ACTION_SPACE_VERSION
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION

    was = next(model.parameters()).device
    model.cpu()
    export_torchscript(
        model,
        example_inputs=(torch.randn(1, FEATURIZER_OUTPUT_DIM),
                        torch.tensor([0], dtype=torch.long)),
        featurizer_version=FEATURIZER_VERSION,
        action_space_version=ACTION_SPACE_VERSION,
        output_path=str(path),
    )
    model.to(was)


if __name__ == "__main__":
    raise SystemExit(main())
