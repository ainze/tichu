"""Distill premise test — do mined corrections TRANSFER, or only memorize?

The mine->correct->distill training bet stands or falls on generalization: the
confirmed blunders are context-unique (max cluster n=2), and deal states never
recur, so a policy that merely memorizes its training corrections gains nothing
(the typed-critic lesson: round-unique memorization, zero held-out value).

The cheap decisive test: fine-tune the play net on a TRAIN split of confirmed
corrections, then measure the fix-rate on a HELD-OUT split (disjoint rounds).
  * held-out fix-rate stays at ~baseline -> NO transfer -> kill the distill bet
    (the paired-world signal must instead be wired into RL — the vine/duplicate-
    deal route).
  * held-out fix-rate rises materially with bounded drift on ordinary decisions
    -> transfer is real -> the full mine->distill loop is worth designing.

Baseline fix-rate is 0% by construction (the policy's argmax IS the blundered
action on every correction row), so any held-out movement is signal; the drift
guard (top-1 agreement with the anchor on ordinary decisions) catches the
trivial way to "fix" things — wholesale distribution shift.

  py -m scripts.distill_premise_test --corrections .../corrections.parquet \
      --ordinary .../ordinary.parquet --config configs/cotrain_wish_v5.yaml \
      --play-checkpoint .../snapshots/iter_06225_play.bin
"""

import argparse
import copy
import json
import sys

import pandas as pd
import torch

HELD_MOD = 4  # round_idx % 4 == 3 -> held-out (25%, split by ROUND, no leakage)


def split_by_round(df: pd.DataFrame, held_mod: int = HELD_MOD):
    held = df[df["round_idx"] % held_mod == held_mod - 1]
    train = df[df["round_idx"] % held_mod != held_mod - 1]
    return train, held


def rows_from_frame(df: pd.DataFrame, *, skill_decile: int) -> list[dict]:
    """Materialise tensor rows once (features, legal mask indices, targets)."""
    out = []
    for rec in df.to_dict("records"):
        out.append(
            {
                "features": torch.tensor(rec["features"], dtype=torch.float32),
                "skill": torch.tensor([skill_decile], dtype=torch.long),
                "legal_idx": [int(i) for i in rec["legal_idx"]],
                "chosen_idx": int(rec["chosen_idx"]),
                "alt_idx": int(rec.get("alt_idx", rec["chosen_idx"])),
            }
        )
    return out


def _play_logits(model, row):
    return model(row["features"].unsqueeze(0), row["skill"])["play"][0]


def _batched_play_logits(model, rows: list[dict], *, chunk: int = 256):
    """One batched forward over all rows (features are fixed-dim) — the loop over
    rows happens on the cheap per-row masked ops, not on 40M-param forwards."""
    device = next(model.parameters()).device
    outs = []
    for lo in range(0, len(rows), chunk):
        part = rows[lo:lo + chunk]
        feats = torch.stack([r["features"] for r in part]).to(device)
        skills = torch.cat([r["skill"] for r in part]).to(device)
        outs.append(model(feats, skills)["play"])
    return torch.cat(outs, dim=0)


def masked_logprob(logits, *, legal_idx: list[int], target_idx: int):
    """Log-probability of `target_idx` under the softmax restricted to the legal
    set — the same legality constraint the served decode applies."""
    legal = torch.as_tensor(legal_idx, dtype=torch.long, device=logits.device)
    sub = logits[legal]
    pos = legal_idx.index(target_idx)
    return torch.log_softmax(sub, dim=0)[pos]


def _top1(logits, legal_idx: list[int]) -> int:
    legal = torch.as_tensor(legal_idx, dtype=torch.long, device=logits.device)
    return int(legal[int(torch.argmax(logits[legal]))])


@torch.no_grad()
def fix_rate(model, rows: list[dict]) -> float:
    """Fraction of correction rows where the model's legal-masked top-1 IS the
    verified-better alternative."""
    if not rows:
        return 0.0
    logits = _batched_play_logits(model, rows).cpu()
    hits = sum(_top1(logits[i], r["legal_idx"]) == r["alt_idx"]
               for i, r in enumerate(rows))
    return hits / len(rows)


@torch.no_grad()
def agreement(model_a, model_b, rows: list[dict]) -> float:
    """Top-1-over-legal agreement between two models (the drift guard)."""
    if not rows:
        return 1.0
    la = _batched_play_logits(model_a, rows).cpu()
    lb = _batched_play_logits(model_b, rows).cpu()
    same = sum(
        _top1(la[i], r["legal_idx"]) == _top1(lb[i], r["legal_idx"])
        for i, r in enumerate(rows)
    )
    return same / len(rows)


def finetune(model, anchor, train_rows: list[dict], *, ordinary_rows: list[dict],
             epochs: int, lr: float, anchor_coef: float) -> None:
    """CE toward the verified-better action on the corrections + KL-to-anchor on
    ordinary decisions (keeps the policy itself; only the blunders should move).
    One batched forward per loss term per epoch; the per-row masked ops are cheap."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    anchor.eval()
    with torch.no_grad():
        anchor_logits = _batched_play_logits(anchor, ordinary_rows) if ordinary_rows else None
    for _ in range(epochs):
        opt.zero_grad()
        logits = _batched_play_logits(model, train_rows)
        loss = -torch.stack([
            masked_logprob(logits[i], legal_idx=r["legal_idx"], target_idx=r["alt_idx"])
            for i, r in enumerate(train_rows)
        ]).mean()
        if anchor_coef > 0.0 and ordinary_rows:
            ord_logits = _batched_play_logits(model, ordinary_rows)
            kls = []
            for i, r in enumerate(ordinary_rows):
                legal = torch.as_tensor(r["legal_idx"], dtype=torch.long,
                                        device=ord_logits.device)
                p_anchor = torch.log_softmax(anchor_logits[i][legal], dim=0)
                p_model = torch.log_softmax(ord_logits[i][legal], dim=0)
                kls.append(torch.nn.functional.kl_div(
                    p_model, p_anchor, log_target=True, reduction="sum"))
            loss = loss + anchor_coef * torch.stack(kls).mean()
        loss.backward()
        opt.step()


def main(argv=None) -> int:
    import yaml

    from tichu_training.bc.training import load_checkpoint
    from tichu_training.cli.train_cotrain import _build_models

    parser = argparse.ArgumentParser(description="Mined-correction transfer premise test")
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--ordinary", required=True)
    parser.add_argument("--config", required=True, help="cotrain config (model arch block)")
    parser.add_argument("--play-checkpoint", required=True,
                        help="play weights: a raw .bin snapshot, or a TorchScript "
                             "export .pt (state_dict round-trips into BCModel — used "
                             "when the raw snapshot was pruned). If recovery is exact, "
                             "the BEFORE fix-rates read 0.000 (chosen == argmax).")
    parser.add_argument("--skill-decile", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--anchor-coef", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="optional JSON result path")
    parser.add_argument("--save-policy", default=None,
                        help="export the fine-tuned play net as a TorchScript policy.pt "
                             "(MLAgent-loadable) for a tournament read")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    model = _build_models(config)["play"]
    if args.play_checkpoint.endswith(".pt"):
        scripted = torch.jit.load(args.play_checkpoint, map_location="cpu")
        model.load_state_dict(scripted.state_dict())
    else:
        load_checkpoint(args.play_checkpoint, model)
    anchor = copy.deepcopy(model)
    model.to(args.device)
    anchor.to(args.device)

    corr = pd.read_parquet(args.corrections)
    train_df, held_df = split_by_round(corr)
    train_rows = rows_from_frame(train_df, skill_decile=args.skill_decile)
    held_rows = rows_from_frame(held_df, skill_decile=args.skill_decile)
    ordinary_rows = rows_from_frame(pd.read_parquet(args.ordinary),
                                    skill_decile=args.skill_decile)
    print(f"corrections: {len(train_rows)} train / {len(held_rows)} held-out "
          f"(disjoint rounds); ordinary drift set: {len(ordinary_rows)}", flush=True)

    before = {
        "train_fix": fix_rate(model, train_rows),
        "held_fix": fix_rate(model, held_rows),
        "drift": agreement(model, anchor, ordinary_rows),
    }
    print(f"before: train_fix={before['train_fix']:.3f} "
          f"held_fix={before['held_fix']:.3f} drift_agreement={before['drift']:.3f}",
          flush=True)

    finetune(model, anchor, train_rows, ordinary_rows=ordinary_rows,
             epochs=args.epochs, lr=args.lr, anchor_coef=args.anchor_coef)

    after = {
        "train_fix": fix_rate(model, train_rows),
        "held_fix": fix_rate(model, held_rows),
        "drift": agreement(model, anchor, ordinary_rows),
    }
    print(f"after:  train_fix={after['train_fix']:.3f} "
          f"held_fix={after['held_fix']:.3f} drift_agreement={after['drift']:.3f}",
          flush=True)

    transfer = after["held_fix"] - before["held_fix"]
    print(f"\nTRANSFER (held-out fix-rate gain): {transfer:+.3f}  "
          f"(train gain {after['train_fix'] - before['train_fix']:+.3f}, "
          f"ordinary drift {1 - after['drift']:.3f})", flush=True)
    result = {"before": before, "after": after, "transfer": transfer,
              "n_train": len(train_rows), "n_held": len(held_rows),
              "epochs": args.epochs, "lr": args.lr, "anchor_coef": args.anchor_coef}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    if args.save_policy:
        from tichu_export.torchscript import export_torchscript
        from tichu_training.action_space import ACTION_SPACE_VERSION
        from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION

        model.cpu()
        export_torchscript(
            model,
            example_inputs=(torch.randn(1, FEATURIZER_OUTPUT_DIM),
                            torch.tensor([0], dtype=torch.long)),
            featurizer_version=FEATURIZER_VERSION,
            action_space_version=ACTION_SPACE_VERSION,
            output_path=args.save_policy,
        )
        print(f"fine-tuned policy exported -> {args.save_policy}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
