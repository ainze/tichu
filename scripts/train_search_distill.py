r"""Fine-tune an exported policy on played-out move comparisons (test build).

Loads the TorchScript `policy.pt` of an export, and for each emitted decision
raises the probability of the candidates that scored better when played out,
relative to the other candidates, while a KL term holds the whole legal-move
distribution near the original policy. The objective is linear in the played-
out scores, so one noisy playout per move is an unbiased training signal.

Writes a full export dir (new `policy.pt`, the other nets copied unchanged).

  py -3.14 -m scripts.train_search_distill `
      --shards C:\workbench\tichu\data\runs\search_distill_v1\shards `
      --export-dir C:\workbench\tichu\data\runs\cotrain_v7_gated\export\iter_15360 `
      --out-dir C:\workbench\tichu\data\runs\search_distill_v1\fit_a
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

KEYS = ("features", "mask", "cand", "q", "phase", "deal")


def load(shard_dir: Path, limit):
    files = sorted(f for f in shard_dir.glob("shard_*.npz") if ".tmp" not in f.name)
    if limit:
        files = files[:limit]
    cols = {k: [] for k in KEYS}
    for f in files:
        d = np.load(f)
        for k in KEYS:
            cols[k].append(d[k])
    return {k: np.concatenate(v) for k, v in cols.items()}, len(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--kl", type=float, default=1.0, help="weight of KL(old || new)")
    ap.add_argument("--clip", type=float, default=3.0,
                    help="clip of the advantage, in points / 100")
    ap.add_argument("--skill-decile", type=int, default=9)
    args = ap.parse_args(argv)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, n_files = load(Path(args.shards), args.limit_shards)
    feats = data["features"]
    n_actions = None
    cand = data["cand"].astype(np.int64)
    valid = cand >= 0
    adv = np.where(valid, np.nan_to_num(data["q"]) - data["q"][:, :1], 0.0).astype(np.float32)
    deal = data["deal"]
    held = deal % 10 == 0
    train_idx, val_idx = np.flatnonzero(~held), np.flatnonzero(held)
    print(f"{n_files} shards, {len(deal):,} decisions, {len(np.unique(deal)):,} deals | "
          f"train {len(train_idx):,} / val {len(val_idx):,} | device {device}", flush=True)
    best_alt = np.where(valid, adv, -np.inf).max(axis=1)
    print(f"alternatives: mean advantage {100 * adv[valid & (np.arange(4) > 0)].mean():+.1f} pts, "
          f"oracle gain {100 * np.maximum(best_alt, 0).mean():+.1f} pts / decision", flush=True)

    export = Path(args.export_dir)
    # The export's version stamps ride along as extra files; carry them over.
    stamps = {"featurizer_version": b"", "action_space_version": b"", "use_legal_mask": b""}
    model = torch.jit.load(str(export / "policy.pt"), map_location=device,
                           _extra_files=stamps)
    anchor = torch.jit.load(str(export / "policy.pt"), map_location=device)
    model.eval()
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    def batch(idx):
        x = torch.from_numpy(feats[idx].astype(np.float32)).to(device)
        m = torch.from_numpy(np.unpackbits(data["mask"][idx], axis=1)).to(device).bool()
        skill = torch.full((len(idx),), args.skill_decile, dtype=torch.long, device=device)
        c = torch.from_numpy(cand[idx]).to(device)
        a = torch.from_numpy(adv[idx]).to(device)
        return x, m, skill, c, a

    def forward(net, x, m, skill):
        logits = net(x, skill, m[:, :n_actions].float())["play"]
        return logits, logits.masked_fill(~m[:, :logits.shape[1]], float("-inf"))

    # Action-space width from one forward (packbits pads the mask to a byte).
    with torch.no_grad():
        x, m, skill, c, a = batch(val_idx[:2])
        n_actions = next(w for w in range(m.shape[1], m.shape[1] - 8, -1)
                         if _try(anchor, x, skill, m[:, :w]))
    print(f"action space {n_actions}", flush=True)

    def cand_logits(masked, c):
        g = masked.gather(1, c.clamp(min=0))
        return g.masked_fill(c < 0, float("-inf"))

    def evaluate():
        gains, flips, kls, moved, outside = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(val_idx), 8192):
                x, m, skill, c, a = batch(val_idx[i:i + 8192])
                _l, new = forward(model, x, m, skill)
                _l, old = forward(anchor, x, m, skill)
                pick = cand_logits(new, c).argmax(dim=1)
                gains.append(a.gather(1, pick[:, None])[:, 0].cpu().numpy())
                flips.append((pick != 0).float().cpu().numpy())
                top = new.argmax(dim=1)
                moved.append((top != c[:, 0]).float().cpu().numpy())
                outside.append((top[:, None] != c).all(dim=1).float().cpu().numpy())
                lo, ln = old.log_softmax(1), new.log_softmax(1)
                kl = (lo.exp() * (lo - ln).nan_to_num(0.0, 0.0, 0.0)).sum(1)
                kls.append(kl.cpu().numpy())
        g = np.concatenate(gains) * 100.0
        return {"gain_pts_per_decision": float(g.mean()),
                "gain_se": float(g.std() / np.sqrt(len(g))),
                "flip_rate": float(np.concatenate(flips).mean()),
                "top_move_changed": float(np.concatenate(moved).mean()),
                "top_move_unscored": float(np.concatenate(outside).mean()),
                "kl": float(np.concatenate(kls).mean())}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = max(1, len(train_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    history = []
    for epoch in range(args.epochs):
        perm = rng.permutation(train_idx)
        t0 = time.monotonic()
        for s in range(steps):
            idx = np.sort(perm[s * args.batch:(s + 1) * args.batch])
            x, m, skill, c, a = batch(idx)
            _l, new = forward(model, x, m, skill)
            with torch.no_grad():
                _l, old = forward(anchor, x, m, skill)
            p_c = cand_logits(new, c).softmax(dim=1)
            gain = (p_c * a.clamp(-args.clip, args.clip)).sum(1).mean()
            lo, ln = old.log_softmax(1), new.log_softmax(1)
            kl = (lo.exp() * (lo - ln).nan_to_num(0.0, 0.0, 0.0)).sum(1).mean()
            loss = -gain + args.kl * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        ev = evaluate()
        ev["epoch"] = epoch + 1
        history.append(ev)
        print(f"  epoch {epoch + 1}/{args.epochs} held-out gain "
              f"{ev['gain_pts_per_decision']:+.2f} +- {ev['gain_se']:.2f} pts/decision | "
              f"flips {ev['flip_rate']:.3f} | top changed {ev['top_move_changed']:.3f} "
              f"(unscored {ev['top_move_unscored']:.3f}) | KL {ev['kl']:.4f} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
        ep_dir = out_dir / f"epoch_{epoch + 1}"
        ep_dir.mkdir(exist_ok=True)
        model.cpu()
        torch.jit.save(model, str(ep_dir / "policy.pt"), _extra_files=dict(stamps))
        model.to(device)
        for name in ("schupfen.pt", "tichu_call.pt", "grand_tichu_call.pt"):
            shutil.copyfile(export / name, ep_dir / name)
    (out_dir / "report.json").write_text(json.dumps(
        {"args": vars(args), "decisions": int(len(deal)), "history": history}, indent=2))
    return 0


def _try(net, x, skill, m) -> bool:
    try:
        net(x, skill, m.float())
        return True
    except Exception:  # noqa: BLE001 - width probe
        return False


if __name__ == "__main__":
    sys.exit(main())
