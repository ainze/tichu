"""THROWAWAY capacity differential for the Schupfen net (DEBUG-cap7e2).

Question (user): "calls and schupfen are small networks. Wouldn't it improve
on a bigger/deeper trunk?"

Decisive signal: can a WIDER / DEEPER net push held-out NLL (and train loss)
below the hidden=256 baseline ON THE SAME DATA? If train loss is already
plateaued at the small net's level and a 4x-wider / residual-deep net cannot
beat it, then ~2.79 is the irreducible human-schupfen entropy (multiple
reasonable cards to pass), not a capacity wall — and bigger trunk won't help.

Method: decode a contiguous N-row block once (sequential memmap, ~1M rows/s),
split games train/val by game_id hash (no within-deal leakage), train each arm
to plateau on GPU, score the SAME held-out val rows. Train loss = capacity to
fit; val NLL = generalization.

Run: PYTHONPATH=<worktree>/src python scripts/_diag_schupfen_capacity.py
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tichu_training.bc.schupfen_materialised import MemmapSchupfenDataset
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.card_slots import CARD_SLOTS


def val_bucket(game_id: np.ndarray, seed: int, n_buckets: int = 10_000) -> np.ndarray:
    """splitmix64 game-level hash → bucket (same family as call_training)."""
    with np.errstate(over="ignore"):
        z = game_id.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return (z % np.uint64(n_buckets)).astype(np.int64)


class _ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim); self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim); self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.fc1(self.ln1(x)))
        h = self.fc2(self.ln2(h))
        return x + h


class ResidualSchupfenNet(nn.Module):
    """Mirror the play TichuTrunk: input_proj → `depth` residual blocks → heads."""

    def __init__(self, feature_dim: int, *, hidden=1024, depth=4, skill_dim=64):
        super().__init__()
        self.skill = nn.Embedding(11, skill_dim)
        self.input_proj = nn.Linear(feature_dim + skill_dim, hidden)
        self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(depth)])
        self.h_next = nn.Linear(hidden, CARD_SLOTS)
        self.h_partner = nn.Linear(hidden, CARD_SLOTS)
        self.h_prev = nn.Linear(hidden, CARD_SLOTS)

    def forward(self, features, skill_decile):
        h = self.input_proj(torch.cat([features, self.skill(skill_decile)], dim=-1))
        for b in self.blocks:
            h = b(h)
        return self.h_next(h), self.h_partner(h), self.h_prev(h)


def loss_and_acc(net, feats, mask, target, skill, weight=None):
    """Sum of 3 hand-masked cross-entropies; per-direction top-1 accuracy."""
    logits = net(feats, skill)
    neg_inf = torch.finfo(logits[0].dtype).min
    per_sample = torch.zeros(feats.shape[0], device=feats.device)
    correct = 0
    for d, ld in enumerate(logits):
        masked = ld.masked_fill(mask == 0, neg_inf)
        per_sample = per_sample + F.cross_entropy(masked, target[:, d], reduction="none")
        correct += (masked.argmax(-1) == target[:, d]).float().sum().item()
    if weight is not None:
        loss = (per_sample * weight).mean()
    else:
        loss = per_sample.mean()
    return loss, per_sample.detach(), correct


def evaluate(net, feats, mask, target, skill, bs=8192):
    net.eval()
    n = feats.shape[0]
    tot_nll = 0.0
    tot_correct = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            _, ps, c = loss_and_acc(net, feats[sl], mask[sl], target[sl], skill[sl])
            tot_nll += float(ps.sum())
            tot_correct += c
    net.train()
    return tot_nll / n, tot_correct / (3 * n)


def train_arm(name, net, dev, train, val, *, epochs, bs, lr, seed):
    net = net.to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = train["feats"].shape[0]
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    last_train = float("nan")
    for ep in range(epochs):
        order = rng.permutation(n)
        run_loss = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = torch.from_numpy(order[i:i + bs]).to(dev)
            loss, _, _ = loss_and_acc(
                net, train["feats"][idx], train["mask"][idx],
                train["target"][idx], train["skill"][idx], train["weight"][idx],
            )
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss.detach()); nb += 1
        last_train = run_loss / nb
        vnll, vacc = evaluate(net, val["feats"], val["mask"], val["target"], val["skill"])
        print(f"  [{name}] epoch {ep}: train_loss={last_train:.4f}  "
              f"val_NLL={vnll:.4f}  val_acc={vacc:.4f}", flush=True)
    dt = time.perf_counter() - t0
    vnll, vacc = evaluate(net, val["feats"], val["mask"], val["target"], val["skill"])
    print(f"  [{name}] DONE params={n_params/1e6:.2f}M  {dt:.0f}s  "
          f"final train_loss={last_train:.4f}  val_NLL={vnll:.4f}  val_acc={vacc:.4f}",
          flush=True)
    return {"name": name, "params": n_params, "train_loss": last_train,
            "val_nll": vnll, "val_acc": vacc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=8_000_000)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MemmapSchupfenDataset("data/materialised_full_v6/schupfen")
    fdim = ds.feature_dim
    print(f"device={dev} corpus={len(ds)} feat_dim={fdim} "
          f"block={args.rows} epochs={args.epochs}", flush=True)

    print("decoding block (sequential)…", flush=True)
    t0 = time.perf_counter()
    idx = np.arange(0, args.rows)
    b = ds._gather_batch(idx)
    gid = np.asarray(ds._meta["game_id"][:args.rows])
    print(f"  decoded {args.rows} rows in {time.perf_counter()-t0:.0f}s, "
          f"{len(np.unique(gid))} distinct games", flush=True)

    is_val = val_bucket(gid, args.seed) < int(round(args.val_frac * 10_000))
    print(f"  split: train={int((~is_val).sum())}  val={int(is_val.sum())}", flush=True)

    def pack(sel):
        return {
            "feats": torch.from_numpy(b["features"][sel]).to(dev),
            "mask": torch.from_numpy(b["hand_mask"][sel]).to(dev),
            "target": torch.from_numpy(b["target"][sel]).long().to(dev),
            "skill": torch.from_numpy(b["skill_decile"][sel]).long().to(dev),
            "weight": torch.from_numpy(b["sample_weight"][sel]).float().to(dev),
        }

    # Features for train pool kept on GPU (8M*591*4 ~ 19GB won't fit 12GB VRAM);
    # keep on CPU, move per-batch. Val is smaller — keep on GPU.
    train_np = ~is_val
    train = {
        "feats": torch.from_numpy(b["features"][train_np]),
        "mask": torch.from_numpy(b["hand_mask"][train_np]),
        "target": torch.from_numpy(b["target"][train_np]).long(),
        "skill": torch.from_numpy(b["skill_decile"][train_np]).long(),
        "weight": torch.from_numpy(b["sample_weight"][train_np]).float(),
    }
    val = pack(is_val)

    # move-per-batch helper since train tensors live on CPU
    def train_arm_cpu(name, net):
        net = net.to(dev)
        n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        n = train["feats"].shape[0]
        rng = np.random.default_rng(args.seed)
        t0 = time.perf_counter()
        last = float("nan")
        for ep in range(args.epochs):
            order = rng.permutation(n)
            rl = 0.0; nb = 0
            for i in range(0, n, args.bs):
                sel = order[i:i + args.bs]
                fe = train["feats"][sel].to(dev, non_blocking=True)
                ma = train["mask"][sel].to(dev, non_blocking=True)
                ta = train["target"][sel].to(dev, non_blocking=True)
                sk = train["skill"][sel].to(dev, non_blocking=True)
                we = train["weight"][sel].to(dev, non_blocking=True)
                loss, _, _ = loss_and_acc(net, fe, ma, ta, sk, we)
                opt.zero_grad(); loss.backward(); opt.step()
                rl += float(loss.detach()); nb += 1
            last = rl / nb
            vnll, vacc = evaluate(net, val["feats"], val["mask"], val["target"], val["skill"])
            print(f"  [{name}] epoch {ep}: train_loss={last:.4f}  "
                  f"val_NLL={vnll:.4f}  val_acc={vacc:.4f}", flush=True)
        dt = time.perf_counter() - t0
        vnll, vacc = evaluate(net, val["feats"], val["mask"], val["target"], val["skill"])
        print(f"  [{name}] DONE params={n_params/1e6:.2f}M {dt:.0f}s "
              f"train_loss={last:.4f} val_NLL={vnll:.4f} val_acc={vacc:.4f}", flush=True)
        return {"name": name, "params": n_params, "train_loss": last,
                "val_nll": vnll, "val_acc": vacc}

    arms = [
        ("base_256x2", SchupfenNetwork(fdim, hidden=256)),       # current arch
        ("wide_1024x2", SchupfenNetwork(fdim, hidden=1024)),     # 4x wider
        ("resid_512x4", ResidualSchupfenNet(fdim, hidden=512, depth=4)),
        ("resid_1024x4", ResidualSchupfenNet(fdim, hidden=1024, depth=4)),  # play-trunk style
    ]
    results = []
    for name, net in arms:
        torch.manual_seed(args.seed)
        print(f"--- arm {name} ---", flush=True)
        results.append(train_arm_cpu(name, net))

    print("\n=== SUMMARY (DEBUG-cap7e2) ===")
    print(f"{'arm':<16}{'params':>9}{'train_loss':>12}{'val_NLL':>10}{'val_acc':>9}")
    for r in results:
        print(f"{r['name']:<16}{r['params']/1e6:>8.2f}M{r['train_loss']:>12.4f}"
              f"{r['val_nll']:>10.4f}{r['val_acc']:>9.4f}")
    base = results[0]
    print("\ndelta vs base_256x2 (negative NLL = better):")
    for r in results[1:]:
        print(f"  {r['name']:<16} d_val_NLL={r['val_nll']-base['val_nll']:+.4f}  "
              f"d_val_acc={r['val_acc']-base['val_acc']:+.4f}  "
              f"d_train={r['train_loss']-base['train_loss']:+.4f}")


if __name__ == "__main__":
    main()
