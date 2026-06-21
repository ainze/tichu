"""THROWAWAY capacity differential for the Call nets (DEBUG-cap7e2).

Sibling of _diag_schupfen_capacity.py for the binary Tichu / Grand-Tichu call
nets. Same question: does a wider/deeper trunk lower held-out loss vs the
current hidden=256 3-layer MLP? Reports val NLL, accuracy, and AUC (the
class-imbalance-robust metric — tichu ~14% pos, grand ~8% pos).

Run: PYTHONPATH=<worktree>/src python scripts/_diag_calls_capacity.py
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tichu_training.bc.call_materialised import MemmapCallDataset
from tichu_training.bc.call_model import CallNetwork
from tichu_training.bc.call_training import _binary_auc


def val_bucket(game_id, seed, n_buckets=10_000):
    with np.errstate(over="ignore"):
        z = game_id.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return (z % np.uint64(n_buckets)).astype(np.int64)


class _ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim); self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim); self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.fc1(self.ln1(x)))
        return x + self.fc2(self.ln2(h))


class ResidualCallNet(nn.Module):
    def __init__(self, feature_dim, *, hidden=512, depth=4, skill_dim=64):
        super().__init__()
        self.skill = nn.Embedding(11, skill_dim)
        self.input_proj = nn.Linear(feature_dim + skill_dim, hidden)
        self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(depth)])
        self.out = nn.Linear(hidden, 2)

    def forward(self, features, skill_decile):
        h = self.input_proj(torch.cat([features, self.skill(skill_decile)], dim=-1))
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def evaluate(net, feats, target, skill, bs=16384):
    net.eval()
    n = feats.shape[0]
    tot_nll = 0.0; tot_correct = 0
    probs_all = []
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            logits = net(feats[sl], skill[sl])
            tot_nll += float(F.cross_entropy(logits, target[sl], reduction="sum"))
            tot_correct += int((logits.argmax(-1) == target[sl]).sum())
            probs_all.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    net.train()
    probs = np.concatenate(probs_all)
    labels = target.cpu().numpy().astype(np.int64)
    return tot_nll / n, tot_correct / n, _binary_auc(probs, labels)


def run_type(call_type, rows, val_frac, epochs, bs, lr, seed, dev):
    ds = MemmapCallDataset("data/materialised_full_v6/calls", call_type=call_type)
    fdim = ds.feature_dim
    n = min(rows, len(ds))
    print(f"\n############ {call_type}  (corpus {len(ds)}, block {n}) ############", flush=True)
    b = next(ds.iter_batches(n))  # one big sequential decode
    gid = np.asarray(ds._meta["game_id"][:n])
    is_val = val_bucket(gid, seed) < int(round(val_frac * 10_000))
    pos = b["target"].mean()
    print(f"  split train={int((~is_val).sum())} val={int(is_val.sum())} pos_frac={pos:.4f}", flush=True)

    tr = ~is_val
    train = {k: torch.from_numpy(b[k][tr]) for k in ("features", "target", "skill_decile", "sample_weight")}
    val = {
        "feats": torch.from_numpy(b["features"][is_val]).to(dev),
        "target": torch.from_numpy(b["target"][is_val]).long().to(dev),
        "skill": torch.from_numpy(b["skill_decile"][is_val]).long().to(dev),
    }

    def train_arm(name, net):
        net = net.to(dev)
        npar = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        ntr = train["features"].shape[0]
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        for ep in range(epochs):
            order = rng.permutation(ntr)
            for i in range(0, ntr, bs):
                sel = order[i:i + bs]
                fe = train["features"][sel].to(dev, non_blocking=True)
                ta = train["target"][sel].long().to(dev, non_blocking=True)
                sk = train["skill_decile"][sel].long().to(dev, non_blocking=True)
                sw = train["sample_weight"][sel].float().to(dev, non_blocking=True)
                logits = net(fe, sk)
                loss = (F.cross_entropy(logits, ta, reduction="none") * sw).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            vnll, vacc, vauc = evaluate(net, val["feats"], val["target"], val["skill"])
            print(f"  [{name}] ep{ep}: val_NLL={vnll:.4f} val_acc={vacc:.4f} val_AUC={vauc:.4f}", flush=True)
        print(f"  [{name}] DONE params={npar/1e6:.2f}M {time.perf_counter()-t0:.0f}s "
              f"val_NLL={vnll:.4f} val_acc={vacc:.4f} val_AUC={vauc:.4f}", flush=True)
        return {"name": name, "params": npar, "nll": vnll, "acc": vacc, "auc": vauc}

    arms = [
        ("base_256x3", CallNetwork(fdim, hidden=256)),     # current arch
        ("wide_1024x3", CallNetwork(fdim, hidden=1024)),
        ("resid_512x4", ResidualCallNet(fdim, hidden=512, depth=4)),
    ]
    res = []
    for name, net in arms:
        torch.manual_seed(seed)
        res.append(train_arm(name, net))
    base = res[0]
    print(f"  --- {call_type} summary (AUC higher=better, NLL lower=better) ---")
    for r in res:
        d = "" if r is base else (f"  dAUC={r['auc']-base['auc']:+.4f} "
                                  f"dNLL={r['nll']-base['nll']:+.4f}")
        print(f"    {r['name']:<14} {r['params']/1e6:>6.2f}M  AUC={r['auc']:.4f} "
              f"NLL={r['nll']:.4f} acc={r['acc']:.4f}{d}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=8_000_000)
    p.add_argument("--val-frac", type=float, default=0.12)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--types", default="call_tichu,call_grand_tichu")
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} (DEBUG-cap7e2)", flush=True)
    for ct in args.types.split(","):
        run_type(ct, args.rows, args.val_frac, args.epochs, args.bs, args.lr, args.seed, dev)


if __name__ == "__main__":
    main()
