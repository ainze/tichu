r"""Does H2 (current_trick_winner) target the partner-overtake blunder?

The pathology: the agent overtakes/bombs a partner who is currently winning the
trick. v6 can only see "partner is winning" when partner *led* the trick
(`trick_leader[partner]`); when partner *overtook* to become the winner, v6 is
blind — and that is exactly what H2 adds. So the decisive question is whether the
blunder-relevant signal lives in the partner-OVERTOOK rows (v6 blind → H2 fixes
it) or the partner-LED rows (v6 already sees → feature won't help; reward problem).

Reads the pre-check cache, trains baseline (v6) vs v6+H2, and on the partner-
winning TEST rows — split by led vs overtook — reports the human grab-rate and
each model's mean predicted P(grab). If on partner-overtook rows humans mostly
cede but baseline over-predicts grab while v6+H2 pulls P(grab) down toward the
human rate, H2 targets the blunder.

Usage:
  $env:PYTHONPATH = "<worktree>\src"
  python scripts/analyze_partner_overtake.py --cache C:\workbench\tichu\data\precheck_cache.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from precheck_trick_stakes import by_game_split3, load_cache
from tichu_training.featurizer import SECTION_OFFSETS


def _train_predict(Xtr, ytr, Xva, yva, Xte, hidden, epochs, seed, patience=4):
    """Early-stopped MLP; returns P(grab) on the test rows."""
    import torch

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = torch.nn.BCEWithLogitsLoss()
    xtr = torch.as_tensor(Xtr); ytt = torch.as_tensor(ytr.astype(np.float32))
    xva = torch.as_tensor(Xva); yvt = torch.as_tensor(yva.astype(np.float32))
    xte = torch.as_tensor(Xte)
    n = xtr.shape[0]; bs = 4096
    best = float("inf"); best_state = None; bad = 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(net(xtr[idx]).squeeze(1), ytt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(xva).squeeze(1), yvt))
        if v < best - 1e-5:
            best = v; best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(xte).squeeze(1)).clamp(1e-6, 1 - 1e-6).numpy()


def _slice_report(name, mask, y, p_base, p_h2):
    n = int(mask.sum())
    if n == 0:
        print(f"  {name:28s} N=0"); return
    yr = float(y[mask].mean())
    pb = float(p_base[mask].mean())
    ph = float(p_h2[mask].mean())
    # NLL per slice
    def nll(p):
        yy = y[mask].astype(np.float64)
        return float(-(yy * np.log(p[mask]) + (1 - yy) * np.log(1 - p[mask])).mean())
    print(f"  {name:28s} N={n:>6}  human_grab={yr:.3f}  "
          f"pred_grab base={pb:.3f} +H2={ph:.3f}  "
          f"NLL base={nll(p_base):.4f} +H2={nll(p_h2):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load_cache(args.cache)
    off = SECTION_OFFSETS["trick_leader"]
    partner_led = d.X[:, off + 2] > 0.5          # trick_leader relseat 2 = partner
    partner_winning = d.S[:, 3] > 0.5            # stakes winner one-hot, partner slot
    Xh2 = np.concatenate([d.X, d.S[:, 1:]], axis=1)  # v6 + H2 (winner one-hot only)

    acc_base = np.zeros(len(d.y)); acc_h2 = np.zeros(len(d.y)); seen = np.zeros(len(d.y), bool)
    for s in range(args.n_seeds):
        tr, va, te = by_game_split3(d.games, 0.1, 0.2, args.seed + s)
        pb = _train_predict(d.X[tr], d.y[tr], d.X[va], d.y[va], d.X[te],
                            args.hidden, args.epochs, args.seed + s)
        ph = _train_predict(Xh2[tr], d.y[tr], Xh2[va], d.y[va], Xh2[te],
                            args.hidden, args.epochs, args.seed + s)
        acc_base[te] += pb; acc_h2[te] += ph; seen[te] = True
    # A row can be a test row in multiple seeds; average over the times it was.
    counts = np.zeros(len(d.y))
    for s in range(args.n_seeds):
        _, _, te = by_game_split3(d.games, 0.1, 0.2, args.seed + s)
        counts[te] += 1
    ok = counts > 0
    p_base = np.where(ok, acc_base / np.maximum(counts, 1), 0.5)
    p_h2 = np.where(ok, acc_h2 / np.maximum(counts, 1), 0.5)

    y = d.y
    print(f"N_total={len(y)}  partner_winning={int(partner_winning.sum())}  "
          f"(led={int((partner_winning & partner_led).sum())}, "
          f"overtook={int((partner_winning & ~partner_led).sum())})")
    print("mean predicted P(grab) — if H2 targets the blunder, it drops toward the")
    print("human grab-rate on the partner-OVERTOOK rows (where v6 is blind):")
    m = ok
    _slice_report("all contested", m, y, p_base, p_h2)
    _slice_report("partner winning (all)", m & partner_winning, y, p_base, p_h2)
    _slice_report("  partner LED (v6 sees)", m & partner_winning & partner_led, y, p_base, p_h2)
    _slice_report("  partner OVERTOOK (blind)", m & partner_winning & ~partner_led, y, p_base, p_h2)


if __name__ == "__main__":
    main()
