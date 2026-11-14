r"""Train the Full-State Value net on same-state move comparisons (test build).

Each training pair is the state reached `ply` decisions after an alternative
move and the state reached `ply` decisions after the chosen move, from the same
decision in the same deal, both with exact outcomes. The loss is the usual MSE
on each state's outcome plus an MSE on the *difference* between the pair —
which is the quantity a search leaf is used for.

Reports, on held-out deals and per ply, how well the predicted difference
matches the true one and what switching on the net's say would have gained.

  py -3.14 -m scripts.train_value_siblings `
      --shards C:\workbench\tichu\data\runs\value_siblings_v1\shards `
      --init C:\workbench\tichu\data\runs\full_state_value_v1\fit_400\full_state_value.pt `
      --out-dir C:\workbench\tichu\data\runs\value_siblings_v1\fit
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

KEYS = ("x", "z", "dec", "cand", "ply", "sign", "deal")


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


def build_pairs(dec, cand, ply):
    """(alt_row, base_row) for every alternative row whose decision also has a
    chosen-move row at the same ply."""
    key = dec * 8 + ply
    base = np.flatnonzero(cand == 0)
    order = base[np.argsort(key[base])]
    base_keys = key[order]
    alt = np.flatnonzero(cand > 0)
    pos = np.searchsorted(base_keys, key[alt])
    ok = (pos < len(base_keys)) & (base_keys[np.minimum(pos, len(base_keys) - 1)] == key[alt])
    return alt[ok], order[pos[ok]]


def predict(model, x, device, batch=65536):
    import torch

    out = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.float32)).to(device)
            out[i:i + batch] = model(xb).float().cpu().numpy()
    return out


def pair_read(pred, z, sign, ply, alt, base):
    """Per ply: predicted vs true difference (points, decider's frame)."""
    pred_d = (sign[alt] * pred[alt] - sign[base] * pred[base]) * 100.0
    true_d = (sign[alt] * z[alt] - sign[base] * z[base]) * 100.0
    out = {}
    for j in sorted(set(ply[alt].tolist())) + ["all"]:
        m = np.ones(len(alt), dtype=bool) if j == "all" else ply[alt] == j
        p, t = pred_d[m], true_d[m]
        row = {"n": int(m.sum()), "true_delta_mean": float(t.mean()),
               "corr": float(np.corrcoef(p, t)[0, 1]),
               "diff_r2": 1.0 - float(((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())}
        for bar in (10.0, 20.0, 40.0):
            take = p >= bar
            row[f"switch_ge_{int(bar)}"] = {
                "rate": float(take.mean()),
                "true_gain_per_switch": float(t[take].mean()) if take.any() else 0.0}
        row["oracle_gain"] = float(np.maximum(t, 0.0).mean())
        out[str(j)] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--abs-weight", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    from tichu_training.awr.value_baseline import ValueBaseline
    from tichu_training.search.full_state_value import save_full_state_value

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, n_files = load(Path(args.shards), args.limit_shards)
    x, z = data["x"], data["z"]
    sign = data["sign"].astype(np.float32)
    ply, deal = data["ply"], data["deal"]
    alt, base = build_pairs(data["dec"], data["cand"], ply)
    bucket = deal[alt] % 20
    splits = {"test": bucket == 0, "val": bucket == 1, "train": bucket > 1}
    tr_alt, tr_base = alt[splits["train"]], base[splits["train"]]
    va_alt, va_base = alt[splits["val"]], base[splits["val"]]
    te_alt, te_base = alt[splits["test"]], base[splits["test"]]
    print(f"{n_files} shards, {len(z):,} rows, {len(np.unique(deal)):,} deals, "
          f"{len(alt):,} pairs (train {len(tr_alt):,}) | device {device}", flush=True)

    model = ValueBaseline(x.shape[1], hidden=args.hidden, depth=args.depth)
    if args.init:
        blob = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
    model.to(device)

    report = {"rows": int(len(z)), "pairs": int(len(alt)), "init": args.init}
    if args.init:
        report["before"] = pair_read(predict(model, x, device), z, sign, ply, te_alt, te_base)
        print(f"before: corr {report['before']['all']['corr']:.3f}", flush=True)

    def tens(idx):
        return torch.from_numpy(x[idx].astype(np.float32)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = max(1, len(tr_alt) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    best, best_state, bad = -np.inf, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(tr_alt))
        t0 = time.monotonic()
        for s in range(steps):
            sel = perm[s * args.batch:(s + 1) * args.batch]
            ia, ib = tr_alt[sel], tr_base[sel]
            va, vb = model(tens(ia)), model(tens(ib))
            za = torch.from_numpy(z[ia]).to(device)
            zb = torch.from_numpy(z[ib]).to(device)
            sa = torch.from_numpy(sign[ia]).to(device)
            sb = torch.from_numpy(sign[ib]).to(device)
            loss_abs = 0.5 * (F.mse_loss(va, za) + F.mse_loss(vb, zb))
            loss_pair = F.mse_loss(sa * va - sb * vb, sa * za - sb * zb)
            loss = args.abs_weight * loss_abs + args.pair_weight * loss_pair
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        rows = np.unique(np.concatenate([va_alt, va_base]))
        pred = np.zeros(len(z), dtype=np.float32)
        pred[rows] = predict(model, x[rows], device)
        val = pair_read(pred, z, sign, ply, va_alt, va_base)["all"]
        print(f"  epoch {epoch + 1}/{args.epochs} val diff R^2 {val['diff_r2']:.4f} "
              f"corr {val['corr']:.3f} ({time.monotonic() - t0:.0f}s)", flush=True)
        if val["diff_r2"] > best + 1e-4:
            best, bad = val["diff_r2"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    pred = predict(model, x, device)
    report["after"] = pair_read(pred, z, sign, ply, te_alt, te_base)
    test_rows = np.flatnonzero(deal % 20 == 0)
    zt, pt = z[test_rows], pred[test_rows]
    report["abs_r2_test"] = 1.0 - float(((pt - zt) ** 2).sum() / ((zt - zt.mean()) ** 2).sum())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_full_state_value(model, out_dir / "full_state_value.pt", hidden=args.hidden,
                          depth=args.depth, note=f"siblings, val diff R^2 {best:.4f}")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nabs R^2 on test rows {report['abs_r2_test']:.3f}")
    print(f"{'ply':<5}{'n':>9}{'corr':>7}{'diffR2':>8}{'sw10 rate':>10}{'gain/sw':>9}"
          f"{'sw20 rate':>10}{'gain/sw':>9}{'sw40 rate':>10}{'gain/sw':>9}")
    for j, r in report["after"].items():
        print(f"{j:<5}{r['n']:>9,}{r['corr']:>7.3f}{r['diff_r2']:>8.3f}"
              + "".join(f"{r[f'switch_ge_{b}']['rate']:>10.3f}"
                        f"{r[f'switch_ge_{b}']['true_gain_per_switch']:>9.1f}"
                        for b in (10, 20, 40)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
