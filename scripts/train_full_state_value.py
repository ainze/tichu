r"""Train and score the Full-State Value net (test build).

Fits two nets on the same deal-level split of the emitted shards, with early
stopping on held-out deals: one on the full state (all four hands) and one on
the acting seat's observable features only. Reports, on test deals:

- R^2 and RMSE in points, overall and by phase (open / mid / late);
- the action-comparison read: for a decision where the round was also played
  with an alternative, how well does `V(after alt) - V(after chosen)` match the
  true difference? This is what a search leaf is used for.

  py -3.14 -m scripts.train_full_state_value `
      --shards C:\workbench\tichu\data\runs\full_state_value_v1\shards `
      --out-dir C:\workbench\tichu\data\runs\full_state_value_v1\fit
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

PHASES = ("open", "mid", "late")


def load_shards(shard_dir: Path, limit: int | None):
    files = sorted(shard_dir.glob("shard_*.npz"))
    if limit:
        files = files[:limit]
    cols = {k: [] for k in ("x", "z", "deal", "phase", "live")}
    pairs = []
    offset = 0
    for f in files:
        d = np.load(f)
        for k in cols:
            cols[k].append(d[k])
        p = d["pairs"].copy()
        for col in (0, 1):
            p[:, col] = np.where(p[:, col] >= 0, p[:, col] + offset, -1)
        pairs.append(p)
        offset += len(d["z"])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["pairs"] = np.concatenate(pairs)
    return out, len(files)


def r2(pred, y):
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((pred - y) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")


def predict(model, x, cols, device, batch=65536):
    import torch

    out = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch][:, :cols].astype(np.float32)).to(device)
            out[i:i + batch] = model(xb).float().cpu().numpy()
    return out


def fit(x, z, train_idx, val_idx, *, cols, hidden, depth, epochs, lr, batch, device,
        patience, tag):
    import torch
    import torch.nn.functional as F

    from tichu_training.awr.value_baseline import ValueBaseline

    torch.manual_seed(0)
    model = ValueBaseline(cols, hidden=hidden, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps_per_epoch = max(1, len(train_idx) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.05)
    rng = np.random.default_rng(0)
    x_val, z_val = x[val_idx], z[val_idx]
    best, best_state, bad = -np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        t0 = time.monotonic()
        for s in range(steps_per_epoch):
            idx = np.sort(perm[s * batch:(s + 1) * batch])
            xb = torch.from_numpy(x[idx][:, :cols].astype(np.float32)).to(device)
            zb = torch.from_numpy(z[idx]).to(device)
            loss = F.mse_loss(model(xb), zb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        val = r2(predict(model, x_val, cols, device), z_val)
        print(f"  [{tag}] epoch {epoch + 1}/{epochs} val R^2 {val:.4f} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
        if val > best + 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def pair_read(pred, pairs, test_deals):
    """Action-comparison metrics on test-deal pairs. `pred` is per-row, in the
    row's own acting-seat frame; signs put both successors in the decider's."""
    keep = np.isin(pairs[:, 7].astype(np.int64), test_deals)
    p = pairs[keep]
    base_row, alt_row = p[:, 0].astype(np.int64), p[:, 1].astype(np.int64)
    q_base = np.where(base_row >= 0, p[:, 4] * pred[np.maximum(base_row, 0)], p[:, 2])
    q_alt = np.where(alt_row >= 0, p[:, 5] * pred[np.maximum(alt_row, 0)], p[:, 3])
    pred_d = (q_alt - q_base) * 100.0
    true_d = (p[:, 3] - p[:, 2]) * 100.0
    out = {"n": int(len(p)), "true_delta_mean": float(true_d.mean()),
           "corr": float(np.corrcoef(pred_d, true_d)[0, 1])}
    big = np.abs(true_d) >= 15.0
    out["sign_agree_when_true_ge_15"] = float((np.sign(pred_d[big]) == np.sign(true_d[big])).mean())
    for bar in (0.0, 10.0, 20.0, 40.0):
        take = pred_d >= bar if bar > 0 else pred_d > 0
        out[f"switch_if_pred_ge_{int(bar)}"] = {
            "rate": float(take.mean()),
            "true_gain_per_switch": float(true_d[take].mean()) if take.any() else 0.0,
            "true_gain_per_decision": float(np.where(take, true_d, 0.0).mean()),
        }
    out["oracle_gain_per_decision"] = float(np.maximum(true_d, 0.0).mean())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--skip-observable", action="store_true")
    args = ap.parse_args(argv)

    import torch

    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    from tichu_training.perfect_info import PERFECT_INFO_DIM
    from tichu_training.search.full_state_value import save_full_state_value

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, n_files = load_shards(Path(args.shards), args.limit_shards)
    x, z, deal, phase = data["x"], data["z"], data["deal"], data["phase"]
    assert x.shape[1] == PERFECT_INFO_DIM
    bucket = deal % 20
    test, val = bucket == 0, bucket == 1
    train_idx = np.flatnonzero(~test & ~val)
    val_idx, test_idx = np.flatnonzero(val), np.flatnonzero(test)
    n_deals = len(np.unique(deal))
    print(f"{n_files} shards, {len(z):,} rows, {n_deals:,} deals | "
          f"train {len(train_idx):,} / val {len(val_idx):,} / test {len(test_idx):,} | "
          f"Var(z)={z.var():.3f} | device {device}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = {"full": PERFECT_INFO_DIM}
    if not args.skip_observable:
        arms["observable"] = FEATURIZER_OUTPUT_DIM
    report = {"rows": int(len(z)), "deals": int(n_deals), "arms": {}}
    test_deals = np.unique(deal[test_idx])
    for tag, cols in arms.items():
        model, best_val = fit(
            x, z, train_idx, val_idx, cols=cols, hidden=args.hidden, depth=args.depth,
            epochs=args.epochs, lr=args.lr, batch=args.batch, device=device,
            patience=args.patience, tag=tag)
        pred = predict(model, x, cols, device)
        arm = {"val_r2": best_val,
               "train_r2": r2(pred[train_idx[:500_000]], z[train_idx[:500_000]]),
               "test": {}}
        splits = {"all": np.ones(len(test_idx), dtype=bool)}
        for i, name in enumerate(PHASES):
            splits[name] = phase[test_idx] == i
        for name, m in splits.items():
            pt, zt = pred[test_idx][m], z[test_idx][m]
            arm["test"][name] = {
                "n": int(m.sum()), "r2": r2(pt, zt),
                "rmse_points": float(np.sqrt(((pt - zt) ** 2).mean()) * 100.0),
                "sd_points": float(zt.std() * 100.0),
            }
        arm["pairs"] = pair_read(pred, data["pairs"], test_deals)
        report["arms"][tag] = arm
        if tag == "full":
            save_full_state_value(model, out_dir / "full_state_value.pt",
                                  hidden=args.hidden, depth=args.depth,
                                  note=f"{n_deals} deals, val R^2 {best_val:.4f}")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{'arm':<11}{'split':<6}{'n':>9}{'R^2':>8}{'RMSE pts':>10}{'sd pts':>8}")
    for tag, arm in report["arms"].items():
        for name, m in arm["test"].items():
            print(f"{tag:<11}{name:<6}{m['n']:>9,}{m['r2']:>8.3f}"
                  f"{m['rmse_points']:>10.1f}{m['sd_points']:>8.1f}")
    for tag, arm in report["arms"].items():
        print(f"\n[{tag}] train R^2 {arm['train_r2']:.3f} | pairs: "
              f"{json.dumps(arm['pairs'], indent=1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
