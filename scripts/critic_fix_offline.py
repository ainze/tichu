"""Lever-2 offline critic experiment: can a differently-trained critic remove the
call-hinge / late-round epistemic error the advantage-SNR probe measured?

(2026-07-24 note: the shared Perfect-Info Critic's residual is 54-77% FIXABLE
error in mid/late-round states — heteroscedastic MSE piles the gradient onto the
high-variance early states and leaves ±100-300 misses on near-deterministic
call-outcome states. This script fits candidate critics offline on frozen-policy
rollouts and scores them on the standing benchmark, BEFORE any RL time.)

Arms:
  baseline  same arch + plain MSE on the terminal outcome R (replication control)
  anorm     aleatoric-normalized MSE: per-step weight 1/sigma^2(hand-size bucket),
            sigma from the probe's measured per-phase noise floors — reallocates
            gradient mass to the low-noise late states
  hetero    heteroscedastic Gaussian NLL: net predicts (mu, log sigma^2) and
            learns the noise scale per state; scored on mu

Stages (run in order; each writes under --out-dir):
  collect   frozen-weights rollouts ({champion,bc} alternation like training) ->
            train_rows.npz (critic_features f16, R, decision_type, hand_size)
  fit       train the arms with a held-out-games val split -> <arm>.pt + fit.json
  score     every arm + the INCUMBENT critic vs the benchmark parquet(s):
            per-phase epistemic share (the probe metric), MSE vs E[R|s]

    py scripts/critic_fix_offline.py collect --config configs/cotrain_v6_pbrs_resid_wish_gated_wishfix.yaml \
        --weights C:/workbench/tichu/data/runs/critic_fix_v1/frozen_weights.pt \
        --run-dir C:/workbench/tichu/data/runs/cotrain_v6_pbrs_resid_wish_gated_wishfix \
        --out-dir C:/workbench/tichu/data/runs/critic_fix_v1 --n 3072
    py scripts/critic_fix_offline.py fit --out-dir ... [--arms baseline,anorm,hetero]
    py scripts/critic_fix_offline.py score --out-dir ... --config ... --weights ... \
        --benchmarks C:/workbench/tichu/data/runs/advantage_snr_probe_v1c/stage_b_states.parquet,...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.cotrain import BatchedCoTrainPolicy
from tichu_training.ppo.rollout import collect_rollout

# Aleatoric noise floors (sd of R | state) measured by the advantage-SNR probe,
# keyed by the actor's hand size. Round-start-ish states get the stage-A read.
_SIGMA_BY_BUCKET = {"start": 170.0, "open": 121.0, "mid": 74.0, "late": 44.0}


def _bucket(hand: int) -> str:
    if hand >= 14:
        return "start"
    return "open" if hand >= 11 else ("mid" if hand >= 6 else "late")


def _phase(hand: int) -> str:  # benchmark parquet convention (probe stage B)
    return "open" if hand >= 11 else ("mid" if hand >= 6 else "late")


def _load_policy(config, weights_path, *, generator=None):
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    models = _build_models(config)
    for dt in _NET_TYPES:
        models[dt].load_state_dict(blob["models"][dt])
        models[dt].eval()
    critic = ValueBaseline(
        PERFECT_INFO_DIM, hidden=int(config["critic"]["hidden"]),
        depth=int(config["critic"].get("depth", 1)))
    critic.load_state_dict(blob["critic"])
    critic.eval()
    policy = BatchedCoTrainPolicy(
        models["play"], models["schupfen"], models["tichu"], models["grand"], critic,
        skill_decile=int(config["ppo"].get("skill_decile", 9)),
        perfect_info=bool(config.get("perfect_info", True)),
        generator=generator, train_wish=bool(config.get("cotrain_wish", False)),
    )
    return policy, critic


def collect(args) -> None:
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    gen = torch.Generator().manual_seed(args.seed)
    learner, _ = _load_policy(config, args.weights, generator=gen)
    run_dir = Path(args.run_dir)
    opponents = [
        ("champion", _load_policy(config, str(run_dir / "_champion.pt"), generator=gen)[0]),
        ("bc", _load_policy(config, str(run_dir / "_bc_opponent.pt"), generator=gen)[0]),
    ]
    positions = generate_full_position_pool(seed=args.pool_seed, n=args.n)

    X, y, dts, game_ids = [], [], [], []
    chunk = 512
    for at in range(0, args.n, chunk):
        batch = positions[at : at + chunk]
        opp = opponents[(at // chunk) % 2][1]  # alternate like training's per-iter cycle
        trajs = collect_rollout(batch, learner, opponent_policy=opp, learner_team=0)
        for g in range(len(batch)):
            for traj in (trajs[2 * g], trajs[2 * g + 1]):
                for s in traj.steps:
                    if s.critic_features is None:
                        continue
                    X.append(np.asarray(s.critic_features, dtype=np.float16))
                    y.append(float(traj.reward))
                    dts.append(s.decision_type)
                    game_ids.append(at + g)
        print(f"collect: {min(at + chunk, args.n)}/{args.n} games, {len(y)} rows", flush=True)

    X = np.stack(X)
    hand = X[:, :56].astype(np.float32).round().sum(axis=1).astype(np.int8)  # own_hand bits
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "train_rows.npz", X=X, y=np.asarray(y, dtype=np.float32),
        decision_type=np.asarray(dts), hand=hand,
        game_id=np.asarray(game_ids, dtype=np.int32),
    )
    print(f"saved {len(y)} rows -> {out / 'train_rows.npz'}")


class _HeteroCritic(torch.nn.Module):
    """A full ValueBaseline for mu (warm-startable from the incumbent state dict)
    + an independent log-sigma^2 head; scored on mu."""

    def __init__(self, in_dim: int, hidden: int, depth: int):
        super().__init__()
        self.vb = ValueBaseline(in_dim, hidden=hidden, depth=depth)
        self.logvar_net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))

    def forward(self, x):
        return self.vb(x).reshape(-1), self.logvar_net(x).reshape(-1).clamp(4.6, 12.0)


def fit(args) -> None:
    out = Path(args.out_dir)
    blob = np.load(out / "train_rows.npz", allow_pickle=True)
    X = torch.from_numpy(blob["X"].astype(np.float32))
    y = torch.from_numpy(blob["y"])
    hand = blob["hand"]
    game = blob["game_id"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # held-out GAMES (not rows) for validation
    rng = np.random.default_rng(0)
    games = np.unique(game)
    val_games = set(rng.choice(games, size=max(1, len(games) // 10), replace=False).tolist())
    val_mask = np.isin(game, list(val_games))
    tr_idx = np.flatnonzero(~val_mask)
    va_idx = np.flatnonzero(val_mask)
    dts = blob["decision_type"]
    # grand is decided on the synthetic 8-card deal-time state — it is a round-START
    # decision (sigma ~ stage A), not a mid-round one; bucket by type, not hand size.
    sigma = np.asarray(
        [_SIGMA_BY_BUCKET["start" if dt == "grand" else _bucket(int(h))]
         for dt, h in zip(dts, hand)], dtype=np.float32)
    w = (1.0 / sigma**2)
    w = torch.from_numpy((w / w[tr_idx].mean()).astype(np.float32))  # mean-1 on train
    late_va = va_idx[hand[va_idx] <= 5]
    print(f"rows: {len(y)} (train {len(tr_idx)}, val {len(va_idx)}, val-late {len(late_va)}), device {dev}")

    hidden, depth = args.hidden, args.depth
    warm_state = None
    if args.warm_start:
        # Fine-tune the INCUMBENT critic under the reallocated loss instead of
        # training from scratch: the incumbent has ~1000x this dataset's worth of
        # step-presentations behind it, so a scratch fit can't reach its level and
        # the arms would only measure under-training. The question is whether the
        # LOSS reallocation fixes the late-round share on top of the incumbent fit.
        warm_state = torch.load(args.warm_start, map_location="cpu", weights_only=False)["critic"]
    results = {}
    for arm in args.arms.split(","):
        torch.manual_seed(0)
        if arm == "hetero":
            net = _HeteroCritic(X.shape[1], hidden, depth).to(dev)
        else:
            net = ValueBaseline(X.shape[1], hidden=hidden, depth=depth).to(dev)
        if warm_state is not None:
            (net.vb if arm == "hetero" else net).load_state_dict(warm_state)
        net = net.to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            order = torch.from_numpy(rng.permutation(tr_idx).copy())
            net.train()
            for b in range(0, len(order), args.batch):
                idx = order[b : b + args.batch]
                xb, yb = X[idx].to(dev), y[idx].to(dev)
                if arm == "hetero":
                    mu, logvar = net(xb)
                    loss = 0.5 * (logvar + (yb - mu) ** 2 / logvar.exp()).mean()
                elif arm == "anorm":
                    pred = net(xb).reshape(-1)
                    loss = (w[idx].to(dev) * (pred - yb) ** 2).mean()
                else:
                    pred = net(xb).reshape(-1)
                    loss = ((pred - yb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                def _mse(ix):
                    preds = []
                    for b in range(0, len(ix), 65536):
                        xb = X[torch.from_numpy(ix[b:b+65536].copy())].to(dev)
                        p = net(xb)[0] if arm == "hetero" else net(xb).reshape(-1)
                        preds.append(p.cpu())
                    p = torch.cat(preds)
                    return float(((p - y[torch.from_numpy(ix.copy())]) ** 2).mean())
                v_all, v_late = _mse(va_idx), _mse(late_va)
            print(f"  {arm} epoch {epoch}: val MSE {v_all:,.0f}  val-late MSE {v_late:,.0f}", flush=True)
        torch.save(net.state_dict(), out / f"critic_{arm}.pt")
        results[arm] = {"val_mse": v_all, "val_late_mse": v_late,
                        "arch": {"hidden": hidden, "depth": depth}}
    with open(out / "fit.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


def _epistemic_table(pred: np.ndarray, bench: pd.DataFrame) -> dict:
    rows = {}
    bench = bench.assign(pred=pred)
    for name, sub in [(p, bench[bench.phase == p]) for p in ("open", "mid", "late")] + [
        ("caller_live", bench[bench.caller_live]), ("all", bench)]:
        if not len(sub):
            continue
        epi = ((sub.pred - sub.r_mean) ** 2 - sub.r_var / sub.k).mean()
        ale = sub.r_var.mean()
        rows[name] = {"n": int(len(sub)), "epistemic_var": float(epi),
                      "epistemic_share": float(epi / (epi + ale)),
                      "rmse_vs_ER": float(np.sqrt(((sub.pred - sub.r_mean) ** 2).mean()))}
    return rows


def score(args) -> None:
    out = Path(args.out_dir)
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fitcfg = json.load(open(out / "fit.json", encoding="utf-8"))

    candidates = {}
    for arm, meta in fitcfg.items():
        h, d = meta["arch"]["hidden"], meta["arch"]["depth"]
        net = (_HeteroCritic(PERFECT_INFO_DIM, h, d) if arm == "hetero"
               else ValueBaseline(PERFECT_INFO_DIM, hidden=h, depth=d))
        net.load_state_dict(torch.load(out / f"critic_{arm}.pt", map_location="cpu"))
        net.eval()
        candidates[arm] = net
    # incumbent = the frozen live critic
    blob = torch.load(args.weights, map_location="cpu", weights_only=False)
    inc = ValueBaseline(PERFECT_INFO_DIM, hidden=int(config["critic"]["hidden"]),
                        depth=int(config["critic"].get("depth", 1)))
    inc.load_state_dict(blob["critic"]); inc.eval()
    candidates = {"incumbent": inc, **candidates}

    for bench_path in args.benchmarks.split(","):
        bench = pd.read_parquet(bench_path)
        Xb = torch.from_numpy(np.stack(bench.critic_features.to_numpy()).astype(np.float32))
        print(f"\n=== benchmark: {bench_path} ({len(bench)} states) ===")
        summary = {}
        for name, net in candidates.items():
            net = net.to(dev)
            with torch.no_grad():
                p = net(Xb.to(dev))
                pred = (p[0] if isinstance(p, tuple) else p.reshape(-1)).cpu().numpy()
            net.cpu()
            summary[name] = _epistemic_table(pred, bench)
            r = summary[name]
            print("%-10s " % name + "  ".join(
                "%s: share %.2f rmse %5.1f" % (k, r[k]["epistemic_share"], r[k]["rmse_vs_ER"])
                for k in ("open", "mid", "late", "caller_live") if k in r))
        with open(out / f"score_{Path(bench_path).parent.name}.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--config", required=True); c.add_argument("--weights", required=True)
    c.add_argument("--run-dir", required=True); c.add_argument("--out-dir", required=True)
    c.add_argument("--n", type=int, default=3072)
    c.add_argument("--pool-seed", type=int, default=200_000_000)
    c.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fit")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--warm-start", default=None,
                   help="rollout-weights .pt whose ['critic'] seeds every arm (fine-tune mode)")
    f.add_argument("--arms", default="baseline,anorm,hetero")
    f.add_argument("--epochs", type=int, default=4)
    f.add_argument("--batch", type=int, default=8192)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--hidden", type=int, default=1024)
    f.add_argument("--depth", type=int, default=3)
    s = sub.add_parser("score")
    s.add_argument("--out-dir", required=True); s.add_argument("--config", required=True)
    s.add_argument("--weights", required=True)
    s.add_argument("--benchmarks", required=True, help="comma-separated stage_b_states.parquet paths")
    args = ap.parse_args()
    {"collect": collect, "fit": fit, "score": score}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
