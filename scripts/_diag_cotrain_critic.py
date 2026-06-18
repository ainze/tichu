"""Diagnostic: how well does the co-train Perfect-Info Critic actually fit?

`ppo_log.csv` only shows `value_loss` (MSE), which is scale-ambiguous: a flat
~21k could be the irreducible return-variance floor OR an under-fit critic sitting
at the no-skill baseline. This probe disambiguates by reporting **R^2** = 1 -
MSE / Var(returns), overall AND per decision type (play / schupfen / tichu / grand)
— the same read that gave the v5 "grand 0.10 / schupfen 0.25" under-fit finding.

It loads the LIVE policy + critic from the run's `_rollout_weights.pt` (so the
critic is measured against the current policy that produced the latest CSV rows,
not the warm-start BC), rolls out a HELD-OUT position pool, recomputes GAE returns
exactly as `build_cotrain_batch` does, and scores the frozen critic on every step.

    py scripts/_diag_cotrain_critic.py --config configs/cotrain_v6.yaml \
        --weights C:/workbench/tichu/data/runs/cotrain_v6/_rollout_weights.pt \
        --n 1024 --seed 9000000
"""

import argparse

import torch
import yaml

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.cotrain import BatchedCoTrainPolicy
from tichu_training.ppo.rollout import collect_rollout
from tichu_training.ppo.update import compute_gae


def _r2(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, float]:
    """Return (R^2, MSE, Var(target), n) for a 1-D pred/target pair."""
    n = target.numel()
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mse = torch.mean((pred - target) ** 2).item()
    var = torch.var(target, unbiased=False).item()
    r2 = 1.0 - mse / var if var > 0 else float("nan")
    return r2, mse, var, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True, help="run's _rollout_weights.pt (live policy+critic)")
    ap.add_argument("--n", type=int, default=1024, help="held-out positions to roll out")
    ap.add_argument("--seed", type=int, default=9_000_000, help="held-out pool seed (disjoint from training)")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    skill_decile = int(config["ppo"].get("skill_decile", 9))
    learner_team = int(config["ppo"].get("learner_team", 0))
    hidden = int(config["critic"]["hidden"])
    depth = int(config["critic"].get("depth", 1))
    gamma = float(config["ppo"].get("gamma", 1.0))
    lam = float(config["ppo"].get("lam", 0.95))

    blob = torch.load(args.weights, map_location="cpu", weights_only=False)
    models = _build_models(config)
    for dt in _NET_TYPES:
        models[dt].load_state_dict(blob["models"][dt])
        models[dt].eval()
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=hidden, depth=depth)
    critic.load_state_dict(blob["critic"])
    critic.eval()

    print(f"critic: PERFECT_INFO_DIM={PERFECT_INFO_DIM}, hidden={hidden}, depth={depth}")
    print(f"rolling out {args.n} held-out positions (seed {args.seed}, decile {skill_decile})...", flush=True)

    policy = BatchedCoTrainPolicy(
        models["play"], models["schupfen"], models["tichu"], models["grand"],
        critic, skill_decile=skill_decile, perfect_info=True,
    )
    positions = generate_full_position_pool(seed=args.seed, n=args.n)
    trajs = collect_rollout(positions, policy, learner_team=learner_team)

    # Recompute GAE returns exactly as build_cotrain_batch, tagging each step with
    # its decision_type so we can split R^2 by net.
    feats: list[torch.Tensor] = []
    rets: list[float] = []
    types: list[str] = []
    for traj in trajs:
        steps = traj.steps
        if not steps:
            continue
        values = torch.tensor([s.value for s in steps], dtype=torch.float32)
        rewards = torch.zeros(len(steps), dtype=torch.float32)
        rewards[-1] = float(traj.reward)
        _adv, ret = compute_gae(rewards, values, gamma=gamma, lam=lam, last_value=0.0)
        for i, s in enumerate(steps):
            cf = s.critic_features if s.critic_features is not None else s.features
            feats.append(torch.as_tensor(cf, dtype=torch.float32))
            rets.append(float(ret[i]))
            types.append(s.decision_type)

    X = torch.stack(feats)
    y = torch.tensor(rets, dtype=torch.float32)
    with torch.no_grad():
        pred = critic(X).reshape(-1)

    print(f"\n{len(y):,} decision-steps over {len(trajs)} trajectories\n")
    r2, mse, var, n = _r2(pred, y)
    print(f"{'ALL':<10} n={n:>7,}  R2={r2:+.3f}  MSE={mse:>10,.0f}  Var(ret)={var:>10,.0f}  RMSE={mse**0.5:>6.1f}")
    print(f"{'-'*70}")
    for dt in ("play", "schupfen", "tichu", "grand"):
        mask = torch.tensor([t == dt for t in types])
        if mask.any():
            r2, mse, var, n = _r2(pred[mask], y[mask])
            print(f"{dt:<10} n={n:>7,}  R2={r2:+.3f}  MSE={mse:>10,.0f}  Var(ret)={var:>10,.0f}  RMSE={mse**0.5:>6.1f}")
    print(f"{'-'*70}")
    print("R2 ~ 0  -> critic at the no-skill floor (predicts the mean); the value_loss")
    print("           plateau IS under-fit, not irreducible variance.")
    print("R2 high -> value_loss plateau is the genuine outcome-variance floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
