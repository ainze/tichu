"""One-off profiler: where does a co-training iteration spend its time?

Runs ONE real iteration (collect_rollout -> build_cotrain_batch -> cotrain_update)
on the warm-started master nets at the config's M, under cProfile, and prints a
coarse phase split plus the top functions by self-time. The question it answers:
is the iteration dominated by NN forwards (GPU would help) or by the per-game
engine/featurize Python (GPU would not)?  See ADR-0034.

    py scripts/profile_cotrain.py --config configs/cotrain_v5.yaml [--m 512]
"""

import argparse
import cProfile
import copy
import io
import pstats
import time

import torch
import yaml

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.training import load_checkpoint
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.cotrain import (
    BatchedCoTrainPolicy,
    build_cotrain_batch,
    cotrain_update,
)
from tichu_training.ppo.rollout import collect_rollout

# Functions whose self-time we bucket as "NN forward" vs "engine/featurize Python".
_NN_HINTS = ("linear", "addmm", "matmul", "gelu", "log_softmax", "softmax",
             "multinomial", "embedding", "_call_impl", "forward")
_ENGINE_HINTS = ("featurize", "legal_actions", "engine.py", "legality.py",
                 "perfect_info", "card_slots", "step", "private_view", "combinations")


def _bucket(stats: pstats.Stats) -> None:
    nn_t = eng_t = other_t = 0.0
    for (fname, _line, func), value in stats.stats.items():
        tottime = value[2]
        key = f"{fname}:{func}".lower()
        if any(h in key for h in _NN_HINTS):
            nn_t += tottime
        elif any(h in key for h in _ENGINE_HINTS):
            eng_t += tottime
        else:
            other_t += tottime
    total = nn_t + eng_t + other_t
    print("\n=== bucketed self-time (cProfile tottime) ===")
    print(f"  NN forwards (GPU-able):     {nn_t:7.2f}s  {100 * nn_t / total:5.1f}%")
    print(f"  engine/featurize (CPU-only):{eng_t:7.2f}s  {100 * eng_t / total:5.1f}%")
    print(f"  other:                      {other_t:7.2f}s  {100 * other_t / total:5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--m", type=int, default=None, help="override positions_per_iter")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    m = args.m or int(config["ppo"]["positions_per_iter"])
    torch.manual_seed(0)

    print(f"profiling 1 iteration at M={m} (torch threads={torch.get_num_threads()}, "
          f"cuda={torch.cuda.is_available()})", flush=True)

    models = _build_models(config)
    for dt in _NET_TYPES:
        load_checkpoint(config["warm_start"][dt], models[dt])
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=int(config["critic"]["hidden"]))
    policy = BatchedCoTrainPolicy(
        models["play"], models["schupfen"], models["tichu"], models["grand"],
        critic, skill_decile=9, perfect_info=True,
    )
    bc_models = {dt: copy.deepcopy(models[dt]) for dt in _NET_TYPES}
    optimizer = torch.optim.Adam(
        [p for dt in _NET_TYPES for p in models[dt].parameters()] + list(critic.parameters()),
        lr=1e-4,
    )
    positions = generate_full_position_pool(seed=0, n=m)

    # Coarse phase split (wall clock), then a profiled rollout for the bucket/topN.
    t0 = time.perf_counter()
    trajs = collect_rollout(positions, policy, learner_team=0)
    t1 = time.perf_counter()
    batch = build_cotrain_batch(trajs, skill_decile=9, gamma=1.0, lam=0.95)
    t2 = time.perf_counter()
    cotrain_update(
        models, bc_models, critic, optimizer, batch,
        clip_eps=0.1, vf_coef=0.5,
        ent_coefs={dt: 0.01 for dt in _NET_TYPES},
        kl_coefs={dt: 1.0 for dt in _NET_TYPES}, epochs=3,
    )
    t3 = time.perf_counter()
    print("\n=== phase wall-clock ===")
    print(f"  rollout (collect):     {t1 - t0:6.2f}s")
    print(f"  build_batch:           {t2 - t1:6.2f}s")
    print(f"  update (3 epochs, CPU):{t3 - t2:6.2f}s")
    print(f"  TOTAL (CPU):           {t3 - t0:6.2f}s")

    # Same update on CUDA (includes the per-call net<->GPU transfer the run pays).
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        g0 = time.perf_counter()
        cotrain_update(
            models, bc_models, critic, optimizer, batch,
            clip_eps=0.1, vf_coef=0.5,
            ent_coefs={dt: 0.01 for dt in _NET_TYPES},
            kl_coefs={dt: 1.0 for dt in _NET_TYPES}, epochs=3, device="cuda",
        )
        torch.cuda.synchronize()
        g1 = time.perf_counter()
        print(f"  update (3 epochs, CUDA):{g1 - g0:6.2f}s  "
              f"(speedup {(t3 - t2) / (g1 - g0):.1f}x; iter would be "
              f"~{(t1 - t0) + (t2 - t1) + (g1 - g0):.1f}s vs {t3 - t0:.1f}s)")

    pr = cProfile.Profile()
    pr.enable()
    collect_rollout(positions, policy, learner_team=0)
    pr.disable()
    stats = pstats.Stats(pr)
    _bucket(stats)
    print("\n=== top 18 functions by self-time (in rollout) ===")
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(18)
    print("\n".join(buf.getvalue().splitlines()[5:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
