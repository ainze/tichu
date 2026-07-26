"""Vine v3 autopsy round 3 — chimera head-swap gate localization (2026-07-26).

Rounds 1-2 refuted the KL-support hole and the continuation-flip mechanism,
and established that decision-level replay cannot resolve the gate's
−2.2/round (variance wall). This round uses the GATE ITSELF — the only
instrument with resolution — to localize which nets carry the deficit.

Three seat-swapped greedy mini-tournaments vs the frozen champion, on the
SAME deal set (Position-ordered margins => paired cross-arm contrasts):

  learner : all four learner nets            — replication anchor (~−2.2)
  chimA   : learner TRUNK net + champion standalones — trunk carries it?
  chimB   : champion trunk + learner STANDALONES     — schupfen/calls carry it?

Honest scope note: the "play" checkpoint is the shared-trunk BCModel, which
also carries the wish and dragon heads — so chimA isolates the whole trunk
net (play + wish + dragon; play trained by vine, wish by GAE), not the play
head alone. chimB isolates the standalone nets (schupfen / tichu / grand).

Readout: chimA ≈ learner & chimB ≈ 0 → the trunk (vine territory) carries
the −2. Reversed → vine exonerated, the standalone-head recipe is the
suspect. Both partial → interaction. learner − chimA − chimB = interaction
term (additivity check). A champion-vs-champion control is intentionally
absent: deterministic seat-swap of identical agents is identically zero.

    $env:PYTHONPATH = "<worktree>\\src"
    py scripts/diag_vine_v3_chimera_gate.py --deals 16384 --workers 10
"""

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.tournament import collect_pair_deltas
from tichu_training.cli.check_cotrain import _build_agent
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models
from tichu_training.ppo.greedy_gate import export_live_models

RUN = Path(r"C:/workbench/tichu/data/runs/cotrain_v6_vine_v3_gated")


def _models_from(config, state_dicts_by_net):
    models = _build_models(config)
    for dt in _NET_TYPES:
        models[dt].load_state_dict(state_dicts_by_net[dt])
        models[dt].eval()
    return models


def _boot_ci(x, iters=10_000, seed=0):
    arr = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, arr.size, size=(iters, arr.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def _fmt(name, x):
    m, lo, hi = _boot_ci(x)
    return f"{name:>18}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  (n={len(x)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cotrain_v6_vine_v3_gated.yaml")
    ap.add_argument("--deals", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=170_000_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(RUN / "diag_chimera_gate.npz"))
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    skill_decile = int(config["ppo"].get("skill_decile", 9))

    payload = torch.load(RUN / "train_state.bin", map_location="cpu", weights_only=False)
    champ = torch.load(RUN / "_champion.pt", map_location="cpu", weights_only=False)
    print(f"bundle iteration: {payload['iteration']}", flush=True)
    L, C = payload["models"], champ["models"]

    arms = {
        "learner": {dt: L[dt] for dt in _NET_TYPES},
        "chimA_trunk": {"play": L["play"], "schupfen": C["schupfen"],
                        "tichu": C["tichu"], "grand": C["grand"]},
        "chimB_standalone": {"play": C["play"], "schupfen": L["schupfen"],
                             "tichu": L["tichu"], "grand": L["grand"]},
    }

    export_root = RUN / "_chimera_export"
    champ_kwargs = export_live_models(_models_from(config, C), export_root / "champion")
    arm_kwargs = {name: export_live_models(_models_from(config, sd), export_root / name)
                  for name, sd in arms.items()}

    positions = generate_full_position_pool(seed=args.seed, n=args.deals)
    print(f"{args.deals} shared deals (seed {args.seed}), {args.workers} workers",
          flush=True)

    results: dict[str, np.ndarray] = {}
    for name in ("learner", "chimA_trunk", "chimB_standalone"):
        t0 = time.time()
        done = [0]

        def _progress(n, _t0=t0, _done=done, _name=name):
            _done[0] += n
            if _done[0] % 2048 < n:
                el = time.time() - _t0
                print(f"  [{_name}] {_done[0]}/{args.deals} deals "
                      f"{el / 60:.1f} min", flush=True)

        builder_arm = partial(_build_agent, "ml", skill_decile=skill_decile,
                              **arm_kwargs[name])
        builder_opp = partial(_build_agent, "ml", skill_decile=skill_decile,
                              **champ_kwargs)
        totals, _ = collect_pair_deltas(builder_arm, builder_opp, positions,
                                        workers=args.workers, progress=_progress)
        results[name] = totals
        print(_fmt(name, totals) + f"  [{(time.time() - t0) / 60:.1f} min]",
              flush=True)
        np.savez(args.out, **results, deals=np.asarray([args.deals]),
                 seed=np.asarray([args.seed]))

    lrn, a, b = (results[k] for k in ("learner", "chimA_trunk", "chimB_standalone"))
    print("\n=== chimera localization vs champion (same deals, paired) ===",
          flush=True)
    for name in results:
        print(_fmt(name, results[name]))
    print("\npaired contrasts (per-deal differences, deal luck cancels):")
    print(_fmt("chimA - learner", a - lrn))
    print(_fmt("chimB - learner", b - lrn))
    print(_fmt("interaction", lrn - a - b))
    print(f"\nrows saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
