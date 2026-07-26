"""Vine v3 autopsy round 2 — the disagreement replay (2026-07-26).

Round 1 (`diag_vine_v3_kl_support.py`) refuted the KL-support hole: drift is
small, leashed, and uniform, yet the ~5.6% of decisions where the learner now
disagrees with its anchor net −2/round at the gate. So the vine GRADIENT is
suspected of being miscalibrated where it acts — the standing mechanism is the
τ-continuation family (pMCPA/piKL killer): branch returns computed under the
frozen guard-ON field can rank actions differently than the learner's own
continuation would.

This script measures both questions directly on the stopped weights. Fresh
trunk games are recorded with the learner (argmax, guard off — vine trunk
semantics). At every eligible decision the anchor net's preferred action is
computed; on intent-level disagreement, BOTH actions are played out to round
end in the same world under TWO continuations:

  field  = the run's Reference Field (champion nets, guard ON) — the exact
           estimator vine v3 trained on
  self   = the learner itself (guard OFF) — v2's estimator semantics, the
           closest available proxy for "what the learner actually does next"

Per state: dF = R_field(learner action) − R_field(anchor action)
           dS = R_self (learner action) − R_self (anchor action)

Readout: mean dF > 0 & mean dS < 0  → continuation bias confirmed (the vine
teaches actions that only look good when the field finishes the round).
Both < 0 → the drift isn't even aligned with the vine's own estimator
(optimization/normalization artifact). Both > 0 → the harm is not at these
single decisions. Harm accounting: mean dS × disagreements/round vs the
gate's −2/round.

    $env:PYTHONPATH = "<worktree>\\src"
    py scripts/diag_vine_v3_disagreement_replay.py --games 400 --seed 160000000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from tichu_engine.legality import legal_actions_for
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models
from tichu_training.featurizer import _combination_to_action_index
from tichu_training.ppo.vine import _is_pool_state
from tichu_training.search.blunder_miner import (
    playout_from,
    record_round,
    team_relative,
)

RUN = Path(r"C:/workbench/tichu/data/runs/cotrain_v6_vine_v3_gated")


def _load_nets(config, state_dicts):
    models = _build_models(config)
    for dt in _NET_TYPES:
        models[dt].load_state_dict(state_dicts[dt])
        models[dt].eval()
    return models


def _agent(models, *, skill_decile, guard):
    from tichu_inference.ml_agent import MLAgent
    return MLAgent.from_loaded(
        models["play"], schupfen=models["schupfen"], tichu_call=models["tichu"],
        grand_call=models["grand"], skill_decile=skill_decile,
        partner_trick_guard=guard,
    )


def _boot_ci(x, iters=10_000, seed=0):
    arr = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, arr.size, size=(iters, arr.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cotrain_v6_vine_v3_gated.yaml")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=160_000_000,
                    help="fresh stream, distinct from the round-1 diag (150M)")
    ap.add_argument("--out", default=str(RUN / "diag_disagreement_replay.npz"))
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    skill_decile = int(config["ppo"].get("skill_decile", 9))

    payload = torch.load(RUN / "train_state.bin", map_location="cpu", weights_only=False)
    champ = torch.load(RUN / "_champion.pt", map_location="cpu", weights_only=False)
    print(f"bundle iteration: {payload['iteration']}")

    learner = _load_nets(config, payload["models"])
    champion = _load_nets(config, champ["models"])

    trunk_agent = _agent(learner, skill_decile=skill_decile, guard=False)
    self_agents = [trunk_agent] * 4                     # v2 estimator semantics
    field_agents = [_agent(champion, skill_decile=skill_decile, guard=True)] * 4
    anchor_ranker = _agent(champion, skill_decile=skill_decile, guard=False)

    positions = generate_full_position_pool(seed=args.seed, n=args.games)
    rows = {k: [] for k in ("game", "seat", "hand", "pool", "n_legal",
                            "rf_l", "rf_a", "rs_l", "rs_a")}
    n_eligible = n_skipped = 0
    t0 = time.time()

    def _save():
        np.savez(args.out, **{k: np.asarray(v) for k, v in rows.items()},
                 games_done=np.asarray([gi + 1]), seed=np.asarray([args.seed]))

    for gi, position in enumerate(positions):
        _, decisions = record_round([trunk_agent] * 4, position)
        for d in decisions:
            pv = d.state.private_view(d.seat)
            legal = list(legal_actions_for(pv))
            if len(legal) <= 1:
                continue
            n_eligible += 1
            anchor_first = anchor_ranker.rank_actions(pv)[0]
            idx_l = _combination_to_action_index(d.chosen)
            idx_a = _combination_to_action_index(anchor_first)
            if idx_l is None or idx_a is None:
                n_skipped += 1
                continue
            if idx_l == idx_a:
                continue  # intent-level agreement (realization ties don't count)

            def _rel(agents, forced):
                return team_relative(
                    playout_from(agents, d.state, forced_action=forced,
                                 asked_tichu=d.asked_tichu,
                                 initial_scores=d.initial_scores),
                    d.seat,
                )

            rows["game"].append(gi)
            rows["seat"].append(d.seat)
            rows["hand"].append(len(d.state.hands[d.seat]))
            rows["pool"].append(_is_pool_state(d))
            rows["n_legal"].append(len(legal))
            rows["rf_l"].append(_rel(field_agents, d.chosen))
            rows["rf_a"].append(_rel(field_agents, anchor_first))
            rows["rs_l"].append(_rel(self_agents, d.chosen))
            rows["rs_a"].append(_rel(self_agents, anchor_first))

        if (gi + 1) % 10 == 0:
            n = len(rows["game"])
            el = time.time() - t0
            msg = (f"  {gi + 1}/{args.games} games  {el / 60:.1f} min  "
                   f"disagreements={n} ({n / max(n_eligible, 1) * 100:.1f}% of "
                   f"{n_eligible} eligible)")
            if n >= 20:
                dF = np.asarray(rows["rf_l"]) - np.asarray(rows["rf_a"])
                dS = np.asarray(rows["rs_l"]) - np.asarray(rows["rs_a"])
                msg += f"  running dF={dF.mean():+.2f} dS={dS.mean():+.2f}"
            print(msg, flush=True)
        if (gi + 1) % 50 == 0:
            _save()

    _save()
    dF = np.asarray(rows["rf_l"], dtype=float) - np.asarray(rows["rf_a"], dtype=float)
    dS = np.asarray(rows["rs_l"], dtype=float) - np.asarray(rows["rs_a"], dtype=float)
    n = dF.size
    per_round = n / args.games
    print(f"\n=== disagreement replay: {n} states from {args.games} rounds "
          f"({per_round:.2f}/round; {n / max(n_eligible, 1) * 100:.1f}% of "
          f"{n_eligible} eligible; {n_skipped} skipped unmappable) ===")
    for name, d in (("dF (field = vine estimator)", dF), ("dS (self-continuation)", dS)):
        m, lo, hi = _boot_ci(d)
        print(f"{name:>28}: mean {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  "
              f"med {np.median(d):+.1f}  >0: {(d > 0).mean() * 100:.1f}%  "
              f"<0: {(d < 0).mean() * 100:.1f}%  =0: {(d == 0).mean() * 100:.1f}%")
    both = ~((dF == 0) | (dS == 0))
    if both.any():
        flip_fa = ((dF > 0) & (dS < 0))[both].mean() * 100
        flip_af = ((dF < 0) & (dS > 0))[both].mean() * 100
        agree = ((dF > 0) == (dS > 0))[both].mean() * 100
        print(f"non-tied on both ({both.sum()}): sign-agree {agree:.1f}%  "
              f"field+/self− {flip_fa:.1f}%  field−/self+ {flip_af:.1f}%")
    print(f"\nharm accounting (× {per_round:.2f} disagreements/round; "
          f"gate said −2.2/round):")
    for name, d in (("field", dF), ("self", dS)):
        m, lo, hi = _boot_ci(d, seed=1)
        print(f"  {name:>5}: {m * per_round:+.2f}/round "
              f"[{lo * per_round:+.2f}, {hi * per_round:+.2f}]")
    pool = np.asarray(rows["pool"], dtype=bool)
    for label, sel in (("pool states", pool), ("early states", ~pool)):
        if sel.any():
            print(f"  {label}: n={sel.sum()}  dF {dF[sel].mean():+.2f}  "
                  f"dS {dS[sel].mean():+.2f}")
    print(f"rows saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
