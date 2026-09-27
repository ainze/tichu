"""Vine v3 autopsy — the KL-support hole check (2026-07-26).

The v3 run safety-stopped CI-clean NEGATIVE (-2.2/round vs its own frozen
warm-start anchor) while the logged play_kl held ~0.02. Suspect: `ppo_update`
measures `kl_anchor_loss` on the play training batch itself, which under
ADR-0040 is the STRATIFIED VINE ROWS (hand<=10 or caller-live) — so early-play
states have no gradient of their own AND no leash; the trunk can drift them
arbitrarily while play_kl reads 0.02.

This script measures that directly on the stopped weights: drive fresh vine
trunk games (learner argmax, guard off — the exact vine visitation
distribution), and compute per-state KL(learner || anchor) over the
legal-masked play distribution, split by `_is_pool_state`. The anchor is the
bundle's own `play_anchor` (the tensor training actually leashed against).

Prediction if the hole is real: pool-state KL ~ the logged 0.02; early-state
KL well above it. Also reported: argmax agreement with the anchor per group
(the drift measure that translates to strength).

    $env:PYTHONPATH = "<worktree>\\src"
    py scripts/diag_vine_v3_kl_support.py --games 64 --seed 150000000
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from tichu_engine.legality import legal_actions_for
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.action_space import legal_mask
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models
from tichu_training.featurizer import featurize
from tichu_training.ppo.vine import _is_pool_state
from tichu_training.search.blunder_miner import record_round

RUN = Path(r"C:/workbench/tichu/data/runs/cotrain_v6_vine_v3_gated")


def _masked_logp(model, features, skill, masks):
    with torch.no_grad():
        logits = model(features, skill)["play"]
    return torch.log_softmax(logits.masked_fill(~masks, float("-inf")), dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cotrain_v6_vine_v3_gated.yaml")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=150_000_000,
                    help="fresh stream, >=100M (clear of every training seed range)")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    skill_decile = int(config["ppo"].get("skill_decile", 9))

    payload = torch.load(RUN / "train_state.bin", map_location="cpu", weights_only=False)
    print(f"bundle iteration: {payload['iteration']}")
    models = _build_models(config)
    for dt in _NET_TYPES:
        models[dt].load_state_dict(payload["models"][dt])
        models[dt].eval()
    anchor = _build_models(config)["play"]
    assert payload.get("play_anchor") is not None, "bundle carries no play_anchor"
    anchor.load_state_dict(payload["play_anchor"])
    anchor.eval()

    # Sanity: with zero promotions the anchor must still equal the champion's play net.
    champ = torch.load(RUN / "_champion.pt", map_location="cpu", weights_only=False)
    same = all(torch.equal(payload["play_anchor"][k], champ["models"]["play"][k])
               for k in payload["play_anchor"])
    print(f"anchor == _champion.pt play weights: {same}")

    from tichu_inference.ml_agent import MLAgent
    agent = MLAgent.from_loaded(
        models["play"], schupfen=models["schupfen"], tichu_call=models["tichu"],
        grand_call=models["grand"], skill_decile=skill_decile,
    )
    agents = [agent] * 4

    positions = generate_full_position_pool(seed=args.seed, n=args.games)
    feats, masks, pool_flags, hand_sizes = [], [], [], []
    for i, position in enumerate(positions):
        _, decisions = record_round(agents, position)
        for d in decisions:
            pv = d.state.private_view(d.seat)
            if len(legal_actions_for(pv)) <= 1:
                continue  # forced moves carry no KL freedom and no training rows
            feats.append(featurize(pv))
            masks.append(legal_mask("play", d.state, d.seat))
            pool_flags.append(_is_pool_state(d))
            hand_sizes.append(len(d.state.hands[d.seat]))
        if (i + 1) % 16 == 0:
            print(f"  {i + 1}/{args.games} games, {len(feats)} decisions")

    features = torch.from_numpy(np.stack(feats))
    mask_t = torch.from_numpy(np.stack(masks)).bool()
    skill = torch.full((features.shape[0],), skill_decile, dtype=torch.long)
    pool = np.asarray(pool_flags)
    hands = np.asarray(hand_sizes)

    lp_new = _masked_logp(models["play"], features, skill, mask_t)
    lp_ref = _masked_logp(anchor, features, skill, mask_t)
    p_new = lp_new.exp()
    diff = (lp_new - lp_ref).masked_fill(~mask_t, 0.0)
    kl = (p_new * diff).sum(dim=-1).numpy()  # KL(learner || anchor), kl_anchor_loss per-row
    agree = (lp_new.argmax(dim=-1) == lp_ref.argmax(dim=-1)).numpy()

    def _report(name, sel):
        if not sel.any():
            print(f"{name:>28}: n=0")
            return
        k = kl[sel]
        print(f"{name:>28}: n={sel.sum():5d}  KL mean={k.mean():.4f} "
              f"med={np.median(k):.4f} p90={np.percentile(k, 90):.4f}  "
              f"argmax-agree={agree[sel].mean() * 100:.1f}%")

    print(f"\nKL(learner || anchor) on {len(kl)} fresh vine-trunk decisions "
          f"(seed {args.seed}); training's logged play_kl ~ 0.02-0.025 on vine rows")
    _report("POOL (hand<=10 or caller)", pool)
    _report("EARLY (hand>=11, no call)", ~pool)
    for lo, hi in ((14, 14), (13, 13), (11, 12), (8, 10), (1, 7)):
        sel = (hands >= lo) & (hands <= hi)
        _report(f"hand {lo}-{hi}", sel)
    if pool.any() and (~pool).any():
        print(f"\nearly-state KL / pool-state KL ratio: "
              f"{kl[~pool].mean() / max(kl[pool].mean(), 1e-9):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
