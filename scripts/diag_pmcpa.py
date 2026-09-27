r"""pMCPA adaptation diagnostic (ADR-0036) — why is adapted << theta_o?

A fast offline proxy for the catastrophic A/B regression (-247/round): instead of
full-round h2h, it measures how badly the per-Round adaptation corrupts the play
policy on a held-out set of UNRELATED decision states. Rows are collected ONCE per
root (they don't depend on the containment knobs), then the kl_coef / lr / steps
grid is swept cheaply on the cached rows.

Per setting it reports, averaged over roots:
  * finite_rate  - fraction of held-out states whose theta_a play logits are all
                   finite. < 1.0 => MLAgent.act falls back to RANDOM (the -247
                   mechanism would then be random play, not subtle overfit). [H0]
  * KL_global    - mean KL(theta_a || theta_o) on held-out states (global drift). [H1/H5]
  * KL_local     - mean KL on the root's OWN adaptation states (local drift). [H2]
  * flip_rate    - fraction of held-out states where the argmax action changed. [H3]

  py -m scripts.diag_pmcpa --theta-o-dir C:\workbench\tichu\data\runs\cotrain_vine_v1\warm \
      --pool C:\workbench\tichu\data\full_position_pool_s0_n20000.parquet --roots 5 --worlds 128
"""

import argparse
import copy
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch


def _masked_argmax(logits, masks):
    return logits.masked_fill(~masks, float("-inf")).argmax(dim=1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Diagnose pMCPA adaptation corruption.")
    p.add_argument("--theta-o-dir", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--tag", default="iter06225")
    p.add_argument("--roots", type=int, default=5, help="adaptation root rounds")
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--eval-states", type=int, default=150)
    args = p.parse_args(argv)

    from tichu_engine.legality import legal_actions_for
    from tichu_eval.full_position_pool import load_full_position_pool
    from tichu_inference.ml_agent import MLAgent
    from tichu_training.action_space import legal_mask
    from tichu_training.featurizer import featurize
    from tichu_training.ppo.pmcpa import (
        adapt_play_net, collect_pmcpa_rows, load_offline_models,
    )
    from tichu_training.ppo.update import kl_anchor_loss
    from tichu_training.search.blunder_miner import record_round

    models = load_offline_models(args.theta_o_dir, tag=args.tag)
    play_o = models["play"]
    agent_o = MLAgent.from_loaded(
        play_o, schupfen=models["schupfen"], tichu_call=models["tichu"],
        grand_call=models["grand"], skill_decile=9,
    )
    pool = load_full_position_pool(Path(args.pool))

    # Roots (adaptation starts) + a held-out set of UNRELATED play-decision states
    # drawn from OTHER rounds, so KL_global measures collateral damage to states the
    # adaptation never touched. Each (state, seat) carries a real legal mask.
    roots, eval_sd = [], []
    for pos in pool[: args.roots + 6]:
        _, decisions = record_round([agent_o] * 4, pos)
        play_dec = [d for d in decisions
                    if len(legal_actions_for(d.state.private_view(d.seat))) > 1]
        if not play_dec:
            continue
        if len(roots) < args.roots:
            roots.append(play_dec[0].state.private_view(play_dec[0].seat))
        eval_sd.extend((d.state, d.seat) for d in play_dec[1::4])
    eval_sd = eval_sd[: args.eval_states]
    print(f"roots={len(roots)}  eval_states={len(eval_sd)}  worlds={args.worlds}", flush=True)

    eval_feats = torch.stack([
        torch.from_numpy(featurize(s.private_view(seat))).float() for s, seat in eval_sd])
    eval_skill = torch.full((eval_feats.shape[0],), 9, dtype=torch.long)
    eval_masks = torch.stack([
        torch.as_tensor(legal_mask("play", s, seat), dtype=torch.bool) for s, seat in eval_sd])

    def logits(net, feats, skill):
        with torch.no_grad():
            return net(feats, skill)["play"]

    base = logits(play_o, eval_feats, eval_skill)
    base_am = _masked_argmax(base, eval_masks)

    # Collect rows ONCE per root (the expensive part), with the production estimator
    # config (emit_branches on) — independent of the containment knobs swept below.
    print("collecting rows per root (one-time)...", flush=True)
    root_rows, root_feats, root_skill, root_masks = [], [], [], []
    for i, rv in enumerate(roots):
        rows = collect_pmcpa_rows(
            models, rv, worlds=args.worlds, decisions_per_world=4, branches=4,
            skill_decile=9, seed=i, emit_branches=True)
        root_rows.append(rows)
        f = torch.stack([torch.as_tensor(r["features"]).float() for r in rows])
        root_feats.append(f)
        root_skill.append(torch.full((f.shape[0],), 9, dtype=torch.long))
        root_masks.append(torch.stack([torch.as_tensor(r["mask"], dtype=torch.bool) for r in rows]))
        print(f"  root {i}: {len(rows)} rows", flush=True)

    # The sweep: production setting first (reproduce), then the containment grid.
    settings = [
        dict(kl_coef=1.0, lr=0.05, steps=3),    # production (the -247 config)
        dict(kl_coef=3.0, lr=0.05, steps=3),
        dict(kl_coef=10.0, lr=0.05, steps=3),
        dict(kl_coef=30.0, lr=0.05, steps=3),
        dict(kl_coef=100.0, lr=0.05, steps=3),
        dict(kl_coef=300.0, lr=0.05, steps=3),
        dict(kl_coef=1.0, lr=0.01, steps=3),    # lr lever at weak anchor
        dict(kl_coef=1.0, lr=0.05, steps=1),    # steps lever at weak anchor
    ]

    print(f"\n{'kl_coef':>8} {'lr':>6} {'steps':>5} | {'finite':>7} {'KL_glob':>8} "
          f"{'KL_loc':>7} {'flip%':>6}")
    print("-" * 56)
    for st in settings:
        finite_rates, kl_globs, kl_locs, flips = [], [], [], []
        for i in range(len(roots)):
            play_a = copy.deepcopy(play_o)
            adapt_play_net(play_a, play_o, root_rows[i], skill_decile=9, **st)
            la = logits(play_a, eval_feats, eval_skill)
            fin = torch.isfinite(la).all(dim=1)
            finite_rates.append(fin.float().mean().item())
            if fin.any():
                kl_globs.append(kl_anchor_loss(la[fin], base[fin], eval_masks[fin]).item())
                flips.append((_masked_argmax(la, eval_masks)[fin] != base_am[fin])
                             .float().mean().item())
            # local drift on the root's own adaptation states
            lo = logits(play_a, root_feats[i], root_skill[i])
            lb = logits(play_o, root_feats[i], root_skill[i])
            if torch.isfinite(lo).all(dim=1).any():
                m = torch.isfinite(lo).all(dim=1)
                kl_locs.append(kl_anchor_loss(lo[m], lb[m], root_masks[i][m]).item())
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(f"{st['kl_coef']:>8.0f} {st['lr']:>6.3f} {st['steps']:>5} | "
              f"{mean(finite_rates):>7.2f} {mean(kl_globs):>8.3f} {mean(kl_locs):>7.3f} "
              f"{100 * mean(flips):>6.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
