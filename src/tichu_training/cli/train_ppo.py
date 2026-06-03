"""train_ppo — PPO Refine of the play policy (ADR-0029).

Warm-starts from a BC Checkpoint, runs KL-anchored self-play through the rules
engine against a small opponent league, and writes a Sharpened Checkpoint that
is byte-compatible with a BC Checkpoint (drop-in `master`-tier swap).

    py -m tichu_training.cli.train_ppo --config configs/ppo_v5_smoke.yaml

The heavy lifting is `run_ppo_training(config)`; `main` only parses args and
loads the YAML. Behavioral dials (caller-passivity) are not computed in-loop —
run `eval_matrix --mode behavioral` on the exported Sharpened Checkpoint, which
the existing tooling already supports.
"""

import argparse
import copy
import sys
from pathlib import Path

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.heads import BCModel
from tichu_training.bc.training import load_checkpoint, save_checkpoint
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.ppo.league import League
from tichu_training.ppo.policy import BatchedPolicy
from tichu_training.ppo.train import AdaptiveKLController, train_ppo


def _build_model(config) -> BCModel:
    m = config.get("model", {})
    return BCModel(
        feature_dim=FEATURIZER_OUTPUT_DIM,
        skill_buckets=10,
        skill_dim=int(m.get("skill_dim", 64)),
        trunk_hidden=int(m.get("trunk_hidden", 1024)),
        trunk_depth=int(m.get("trunk_depth", 4)),
        trunk_out_dim=int(m.get("trunk_out_dim", 512)),
        head_hidden=int(m.get("head_hidden", 256)),
    )


def _freeze(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def run_ppo_training(config, *, on_iteration=None) -> dict:
    """Run PPO Refine from `config` (a parsed dict) and write the Sharpened
    Checkpoint. Returns `{history, checkpoint_path, league_size}`."""
    ppo = config["ppo"]
    skill_decile = int(ppo.get("skill_decile", 9))
    learner_team = int(ppo.get("learner_team", 0))

    # Warm-start the learner from the BC Checkpoint; keep a frozen copy as the
    # KL-anchor reference and the league's initial (frozen-BC) opponent.
    model = _build_model(config)
    load_checkpoint(config["warm_start"], model)
    bc_model = _freeze(copy.deepcopy(model))
    critic = ValueBaseline(FEATURIZER_OUTPUT_DIM, hidden=int(config.get("critic", {}).get("hidden", 512)))

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": float(ppo.get("policy_lr", 1e-4))},
        {"params": critic.parameters(), "lr": float(ppo.get("critic_lr", 1e-3))},
    ])

    kl_cfg = config.get("kl", {})
    kl_controller = AdaptiveKLController(
        coef=float(kl_cfg.get("coef", 1.0)),
        target=float(kl_cfg.get("target", 0.02)),
        factor=float(kl_cfg.get("factor", 2.0)),
    )

    league_cfg = config.get("league", {})
    snapshot_every = int(league_cfg.get("snapshot_every", 5))
    bc_opponent = BatchedPolicy(bc_model, _freeze(copy.deepcopy(critic)), skill_decile=skill_decile)
    league = League([bc_opponent], max_snapshots=int(league_cfg.get("max_snapshots", 5)))

    pool_seed = int(ppo.get("pool_seed", 0))
    positions_per_iter = int(ppo["positions_per_iter"])

    def _sample_positions(iteration: int):
        return generate_full_position_pool(
            seed=pool_seed + iteration * positions_per_iter, n=positions_per_iter
        )

    def _on_iteration(iteration: int, stats: dict) -> None:
        if (iteration + 1) % snapshot_every == 0:
            league.snapshot(model, critic, skill_decile=skill_decile)
        if on_iteration is not None:
            on_iteration(iteration, stats)

    history = train_ppo(
        model, critic, bc_model, _sample_positions,
        optimizer=optimizer,
        kl_controller=kl_controller,
        iterations=int(ppo["iterations"]),
        gamma=float(ppo.get("gamma", 1.0)),
        lam=float(ppo.get("lam", 0.95)),
        clip_eps=float(ppo.get("clip_eps", 0.1)),
        vf_coef=float(ppo.get("vf_coef", 0.5)),
        ent_coef=float(ppo.get("ent_coef", 0.01)),
        ppo_epochs=int(ppo.get("ppo_epochs", 3)),
        skill_decile=skill_decile,
        learner_team=learner_team,
        opponent_policy_provider=league.sample,
        on_iteration=_on_iteration,
    )

    out_path = config["out"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, step=int(ppo["iterations"]), path=out_path)

    return {"history": history, "checkpoint_path": out_path, "league_size": len(league)}


def main(argv=None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description="PPO Refine of the play policy (ADR-0029)")
    parser.add_argument("--config", required=True, help="Path to the PPO config YAML.")
    parser.add_argument("--out", default=None, help="Override the config's output checkpoint path.")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if args.out is not None:
        config["out"] = args.out

    result = run_ppo_training(config)
    final = result["history"][-1] if result["history"] else {}
    print(
        f"PPO Refine done: {len(result['history'])} iterations, "
        f"league_size={result['league_size']}, "
        f"final loss={final.get('loss', float('nan')):.4f}, "
        f"kl_to_bc={final.get('kl_to_bc', float('nan')):.4f} -> {result['checkpoint_path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
