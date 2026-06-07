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
import csv
import sys
import time
from pathlib import Path

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.heads import BCModel
from tichu_training.bc.training import load_checkpoint, save_checkpoint
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.league import League
from tichu_training.ppo.policy import BatchedPolicy
from tichu_training.ppo.rollout import collect_rollout
from tichu_training.ppo.train import AdaptiveKLController, train_ppo
from tichu_training.ppo.update import build_batch, value_loss


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


def _critic_warmup(
    bc_model, critic, sample_positions, *,
    iters: int, epochs: int, critic_lr: float,
    skill_decile: int, learner_team: int, gamma: float, lam: float,
    progress: bool, perfect_info: bool = False,
) -> None:
    """Fit the critic under the FROZEN BC policy before the policy moves, so the
    first PPO advantages aren't computed against an ignorant value baseline
    (ADR-0029 §value). Pure BC self-play; only the critic learns. Also a clean
    test of the value floor: if `value_loss` keeps falling here the critic was
    under-fit; if it plateaus immediately, ~that floor is irreducible variance.

    With `perfect_info`, the critic is fit on the 392-dim Perfect-Info features
    (ADR-0033); `batch.critic_features` carries them (and falls back to the
    observable features in the symmetric case, so this call works for both)."""
    warm_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    actor = BatchedPolicy(bc_model, critic, skill_decile=skill_decile, perfect_info=perfect_info)
    for it in range(iters):
        positions = sample_positions(it)
        traj = collect_rollout(positions, actor, opponent_policy=actor, learner_team=learner_team)
        batch = build_batch(traj, skill_decile=skill_decile, gamma=gamma, lam=lam)
        last = 0.0
        for _ in range(epochs):
            loss = value_loss(critic(batch.critic_features), batch.returns)
            warm_opt.zero_grad()
            loss.backward()
            warm_opt.step()
            last = float(loss.detach())
        if progress:
            print(f"warmup {it + 1:>3}/{iters} | value_loss {last:.3e}", flush=True)


def run_ppo_training(config, *, on_iteration=None, progress: bool = True) -> dict:
    """Run PPO Refine from `config` (a parsed dict) and write the Sharpened
    Checkpoint. Returns `{history, checkpoint_path, league_size}`.

    `progress=True` prints a one-line-per-iteration progress readout (loss /
    value / policy / entropy / kl / beta + seconds and rounds/hour) so a
    foreground or backgrounded run shows live progress, not just a final line."""
    ppo = config["ppo"]
    skill_decile = int(ppo.get("skill_decile", 9))
    learner_team = int(ppo.get("learner_team", 0))
    # Asymmetric Perfect-Info Critic (ADR-0033): the critic sees all four hands
    # (392-dim) at training time; the policy stays observable (224). Default off
    # reproduces the ADR-0029 symmetric run byte-for-byte.
    perfect_info = bool(config.get("perfect_info", False))
    critic_dim = PERFECT_INFO_DIM if perfect_info else FEATURIZER_OUTPUT_DIM

    # Warm-start the learner from the BC Checkpoint; keep a frozen copy as the
    # KL-anchor reference and the league's initial (frozen-BC) opponent.
    model = _build_model(config)
    load_checkpoint(config["warm_start"], model)
    bc_model = _freeze(copy.deepcopy(model))
    critic = ValueBaseline(critic_dim, hidden=int(config.get("critic", {}).get("hidden", 512)))

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
    bc_opponent = BatchedPolicy(
        bc_model, _freeze(copy.deepcopy(critic)),
        skill_decile=skill_decile, perfect_info=perfect_info,
    )
    league = League(
        [bc_opponent], max_snapshots=int(league_cfg.get("max_snapshots", 5)),
        perfect_info=perfect_info,
    )

    pool_seed = int(ppo.get("pool_seed", 0))
    positions_per_iter = int(ppo["positions_per_iter"])
    total_iters = int(ppo["iterations"])

    def _sample_positions(iteration: int):
        return generate_full_position_pool(
            seed=pool_seed + iteration * positions_per_iter, n=positions_per_iter
        )

    # Observability for the long background run (ADR-0029 §eval/stopping/kill needs
    # mid-run dials + a crash-safe trail). The training loop itself only returns
    # `history` at the very end and persists one final Checkpoint; here we (a) append
    # each iteration's stats to a CSV under the run dir, and (b) persist an
    # intermediate .bin every `snapshot_every` iterations so it can be exported and
    # behaviorally evaluated to apply the kill rule. This is plumbing, not a change
    # to the locked algorithm.
    run_dir = Path(config["out"]).parent
    snapshots_dir = run_dir / "snapshots"
    log_path = run_dir / "ppo_log.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    log_state = {"writer": None, "fh": None, "last_t": time.perf_counter()}

    def _on_iteration(iteration: int, stats: dict) -> None:
        now = time.perf_counter()
        row = {"iter": iteration, "wall_s": round(now - log_state["last_t"], 3), **stats}
        log_state["last_t"] = now
        if log_state["writer"] is None:
            fh = open(log_path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            log_state["fh"], log_state["writer"] = fh, writer
        log_state["writer"].writerow(row)
        log_state["fh"].flush()

        snapped = (iteration + 1) % snapshot_every == 0
        if snapped:
            league.snapshot(model, critic, skill_decile=skill_decile)
            save_checkpoint(
                model, optimizer, step=iteration + 1,
                path=str(snapshots_dir / f"iter_{iteration + 1:05d}.bin"),
            )

        if progress:
            wall_s = row["wall_s"]
            rph = positions_per_iter * 3600.0 / wall_s if wall_s > 0 else float("nan")
            print(
                f"iter {iteration + 1:>4}/{total_iters} | "
                f"loss {stats['loss']:.3e}  v {stats['value_loss']:.3e}  "
                f"p {stats['policy_loss']:+.4f}  ent {stats['entropy']:.3f}  "
                f"kl {stats['kl_to_bc']:.4f}  beta {stats['kl_coef']:.2e} | "
                f"{wall_s:.1f}s  ~{rph:,.0f} r/h"
                f"{'  [snap]' if snapped else ''}",
                flush=True,
            )

        if on_iteration is not None:
            on_iteration(iteration, stats)

    warmup_iters = int(config.get("critic", {}).get("warmup_iters", 0))
    if progress:
        print(
            f"PPO Refine: {total_iters} iters x M={positions_per_iter} "
            f"(decile {skill_decile}, learner team {learner_team})\n"
            f"  warm_start: {config['warm_start']}\n"
            f"  out:        {config['out']}  (snapshots every {snapshot_every} -> {snapshots_dir})\n"
            f"  log:        {log_path}"
            + (f"\n  critic warm-up: {warmup_iters} iters (frozen BC policy)" if warmup_iters else ""),
            flush=True,
        )

    if warmup_iters > 0:
        _critic_warmup(
            bc_model, critic, _sample_positions,
            iters=warmup_iters,
            epochs=int(config.get("critic", {}).get("warmup_epochs", 3)),
            critic_lr=float(ppo.get("critic_lr", 1e-3)),
            skill_decile=skill_decile, learner_team=learner_team,
            gamma=float(ppo.get("gamma", 1.0)), lam=float(ppo.get("lam", 0.95)),
            progress=progress, perfect_info=perfect_info,
        )

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
        perfect_info=perfect_info,
    )

    if log_state["fh"] is not None:
        log_state["fh"].close()

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
