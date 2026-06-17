"""train_cotrain — full-stack co-training PPO (ADR-0034).

Sharpens play + the Schupfen Network + the Tichu/Grand Call Networks together in
one self-play run, warm-started from each net's BC Checkpoint and KL-anchored to
its frozen BC. The command is **idempotent**: a Resume Bundle in the run dir is
auto-resumed (continue from the saved iteration); otherwise it fresh-starts from
the BC warm-starts. `--restart` forces fresh. The whole training state is written
atomically every iteration, so a Ctrl-C at any moment continues losslessly.

    py -m tichu_training.cli.train_cotrain --config configs/cotrain_v5.yaml
    #   Ctrl-C any time -> re-run the same command to resume
    #   --restart -> ignore the bundle, start fresh

Heavy strength reads (the seat-swap Tournament) are NOT run in-loop (OOM safety,
ADR-0034) — use the separate `check_cotrain` command on a serving snapshot.
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
from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.heads import BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.bc.training import load_checkpoint, save_checkpoint
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.cotrain import (
    BatchedCoTrainPolicy,
    build_cotrain_batch,
    train_cotrain,
)
from tichu_training.ppo.cotrain_resume import (
    PREV_SUFFIX,
    capture_rng,
    load_resume_bundle,
    restore_rng,
    save_resume_bundle,
)
from tichu_training.ppo.train import AdaptiveKLController
from tichu_training.ppo.update import value_loss
from tichu_training.ppo.rollout import collect_rollout

_NET_TYPES = ("play", "schupfen", "tichu", "grand")
_BUNDLE_NAME = "train_state.bin"
# The CSV schema (`log_fields`) is built per-run from the active decision types so the
# per-net dials always line up across a stop/resume (ADR-0034 attribution); the wish
# columns appear only when `cotrain_wish` is on, keeping existing run logs unchanged.


def _build_models(config) -> dict:
    m = config.get("model", {})
    sm = config.get("schupfen_model", {})
    cm = config.get("call_model", {})
    return {
        "play": BCModel(
            feature_dim=FEATURIZER_OUTPUT_DIM, skill_buckets=10,
            skill_dim=int(m.get("skill_dim", 64)),
            trunk_hidden=int(m.get("trunk_hidden", 1024)),
            trunk_depth=int(m.get("trunk_depth", 4)),
            trunk_out_dim=int(m.get("trunk_out_dim", 512)),
            head_hidden=int(m.get("head_hidden", 256)),
        ),
        "schupfen": SchupfenNetwork(
            FEATURIZER_OUTPUT_DIM, skill_dim=int(sm.get("skill_dim", 64)),
            hidden=int(sm.get("hidden", 256)),
        ),
        "tichu": TichuCallNetwork(
            FEATURIZER_OUTPUT_DIM, skill_dim=int(cm.get("skill_dim", 64)),
            hidden=int(cm.get("hidden", 256)),
        ),
        "grand": GrandTichuCallNetwork(
            FEATURIZER_OUTPUT_DIM, skill_dim=int(cm.get("skill_dim", 64)),
            hidden=int(cm.get("hidden", 256)),
        ),
    }


def _freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _build_optimizer(models, critic, *, policy_lr: float, critic_lr: float,
                     freeze_nets=()):
    """Adam over the trainable policy nets + the shared critic.

    Nets named in `freeze_nets` are frozen at their warm-started BC weights: still
    driven in the rollout (so the full stack plays at BC quality) but excluded from
    the optimizer, so only the remaining nets (+ critic) learn. This isolates one
    net's training from cross-net interference — the shared critic and the single
    joint `optimizer.step()` are the only coupling between the otherwise-separate
    nets (ADR-0034), so freezing the call/schupfen nets lets the play net train
    against a STATIONARY teammate environment instead of chasing drifting calls.

    Returns `(optimizer, trainable_net_names)`. Freezing happens here so the params
    are excluded before the optimizer captures them."""
    freeze = set(freeze_nets)
    if "play" in freeze:
        raise ValueError("cannot freeze 'play' — there would be nothing to train")
    unknown = freeze - set(_NET_TYPES)
    if unknown:
        raise ValueError(f"freeze_nets has unknown nets {sorted(unknown)}; valid: {list(_NET_TYPES)}")
    for dt in freeze:
        _freeze(models[dt])
    trainable = [dt for dt in _NET_TYPES if dt not in freeze]
    groups = [{"params": models[dt].parameters(), "lr": policy_lr} for dt in trainable]
    groups.append({"params": critic.parameters(), "lr": critic_lr})
    return torch.optim.Adam(groups), trainable


def _kl_controllers(config, decision_types) -> dict:
    kl = config.get("kl", {})
    out = {}
    for dt in decision_types:
        c = kl.get(dt, {})
        tf = c.get("target_final")
        out[dt] = AdaptiveKLController(
            coef=float(c.get("coef", 1.0)),
            target=float(c.get("target", 0.02)),
            factor=float(c.get("factor", 2.0)),
            target_final=None if tf is None else float(tf),
            anneal_start=int(c.get("anneal_start", 0)),
            anneal_iters=int(c.get("anneal_iters", 0)),
        )
    return out


def _critic_warmup(bc_policy, critic, sample_positions, *, iters, epochs, lr,
                   skill_decile, learner_team, gamma, lam, progress):
    """Fit the shared critic under the FROZEN BC full-stack policy before the policy
    moves, so the first advantages aren't computed against an ignorant baseline
    (ADR-0034, mirroring ADR-0029 §value but over the full call/schupfen/play stack)."""
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    for it in range(iters):
        trajs = collect_rollout(sample_positions(it), bc_policy, opponent_policy=bc_policy,
                                learner_team=learner_team)
        batch = build_cotrain_batch(trajs, skill_decile=skill_decile, gamma=gamma, lam=lam)
        last = 0.0
        for _ in range(epochs):
            loss = value_loss(critic(batch.critic_features), batch.returns)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach())
        if progress:
            print(f"critic-warmup {it + 1:>3}/{iters} | value_loss {last:.3e}", flush=True)


def run_cotrain_training(config, *, restart: bool = False, on_iteration=None, progress: bool = True) -> dict:
    """Run (or resume) full-stack co-training from `config`. Returns
    `{history, run_dir, start_iter, final_iter, bundle_path}`."""
    ppo = config["ppo"]
    skill_decile = int(ppo.get("skill_decile", 9))
    learner_team = int(ppo.get("learner_team", 0))
    perfect_info = bool(config.get("perfect_info", True))
    # GPU the heavy update step only (profiling showed the rollout is CPU-engine-bound,
    # ADR-0034). Falls back to CPU with a notice if CUDA is unavailable.
    update_device = str(config.get("update_device", "cpu"))
    if update_device != "cpu" and not torch.cuda.is_available():
        if progress:
            print(f"  update_device={update_device} requested but CUDA unavailable -> CPU", flush=True)
        update_device = "cpu"
    rollout_workers = int(config.get("rollout_workers", 1))
    # Opt-in opponent league (default off = pure self-play). Parallel-only: opponents
    # run in rollout workers that load frozen weight files (ADR-0034 league lever).
    league_cfg = config.get("league", {})
    league_enabled = bool(league_cfg.get("enabled", False))
    if league_enabled and rollout_workers <= 1:
        if progress:
            print("  league requires rollout_workers>1 -> disabled (pure self-play)", flush=True)
        league_enabled = False
    # Wish co-training is opt-in (ADR-0034 addendum). OFF -> the wish stays on the
    # frozen inline seat agent and training is byte-identical to before (no wish data,
    # so the wish controller/coef/log-columns stay inert). The wish head rides the
    # play net's trunk either way.
    cotrain_wish = bool(config.get("cotrain_wish", False))
    decision_types = _NET_TYPES + (("wish",) if cotrain_wish else ())
    log_fields = ["iter", "wall_s", "loss", "value_loss"] + [
        f"{dt}_{k}" for dt in decision_types
        for k in ("policy_loss", "kl", "entropy", "kl_coef")
    ]
    if bool(config.get("vine", {}).get("enabled", False)):
        log_fields.append("vine_rows")
    gamma = float(ppo.get("gamma", 1.0))
    lam = float(ppo.get("lam", 0.95))
    pool_seed = int(ppo.get("pool_seed", 0))
    positions_per_iter = int(ppo["positions_per_iter"])
    total_iters = int(ppo["iterations"])

    run_dir = Path(config["run_dir"])
    snapshots_dir = run_dir / "snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = str(run_dir / _BUNDLE_NAME)
    log_path = run_dir / "ppo_log.csv"

    # Build nets, warm-start each from its BC Checkpoint; keep frozen BC anchors.
    models = _build_models(config)
    warm = config["warm_start"]
    for dt in _NET_TYPES:
        load_checkpoint(warm[dt], models[dt])
    bc_models = {dt: _freeze(copy.deepcopy(models[dt])) for dt in _NET_TYPES}

    critic_dim = PERFECT_INFO_DIM if perfect_info else FEATURIZER_OUTPUT_DIM
    critic = ValueBaseline(
        critic_dim,
        hidden=int(config.get("critic", {}).get("hidden", 512)),
        depth=int(config.get("critic", {}).get("depth", 1)),
    )

    freeze_nets = list(config.get("freeze_nets", []))
    optimizer, trainable_nets = _build_optimizer(
        models, critic,
        policy_lr=float(ppo.get("policy_lr", 1e-4)),
        critic_lr=float(ppo.get("critic_lr", 1e-3)),
        freeze_nets=freeze_nets,
    )
    if freeze_nets and progress:
        print(f"  freeze_nets: {freeze_nets} held at BC; training {trainable_nets} + critic", flush=True)

    kl_controllers = _kl_controllers(config, decision_types)
    ent_cfg = config.get("entropy", {})
    ent_coefs = {dt: float(ent_cfg.get(dt, 0.01)) for dt in decision_types}
    # Periodic play re-anchor (ADR-0035 addendum): the KL ball around a frozen
    # anchor caps TOTAL movement; moving the anchor every N iterations turns it
    # into a trail of contained steps. 0 = off (the fixed-anchor regime).
    reanchor_play_every = int(config.get("reanchor_play_every", 0))

    def _sample_positions(iteration: int):
        return generate_full_position_pool(
            seed=pool_seed + iteration * positions_per_iter, n=positions_per_iter
        )

    # Idempotent resume: a bundle in the run dir continues from the saved iteration;
    # absent (or --restart) -> fresh from the BC warm-starts (ADR-0034).
    start_iter = 0
    resumed = False
    resume_payload = None
    if Path(bundle_path).exists() and not restart:
        resume_payload = load_resume_bundle(bundle_path, models=models, critic=critic,
                                            optimizer=optimizer, bc_models=bc_models)
        for dt in decision_types:
            if dt in resume_payload["kl_coefs"]:
                kl_controllers[dt].coef = float(resume_payload["kl_coefs"][dt])
        start_iter = int(resume_payload["iteration"])
        restore_rng(resume_payload["rng"])
        resumed = True

    # Opponent league setup (opt-in). Restored from the bundle on resume; otherwise
    # seeded with a frozen-BC base opponent. Snapshot files live under run_dir/league.
    league = None
    league_snapshot_every = int(league_cfg.get("snapshot_every", 25))
    if league_enabled:
        import random as _random

        from tichu_training.ppo.cotrain_league import CoTrainLeague
        from tichu_training.ppo.rollout_parallel import save_rollout_weights as _save_weights
        league_dir = run_dir / "league"
        league_dir.mkdir(parents=True, exist_ok=True)
        league_rng = _random.Random(pool_seed)
        if resume_payload is not None and resume_payload.get("league"):
            league = CoTrainLeague.from_state(
                resume_payload["league"], league_dir=league_dir, rng=league_rng)
        else:
            base_path = str(league_dir / "base_bc.pt")
            if not Path(base_path).exists():
                _save_weights(base_path, bc_models, critic)  # frozen-BC base opponent
            league = CoTrainLeague(
                league_dir, base_path,
                max_snapshots=int(league_cfg.get("max_snapshots", 3)), rng=league_rng)

    warmup_iters = int(config.get("critic", {}).get("warmup_iters", 0))
    if warmup_iters > 0 and not resumed:
        bc_policy = BatchedCoTrainPolicy(
            *(bc_models[dt] for dt in _NET_TYPES), critic,
            skill_decile=skill_decile, perfect_info=perfect_info, train_wish=cotrain_wish,
        )
        _critic_warmup(
            bc_policy, critic, _sample_positions,
            iters=warmup_iters, epochs=int(config.get("critic", {}).get("warmup_epochs", 3)),
            lr=float(ppo.get("critic_lr", 1e-3)), skill_decile=skill_decile,
            learner_team=learner_team, gamma=gamma, lam=lam, progress=progress,
        )

    snapshot_every = int(config.get("snapshot_every", 5))
    log_state = {"writer": None, "fh": None, "last_t": time.perf_counter()}

    def _on_iteration(iteration: int, stats: dict) -> None:
        now = time.perf_counter()
        row = {"iter": iteration, "wall_s": round(now - log_state["last_t"], 3),
               **{k: v for k, v in stats.items() if k != "iter"}}
        log_state["last_t"] = now
        if log_state["writer"] is None:
            fh = open(log_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(fh, fieldnames=log_fields, extrasaction="ignore", restval="")
            if log_path.stat().st_size == 0:
                writer.writeheader()
            log_state["fh"], log_state["writer"] = fh, writer
        log_state["writer"].writerow(row)
        log_state["fh"].flush()

        # Freeze the current learner into the league before checkpointing, so the
        # Resume Bundle's league state matches the files on disk.
        if league is not None and (iteration + 1) % league_snapshot_every == 0:
            league.snapshot(models, critic, iteration + 1)

        # Resume Bundle every iteration (atomic, keep-2): a kill loses <=1 iter.
        # The play anchor rides along only on re-anchoring runs (it's a full play
        # net; without it a resume would snap the anchor back to the warm start).
        save_resume_bundle(
            bundle_path, models=models, critic=critic, optimizer=optimizer,
            kl_coefs={dt: kl_controllers[dt].coef for dt in decision_types},
            iteration=iteration + 1, rng_state=capture_rng(),
            league=(league.state() if league is not None else None),
            play_anchor=(bc_models["play"].state_dict()
                         if reanchor_play_every else None),
        )
        # Serving snapshots every snapshot_every (for the offline `check` command).
        if (iteration + 1) % snapshot_every == 0:
            for dt in _NET_TYPES:
                save_checkpoint(models[dt], optimizer, step=iteration + 1,
                                path=str(snapshots_dir / f"iter_{iteration + 1:05d}_{dt}.bin"))

        if progress:
            print(
                f"iter {iteration + 1:>4}/{total_iters} | loss {stats['loss']:.3e} "
                f"v {stats['value_loss']:.3e} | "
                + " ".join(f"{dt[:2]} kl {stats[f'{dt}_kl']:.3f}" for dt in decision_types)
                + (f" | vine {stats['vine_rows']}" if "vine_rows" in stats else "")
                + f" | {row['wall_s']:.1f}s", flush=True,
            )
        if on_iteration is not None:
            on_iteration(iteration, stats)

    remaining = max(0, total_iters - start_iter)
    if progress:
        print(
            f"Co-training: target {total_iters} iters x M={positions_per_iter} "
            f"({'resumed at ' + str(start_iter) if resumed else 'fresh'}; running {remaining})\n"
            f"  run_dir: {run_dir}  (perfect_info={perfect_info}, "
            f"update_device={update_device}, rollout_workers={rollout_workers}, "
            f"league={'on/' + str(len(league.members())) + ' members' if league is not None else 'off'})",
            flush=True,
        )

    # Optional process-parallel rollout (ADR-0034 #1): the rollout is CPU-engine-bound,
    # so fan the M games across worker processes. workers<=1 keeps the single-process
    # path. The injected collector saves the live weights to a small file each iter and
    # dispatches chunks to a persistent spawn pool.
    parallel = None
    rollout_collect = None
    if rollout_workers > 1:
        from tichu_training.ppo.rollout_parallel import ParallelRollout, save_rollout_weights
        arch_cfg = {k: config.get(k, {}) for k in ("model", "schupfen_model", "call_model")}
        parallel = ParallelRollout(
            arch_cfg, critic_hidden=int(config.get("critic", {}).get("hidden", 512)),
            critic_depth=int(config.get("critic", {}).get("depth", 1)),
            skill_decile=skill_decile, perfect_info=perfect_info, workers=rollout_workers,
            train_wish=cotrain_wish,
        )
        weights_path = str(run_dir / "_rollout_weights.pt")

        def rollout_collect(iteration: int, positions):
            save_rollout_weights(weights_path, models, critic)
            opp = league.sample() if league is not None else None
            return parallel.collect(
                positions, weights_path, learner_team=learner_team,
                base_seed=pool_seed + iteration * positions_per_iter,
                opp_weights_path=opp,
            )

    # Vine play advantages (ADR-0035): dedicated deterministic vine games whose
    # paired-branch advantages REPLACE the play head's GAE group each iteration.
    # Parallel-only (the branch playouts are the same CPU-engine work as rollouts);
    # vine games draw from their own position-seed stream, disjoint from the main
    # rollout's, so the play data distribution isn't correlated with the GAE data.
    vine_cfg = config.get("vine", {})
    vine_collect = None
    if bool(vine_cfg.get("enabled", False)):
        if parallel is None:
            print("  vine requires rollout_workers>1 -> disabled", flush=True)
        else:
            vine_games = int(vine_cfg.get("games_per_iter", 64))
            vine_decisions = int(vine_cfg.get("decisions_per_game", 4))
            vine_branches = int(vine_cfg.get("branches", 4))
            vine_seed = int(vine_cfg.get("pool_seed", 555000))
            # Enrichment dials (v1 autopsy 2026-06-12): all-branch rows carry the
            # corrective direction at no extra playout cost; the |A| floor drops
            # near-tie rows that only dilute the play batch's normalization.
            vine_emit_branches = bool(vine_cfg.get("emit_branches", False))
            vine_min_abs_adv = float(vine_cfg.get("min_abs_advantage", 0.0))

            def vine_collect(iteration: int):
                # rollout_collect already saved this iteration's weights to
                # weights_path; no update happens in between, so reuse it.
                vpos = generate_full_position_pool(
                    seed=vine_seed + iteration * vine_games, n=vine_games
                )
                return parallel.collect_vine(
                    vpos, weights_path, decisions_per_game=vine_decisions,
                    branches=vine_branches,
                    base_seed=vine_seed + iteration * vine_games,
                    emit_branches=vine_emit_branches,
                    min_abs_advantage=vine_min_abs_adv,
                )

    try:
        history = train_cotrain(
            models, bc_models, critic, _sample_positions,
            optimizer=optimizer, kl_controllers=kl_controllers, ent_coefs=ent_coefs,
            iterations=remaining, gamma=gamma, lam=lam,
            clip_eps=float(ppo.get("clip_eps", 0.1)), vf_coef=float(ppo.get("vf_coef", 0.5)),
            ppo_epochs=int(ppo.get("ppo_epochs", 3)), skill_decile=skill_decile,
            learner_team=learner_team, perfect_info=perfect_info,
            on_iteration=_on_iteration, start_iter=start_iter,
            update_device=update_device, rollout_collect=rollout_collect,
            train_wish=cotrain_wish, vine_collect=vine_collect,
            reanchor_play_every=reanchor_play_every,
        )
    finally:
        if parallel is not None:
            parallel.close()
    if log_state["fh"] is not None:
        log_state["fh"].close()

    return {
        "history": history, "run_dir": str(run_dir), "start_iter": start_iter,
        "final_iter": start_iter + remaining, "bundle_path": bundle_path,
    }


def main(argv=None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description="Full-stack co-training PPO (ADR-0034)")
    parser.add_argument("--config", required=True, help="Path to the co-training config YAML.")
    parser.add_argument("--restart", action="store_true", help="Ignore any Resume Bundle; start fresh.")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    result = run_cotrain_training(config, restart=args.restart)
    print(
        f"co-training done: ran to iter {result['final_iter']} "
        f"(from {result['start_iter']}) -> bundle {result['bundle_path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
