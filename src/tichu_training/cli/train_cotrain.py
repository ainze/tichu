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
# Fixed CSV schema so the per-net dials (which vary by iteration — e.g. Tichu may
# not fire some iters) always line up across a stop/resume (ADR-0034 attribution).
_LOG_FIELDS = ["iter", "wall_s", "loss", "value_loss"] + [
    f"{dt}_{k}" for dt in _NET_TYPES for k in ("policy_loss", "kl", "entropy", "kl_coef")
]


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


def _kl_controllers(config) -> dict:
    kl = config.get("kl", {})
    out = {}
    for dt in _NET_TYPES:
        c = kl.get(dt, {})
        out[dt] = AdaptiveKLController(
            coef=float(c.get("coef", 1.0)),
            target=float(c.get("target", 0.02)),
            factor=float(c.get("factor", 2.0)),
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
    critic = ValueBaseline(critic_dim, hidden=int(config.get("critic", {}).get("hidden", 512)))

    optimizer = torch.optim.Adam(
        [{"params": models[dt].parameters(), "lr": float(ppo.get("policy_lr", 1e-4))} for dt in _NET_TYPES]
        + [{"params": critic.parameters(), "lr": float(ppo.get("critic_lr", 1e-3))}]
    )

    kl_controllers = _kl_controllers(config)
    ent_cfg = config.get("entropy", {})
    ent_coefs = {dt: float(ent_cfg.get(dt, 0.01)) for dt in _NET_TYPES}

    def _sample_positions(iteration: int):
        return generate_full_position_pool(
            seed=pool_seed + iteration * positions_per_iter, n=positions_per_iter
        )

    # Idempotent resume: a bundle in the run dir continues from the saved iteration;
    # absent (or --restart) -> fresh from the BC warm-starts (ADR-0034).
    start_iter = 0
    resumed = False
    if Path(bundle_path).exists() and not restart:
        payload = load_resume_bundle(bundle_path, models=models, critic=critic, optimizer=optimizer)
        for dt in _NET_TYPES:
            if dt in payload["kl_coefs"]:
                kl_controllers[dt].coef = float(payload["kl_coefs"][dt])
        start_iter = int(payload["iteration"])
        restore_rng(payload["rng"])
        resumed = True

    warmup_iters = int(config.get("critic", {}).get("warmup_iters", 0))
    if warmup_iters > 0 and not resumed:
        bc_policy = BatchedCoTrainPolicy(
            *(bc_models[dt] for dt in _NET_TYPES), critic,
            skill_decile=skill_decile, perfect_info=perfect_info,
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
            writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS, extrasaction="ignore", restval="")
            if log_path.stat().st_size == 0:
                writer.writeheader()
            log_state["fh"], log_state["writer"] = fh, writer
        log_state["writer"].writerow(row)
        log_state["fh"].flush()

        # Resume Bundle every iteration (atomic, keep-2): a kill loses <=1 iter.
        save_resume_bundle(
            bundle_path, models=models, critic=critic, optimizer=optimizer,
            kl_coefs={dt: kl_controllers[dt].coef for dt in _NET_TYPES},
            iteration=iteration + 1, rng_state=capture_rng(),
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
                + " ".join(f"{dt[:2]} kl {stats[f'{dt}_kl']:.3f}" for dt in _NET_TYPES)
                + f" | {row['wall_s']:.1f}s", flush=True,
            )
        if on_iteration is not None:
            on_iteration(iteration, stats)

    remaining = max(0, total_iters - start_iter)
    if progress:
        print(
            f"Co-training: target {total_iters} iters x M={positions_per_iter} "
            f"({'resumed at ' + str(start_iter) if resumed else 'fresh'}; running {remaining})\n"
            f"  run_dir: {run_dir}  (perfect_info={perfect_info})", flush=True,
        )

    history = train_cotrain(
        models, bc_models, critic, _sample_positions,
        optimizer=optimizer, kl_controllers=kl_controllers, ent_coefs=ent_coefs,
        iterations=remaining, gamma=gamma, lam=lam,
        clip_eps=float(ppo.get("clip_eps", 0.1)), vf_coef=float(ppo.get("vf_coef", 0.5)),
        ppo_epochs=int(ppo.get("ppo_epochs", 3)), skill_decile=skill_decile,
        learner_team=learner_team, perfect_info=perfect_info,
        on_iteration=_on_iteration, start_iter=start_iter,
    )
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
