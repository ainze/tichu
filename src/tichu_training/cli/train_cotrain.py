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


# The arch blocks `_build_models` reads. Every place that ships an arch sub-config
# to a worker / gate / opponent rebuild MUST forward all of these — a partial copy
# silently drops `grand_model` and rebuilds grand with the residual `call_model`
# arch, crashing the warm-start load. Keep this the single source of truth.
_ARCH_CFG_KEYS = ("model", "schupfen_model", "call_model", "grand_model")


def _arch_cfg(config) -> dict:
    return {k: config.get(k, {}) for k in _ARCH_CFG_KEYS}


def _build_models(config) -> dict:
    m = config.get("model", {})
    sm = config.get("schupfen_model", {})
    cm = config.get("call_model", {})
    # Grand defaults to the tichu/call arch (back-compat: configs with only
    # `call_model` keep grand == tichu shape). An explicit `grand_model` block
    # lets tichu carry a residual trunk (PR #64) while grand stays the small
    # non-residual MLP its capacity probe showed is irreducible.
    # `or cm` (not a `.get` default): `_arch_cfg` materialises EVERY arch key, so a
    # config without a grand block reaches the workers as `grand_model: {}` — with a
    # plain default that empty dict would win and grand would rebuild at the library
    # defaults, mismatching the main process's call_model-shaped grand on load.
    gm = config.get("grand_model") or cm
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
            depth=int(sm.get("depth", 4)),
            residual=bool(sm.get("residual", False)),
        ),
        "tichu": TichuCallNetwork(
            FEATURIZER_OUTPUT_DIM, skill_dim=int(cm.get("skill_dim", 64)),
            hidden=int(cm.get("hidden", 256)),
            depth=int(cm.get("depth", 4)),
            residual=bool(cm.get("residual", False)),
        ),
        "grand": GrandTichuCallNetwork(
            FEATURIZER_OUTPUT_DIM, skill_dim=int(gm.get("skill_dim", 64)),
            hidden=int(gm.get("hidden", 256)),
            depth=int(gm.get("depth", 4)),
            residual=bool(gm.get("residual", False)),
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


def _append_gate_row(path: Path, iteration: int, v: dict) -> None:
    """Append a promotion-gate verdict to `promotion_gate.csv` — one row per opponent
    (the run's record of when the champion advanced and on what margin vs each)."""
    fresh = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if fresh:
            w.writerow(["iter", "opponent", "n", "mean", "ci_lo", "ci_hi", "promoted"])
        for opp, s in v["opponents"].items():
            w.writerow([iteration, opp, s["n"], f"{s['mean']:.4f}", f"{s['ci_lo']:.4f}",
                        f"{s['ci_hi']:.4f}", int(v["promote"])])


def _append_gate_components(path: Path, iteration: int, v: dict) -> None:
    """Append the per-opponent margin DECOMPOSITION to `promotion_gate_components.csv`.

    Deliberately a separate file rather than extra columns on `promotion_gate.csv`:
    that file already exists mid-run for every live run, its header is written once,
    and appending wider rows under a 7-column header would corrupt it for
    `plot_cotrain_csv` and every other reader. NB nothing plots this yet.
    """
    rows = [
        (opp, name, s)
        for opp, st in v["opponents"].items()
        for name, s in (st.get("components") or {}).items()
    ]
    if not rows:
        return
    fresh = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if fresh:
            w.writerow(["iter", "opponent", "component", "n", "mean", "se",
                        "ci_lo", "ci_hi", "promoted"])
        for opp, name, s in rows:
            w.writerow([iteration, opp, name, v["opponents"][opp]["n"],
                        f"{s['mean']:.4f}", f"{s['se']:.4f}",
                        f"{s['ci_lo']:.4f}", f"{s['ci_hi']:.4f}", int(v["promote"])])


def _gate_logged_iters(path: Path) -> set[int]:
    """Iters that already have verdict rows in `promotion_gate.csv`. Used on resume to
    detect a greedy-gate window killed mid-tournament: the Resume Bundle is saved
    BEFORE the gate runs, so on restart the boundary is already behind the loop and
    its window would silently be skipped."""
    if not path.exists():
        return set()
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return set()
    out: set[int] = set()
    for row in rows[1:]:
        try:
            out.add(int(row[0]))
        except (ValueError, IndexError):
            continue
    return out


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
    # Forced Play Decisions (|legal| == 1, ~38% of them) carry no decision and
    # contribute exactly zero policy gradient. Dropping them from the rollout is
    # DEFAULT ON: cheaper iterations, and no forced step lengthening a
    # trajectory. It does shift the play advantage-normalisation stats (and, at
    # lam<1, credit assignment), so a run that must reproduce pre-2026-08-01
    # numbers has to set this false explicitly.
    skip_forced_play = bool(ppo.get("skip_forced_play", True))

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

    # Champion promotion gate (rounds-as-gate ratchet): the rollout opponent becomes
    # a FROZEN champion, so each iter's learner round_outcomes ARE the candidate-vs-
    # champion margin; the gate accumulates them and promotes the candidate to
    # champion on a CI-validated improvement (PromotionGate). The principled,
    # VALIDATED sibling of `reanchor_play_every`. Parallel-only (uses the worker
    # opp_weights mechanism); takes precedence over the league as the opponent. The
    # champion file persists on disk, so the ratchet survives restarts (the in-memory
    # margin window resets on resume — at most one window is re-accumulated).
    gate = None
    champion_path = None
    gate_opp_cycle = None   # [(opponent_name, weights_path)] the rollout alternates over
    gate_eval_cycle = None  # gate_opp_cycle + observe-only extras — what the greedy eval scores
    gate_log_path = run_dir / "promotion_gate.csv"
    _save_champion = None
    reanchor_on_promote = False
    gate_cfg = config.get("promotion_gate", {})
    if bool(gate_cfg.get("enabled", False)):
        if rollout_workers <= 1:
            if progress:
                print("  promotion_gate requires rollout_workers>1 -> disabled", flush=True)
        else:
            # reanchor_on_promote: on a validated promotion, also move the PLAY KL
            # anchor onto the new champion (= current net), so the learner can leave
            # the BC ball and keep improving. Without it the gate ratchets only the
            # OPPONENT and the learner stays capped at 0.02 from the ORIGINAL BC — it
            # stalls at the in-ball optimum. With it, the gate is a VALIDATED escape:
            # the anchor advances only on a CI-confirmed win (vs the ungated
            # reanchor_play_every trail, which moved blindly and regressed).
            reanchor_on_promote = bool(gate_cfg.get("reanchor_on_promote", False))
            # also_beat_bc: add the FROZEN BC as a second opponent the candidate must
            # also beat to promote (anti-cycling). The rollout alternates champion/BC
            # per iter; a promotion needs a CI-validated win over BOTH over their
            # respective windows. BC is an OPPONENT here, not the KL anchor — so the
            # policy diverges from BC freely (anchor follows the champion) but must
            # still out-SCORE BC, rejecting champion-beating exploits that don't hold
            # up vs the fixed human reference.
            also_beat_bc = bool(gate_cfg.get("also_beat_bc", False))
            # greedy: source the gate's margins from a GREEDY mini-tournament (the
            # deployed MLAgent via run_full_tournament) instead of the SAMPLED rollout
            # reward. The sampled margin inflates as the policy sharpens (low-entropy
            # learner barely hurt by sampling; the BC opponent is) — an artifact that
            # drove 201 phantom promotions while DEPLOYED strength fell (2026-06-18
            # close). The greedy margin is exactly what check_cotrain measures, so the
            # gate validates deployed strength. verdict/promote/reanchor are unchanged;
            # only the margin SOURCE differs (see ppo/greedy_gate.py).
            gate_greedy = bool(gate_cfg.get("greedy", False))
            greedy_n_deals = int(gate_cfg.get("greedy_n_deals", 2048))
            greedy_every = int(gate_cfg.get("greedy_every", 64))
            greedy_workers = int(gate_cfg.get("greedy_workers", rollout_workers))
            gate_arch_cfg = _arch_cfg(config)
            from tichu_training.ppo.promotion_gate import PromotionGate
            from tichu_training.ppo.rollout_parallel import save_rollout_weights as _save_champion
            champion_path = str(run_dir / "_champion.pt")
            if not Path(champion_path).exists():
                _save_champion(champion_path, bc_models, critic)  # champion := frozen BC
            gate_opp_cycle = [("champion", champion_path)]
            opponents = ["champion"]
            if also_beat_bc:
                bc_opp_path = str(run_dir / "_bc_opponent.pt")
                if not Path(bc_opp_path).exists():
                    _save_champion(bc_opp_path, bc_models, critic)  # frozen BC opponent
                gate_opp_cycle.append(("bc", bc_opp_path))
                opponents.append("bc")
            # extra_opponents: additional FIXED external models (e.g. the current
            # shipped champion). Each entry is `{name, path[, require]}` where path is
            # a rollout-weights .pt ({"models": {net: state_dict}}) whose arch matches
            # THIS run's _arch_cfg (so both the rollout worker's _league_opponent and
            # the greedy gate's export_opponent rebuild it with
            # _build_models(gate_arch_cfg)). Unlike `bc`, these never advance (fixed
            # reference points). Two modes per entry:
            #   require: true (default) — joins the cycle exactly like champion/bc:
            #     the rollout best-responds to it in turn AND promotion needs a
            #     CI-validated win over it (beat-ALL).
            #   require: false — OBSERVE-ONLY: the greedy gate scores it every window
            #     (its margin lands in promotion_gate.csv → plot panel 6) but it is
            #     excluded from the promote decision AND from the rollout opponent
            #     cycle, so it never hard-blocks the ratchet and never shifts the
            #     training distribution. Greedy gate only (a sampled window would
            #     never fill an un-rolled-out stream).
            gate_observe_only: list[str] = []
            gate_eval_extras: list[tuple[str, str]] = []
            for spec in gate_cfg.get("extra_opponents", []) or []:
                name = str(spec["name"])
                path = str(spec["path"])
                if name in opponents:
                    raise ValueError(f"duplicate gate opponent name {name!r}")
                if not Path(path).exists():
                    raise FileNotFoundError(
                        f"extra_opponent {name!r} weights not found: {path}")
                gate_eval_extras.append((name, path))
                opponents.append(name)
                if bool(spec.get("require", True)):
                    gate_opp_cycle.append((name, path))
                else:
                    gate_observe_only.append(name)
            if gate_observe_only and not gate_greedy:
                raise ValueError(
                    "extra_opponents with require:false need the greedy gate "
                    f"(sampled windows never fill an observe-only stream): {gate_observe_only}")
            # The greedy eval scores EVERY opponent (required + observe-only); the
            # rollout alternates over gate_opp_cycle (required only).
            gate_eval_cycle = gate_opp_cycle + [
                e for e in gate_eval_extras if e[0] in gate_observe_only]
            # A greedy window is ONE mini-tournament => 2*n_deals seat-swapped obs per
            # opponent, filled in a single eval (so gate.ready() trips immediately after
            # it); the sampled window instead accumulates positions_per_iter per iter.
            window_games = (2 * greedy_n_deals if gate_greedy
                            else int(gate_cfg.get("window_games", positions_per_iter * 8)))
            # Seat-swap cluster bootstrap. The greedy window is 2*n_deals values with
            # the two arrangements of each deal ADJACENT; resampling them flat counts
            # card luck the swap already cancelled. Greedy-only — the sampled stream
            # is one margin per game and has no pairs to cluster.
            gate_paired = bool(gate_cfg.get("paired", False))
            if gate_paired and not gate_greedy:
                raise ValueError(
                    "promotion_gate.paired requires greedy:true — the sampled stream "
                    "records one margin per game, so there are no seat-swap pairs to "
                    "cluster and the reshape would pair unrelated games")
            gate = PromotionGate(
                opponents=tuple(opponents),
                window_games=window_games,
                threshold=float(gate_cfg.get("threshold", 0.0)),
                bootstrap_iters=int(gate_cfg.get("bootstrap_iters", 1000)),
                seed=pool_seed,
                observe_only=tuple(gate_observe_only),
                pooled=bool(gate_cfg.get("pooled", False)),
                paired=gate_paired,
                # 0 = unbounded (ADR-0040). A cap bounds how long a bad patch keeps
                # poisoning the pooled estimate: the learner is NOT stationary over a
                # long pooling stretch, so an early dip can take thousands of
                # iterations to average out even after the policy recovers.
                pool_windows=int(gate_cfg.get("pool_windows", 0)),
            )
            if progress:
                src = (f"GREEDY mini-tournament ({greedy_n_deals} deals/opp every "
                       f"{greedy_every} iters, {greedy_workers}w)" if gate_greedy
                       else "sampled rollout reward")
                if gate_observe_only:
                    src += f", observe-only (never required): {gate_observe_only}"
                print(f"  promotion_gate ON: opponents={opponents}, window {gate.window_games} games/opp, "
                      f"threshold {gate.threshold:+g}, reanchor_on_promote={reanchor_on_promote}, "
                      f"margin source = {src}"
                      + (" (league opponent ignored)" if league is not None else ""), flush=True)

    # ADR-0040 early validation (before the worker pool spins up): the Reference
    # Field is the gate's champion file, so a vine reference without the gate has
    # neither a file to read nor a promotion to ever advance it.
    _vine_reference = str(config.get("vine", {}).get("reference", "") or "")
    if _vine_reference and _vine_reference != "champion":
        raise ValueError(
            f"vine.reference must be 'champion' or absent, got {_vine_reference!r}")
    if _vine_reference == "champion" and gate is None:
        raise ValueError(
            "vine.reference: champion requires promotion_gate.enabled — the gate's "
            "champion file IS the Reference Field and only promotions advance it")

    # Restore a pooled gate's accumulated margins. Without this the gate rebuilds
    # empty on every resume, silently discarding the accumulation the pooled ratchet
    # exists to build (5-20 windows = 640-2560 iters at greedy_every=128). Bundles
    # written before this existed carry no "gate" key and simply start empty.
    if gate is not None and resume_payload is not None and resume_payload.get("gate"):
        gate.load_state(resume_payload["gate"])
        if progress and any(gate.n(o) for o in gate.opponents):
            depth = "  ".join(f"{o}={gate.n(o)}" for o in gate.opponents)
            print(f"  gate pool restored from bundle: {depth} "
                  f"({'pooled' if gate.pooled else 'partial window'})", flush=True)

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

    def _run_greedy_window(boundary: int) -> None:
        """One greedy-gate mini-tournament for the window ending at iter `boundary`
        (deals seeded by the boundary, so a re-run reproduces the same window)."""
        from tichu_training.ppo.greedy_gate import record_greedy_window
        eval_idx = boundary // greedy_every
        eval_positions = generate_full_position_pool(
            seed=1_000_000_000 + eval_idx * greedy_n_deals, n=greedy_n_deals
        )
        if progress:
            print(f"  greedy gate eval @ iter {boundary}: starting {greedy_n_deals} "
                  f"deals x2 vs {[o for o, _ in gate_eval_cycle]} ({greedy_workers}w)...",
                  flush=True)
        record_greedy_window(
            gate, models, gate_eval_cycle, arch_cfg=gate_arch_cfg,
            positions=eval_positions, skill_decile=skill_decile,
            workers=greedy_workers, export_root=run_dir / "_gate_export",
        )
        if progress:
            # Report the POOL DEPTH each opponent now carries, not just "done". Under
            # `pooled` the verdict is drawn on everything accumulated since the last
            # promotion, so the depth (and how many windows it represents) is what
            # explains the CI width — a hold at 8192 games and a hold at 40960 are
            # very different states and used to look identical in the log.
            depth = "  ".join(
                f"{o}={gate.n(o)}({gate.n(o) / gate.window_games:.1f}w)"
                for o in gate.opponents)
            print(f"  greedy gate eval @ iter {boundary}: done — pooled games: {depth}"
                  if gate.pooled else
                  f"  greedy gate eval @ iter {boundary}: done — games: {depth}",
                  flush=True)

    def _gate_verdict(boundary: int) -> None:
        """Draw the promotion verdict for the window ending at iter `boundary` once
        full, log it, and on a CI-validated win promote (champion file + serving
        snapshot + optional play re-anchor)."""
        if not gate.ready():
            return
        v = gate.verdict()
        _append_gate_row(gate_log_path, boundary, v)
        _append_gate_components(run_dir / "promotion_gate_components.csv", boundary, v)
        if v["promote"]:
            _save_champion(champion_path, models, critic)
            # Serving snapshot AT the promotion iter so the champion is directly
            # check_cotrain-able: the regular periodic snapshots are the LEARNER,
            # which wanders after a promotion, so the nearest one understates the
            # champion. Same naming, so it coincides cleanly with a periodic save.
            for dt in _NET_TYPES:
                save_checkpoint(models[dt], optimizer, step=boundary,
                                path=str(snapshots_dir / f"iter_{boundary:05d}_{dt}.bin"))
            if reanchor_on_promote:
                # Validated escape: the KL anchor follows the new champion, so the
                # learner can move further from the original BC next window.
                bc_models["play"].load_state_dict(models["play"].state_dict())
        if progress:
            # n= is the games the CI was actually drawn on. Under `pooled` that grows
            # across held windows, so it is the number that explains a shrinking CI.
            def _opp(o, s):
                n = f"n={s['n']}"
                if gate.pooled:
                    n += f"/{s['n'] / gate.window_games:.1f}w"
                return f"{o} {n} {s['mean']:+.1f}[{s['ci_lo']:+.1f},{s['ci_hi']:+.1f}]"

            opp_str = "  ".join(_opp(o, s) for o, s in v["opponents"].items())
            print(f"  gate {'PROMOTE' if v['promote'] else 'hold'} @ iter {boundary}: "
                  f"{opp_str}", flush=True)
            # Decomposition line: a total near zero can be a large card-play gain
            # cancelled by a large call-bonus loss. Diagnostic only — the verdict
            # above is drawn on the total.
            for o, s in v["opponents"].items():
                comps = s.get("components") or {}
                if comps:
                    parts = "  ".join(
                        f"{n}={c['mean']:+.2f}+/-{1.96 * c['se']:.2f}"
                        for n, c in comps.items())
                    print(f"       {o}: {parts}", flush=True)
        gate.conclude(v)  # Pooled Verdict: the pool survives holds, clears on promotion

    # Resume catch-up: the Resume Bundle is written BEFORE the gate window runs, so
    # a kill DURING the greedy mini-tournament resumes with start_iter already at the
    # boundary — the in-loop modulo check never re-fires and the window would be
    # silently skipped. Detect the hole (boundary reached, no verdict rows for it in
    # promotion_gate.csv) and re-run that window first: same learner weights (the
    # bundle at the boundary) and same deal seed, so it reproduces the killed eval.
    if (resumed and gate is not None and gate_greedy and start_iter > 0
            and start_iter % greedy_every == 0
            and start_iter not in _gate_logged_iters(gate_log_path)):
        print(f"resume: greedy gate window @ iter {start_iter} was interrupted by the "
              f"shutdown — running it first, then resuming training", flush=True)
        _run_greedy_window(start_iter)
        _gate_verdict(start_iter)

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
                         if (reanchor_play_every or reanchor_on_promote) else None),
            # The accumulated gate pool, so a pooled ratchet survives a restart.
            # NB this bundle is deliberately written BEFORE this iteration's gate
            # window runs (see the resume catch-up above), so the persisted pool is
            # the PRE-window state. That pairs correctly with the catch-up: a kill
            # DURING the mini-tournament resumes with the pre-window pool and re-runs
            # the window, re-adding its margins exactly once. The residual gap is a
            # kill in the moment between a concluded verdict and the next iteration's
            # save — that loses one window's margins from the pool, but never
            # double-counts and never loses an already-logged promotion.
            gate=(gate.state() if gate is not None else None),
        )
        # Serving snapshots every snapshot_every (for the offline `check` command).
        if (iteration + 1) % snapshot_every == 0:
            for dt in _NET_TYPES:
                save_checkpoint(models[dt], optimizer, step=iteration + 1,
                                path=str(snapshots_dir / f"iter_{iteration + 1:05d}_{dt}.bin"))

        # Greedy gate: every greedy_every iters, run ONE mini-tournament of the current
        # learner vs each pool opponent (the deployed greedy MLAgent), recording the
        # per-deal margins — the deployed-strength signal that replaces the confounded
        # sampled reward. A single eval fills every opponent's window, so the verdict
        # below fires right after. Fresh, unseen deals each window (base 1e9, far above
        # the rollout/vine seed streams) so the gate can't overfit a fixed eval set.
        if gate is not None and gate_greedy and (iteration + 1) % greedy_every == 0:
            _run_greedy_window(iteration + 1)

        # Champion promotion gate verdict: this iter's per-game margins were recorded
        # in rollout_collect (sampled) or by the greedy eval above; once a full window
        # has accumulated, draw a verdict and — on a CI-validated win — overwrite the
        # champion file with the current nets, then reset for the next window.
        if gate is not None:
            _gate_verdict(iteration + 1)

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
        arch_cfg = _arch_cfg(config)
        parallel = ParallelRollout(
            arch_cfg, critic_hidden=int(config.get("critic", {}).get("hidden", 512)),
            critic_depth=int(config.get("critic", {}).get("depth", 1)),
            skill_decile=skill_decile, perfect_info=perfect_info, workers=rollout_workers,
            train_wish=cotrain_wish, skip_forced_play=skip_forced_play,
        )
        weights_path = str(run_dir / "_rollout_weights.pt")

        def rollout_collect(iteration: int, positions):
            save_rollout_weights(weights_path, models, critic)
            # Champion gate takes precedence over the league as the opponent: playing
            # a frozen opponent is what makes the learner's round_outcomes a
            # candidate-vs-opponent margin (self-play / a random league sample would
            # average to ~0 and carry no promotion signal). With also_beat_bc the
            # cycle alternates champion/BC per iter so each accumulates its own window.
            if gate is not None:
                opp_name, opp = gate_opp_cycle[iteration % len(gate_opp_cycle)]
            else:
                opp_name, opp = None, (league.sample() if league is not None else None)
            trajs = parallel.collect(
                positions, weights_path, learner_team=learner_team,
                base_seed=pool_seed + iteration * positions_per_iter,
                opp_weights_path=opp,
            )
            if gate is not None and not gate_greedy:
                # SAMPLED margin: one per game = the learner team's round_outcome at its
                # representative seat (== learner_team) so each game counts once. Skipped
                # under the greedy gate — the margin comes from the mini-tournament in
                # _on_iteration, not the sampled rollout reward (the confounded signal).
                gate.record(opp_name, (t.reward for t in trajs if t.seat == learner_team))
            return trajs

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
            vine_stratify = bool(vine_cfg.get("stratify", False))
            # Reference Field (ADR-0040): branch continuations play under the
            # gate's champion file — frozen between promotions, advanced only
            # when a promotion rewrites the file (validated above).
            vine_ref_path = (str(run_dir / "_champion.pt")
                             if _vine_reference == "champion" else None)

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
                    ref_weights_path=vine_ref_path,
                    stratify=vine_stratify,
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
            skip_forced_play=skip_forced_play,
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
