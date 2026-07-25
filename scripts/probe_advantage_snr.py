"""Advantage-SNR probe: decompose the co-train critic's residual into ALEATORIC
(continuation-sampling variance no state-value function can remove) vs EPISTEMIC
(V(s) != E[R|s] — fixable critic error), by K-replaying the same state under the
training rollout distribution.

This is the "same-state stochastic K-replay variance probe" the 2026-06-18/19
critic diagnosis named as the decisive-but-unbuilt confirmation of the
"irreducible floor" claim (see memory: cotrain learning dynamics; handoff
2026-07-24). It decides between the two open strength levers:

  * epistemic share LOW  -> the critic is done; "fix the critic" is dead and the
    remaining EV must be delivered by a non-`R - V(s)` estimator (paired-playout /
    mine->distill family).
  * epistemic share HIGH (concentrated in a stratum) -> the critic still has
    fixable headroom there; attack critic inputs/loss/target first.

Two stages, both under the EXACT training distribution (BatchedCoTrainPolicy
sampling, learner_team=0, opponents alternating {champion, bc} like
train_cotrain's gate_opp_cycle):

  Stage A (round start): M held-out positions x K replicas through
    `collect_rollout`. Per position: E[R|s0], Var(R|s0) (within), deal-luck
    variance (between), and the critic's V at the first Schupfen step.
    Decomposition: MSE(V0) = epistemic (V0 - E[R|s0])^2 + aleatoric Var(R|s0).

  Stage B (mid-round): record fresh rounds one at a time with a recording
    wrapper, harvest learner-seat Play-decision states stratified by phase
    (open >= 11 / mid 6-10 / late <= 5 cards), then K-replay each state via
    `_GameRun` + `_drive` (the real rollout driver) and decompose per phase.

  Full-round outcomes R are team-0-relative with initial scores (0,0) — the same
  reward `collect_rollout` assigns (probed seats are 0/2 only). `asked_tichu` at
  a resume is exact from the state: a seat has made a non-Pass Play iff it holds
  fewer than 14 cards.

Usage (wishfix plateau learner, ~10-20 min CPU):

    py scripts/probe_advantage_snr.py \
        --config configs/cotrain_v6_pbrs_resid_wish_gated_wishfix.yaml \
        --run-dir C:/workbench/tichu/data/runs/cotrain_v6_pbrs_resid_wish_gated_wishfix \
        --out-dir C:/workbench/tichu/data/runs/advantage_snr_probe_v1 \
        [--m 256 --k 16 --m2 96 --k2 16 --states-per-round 3 --smoke]
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_ml.rule_agent import RuleAgent
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.cli.train_cotrain import _NET_TYPES, _build_models
from tichu_training.perfect_info import PERFECT_INFO_DIM, featurize_perfect_info
from tichu_training.ppo.cotrain import BatchedCoTrainPolicy
from tichu_training.ppo.rollout import _GameRun, _drive, collect_rollout

_NUM_PLAYERS = 4
_PHASES = ("open", "mid", "late")


def _phase(hand_size: int) -> str:
    return "open" if hand_size >= 11 else ("mid" if hand_size >= 6 else "late")


def _load_policy(config: dict, weights_path: str, *, generator) -> BatchedCoTrainPolicy:
    """Build a BatchedCoTrainPolicy from a rollout-weights blob ({'models', 'critic'}),
    exactly as train_cotrain / rollout_parallel load them."""
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    models = _build_models(config)
    for dt in _NET_TYPES:
        models[dt].load_state_dict(blob["models"][dt])
        models[dt].eval()
    critic = ValueBaseline(
        PERFECT_INFO_DIM,
        hidden=int(config["critic"]["hidden"]),
        depth=int(config["critic"].get("depth", 1)),
    )
    critic.load_state_dict(blob["critic"])
    critic.eval()
    return BatchedCoTrainPolicy(
        models["play"], models["schupfen"], models["tichu"], models["grand"], critic,
        skill_decile=int(config["ppo"].get("skill_decile", 9)),
        perfect_info=bool(config.get("perfect_info", True)),
        generator=generator,
        train_wish=bool(config.get("cotrain_wish", False)),
    )


@dataclass
class RecordedPlayState:
    """One learner-seat Play Decision harvested during Stage B recording."""

    game_idx: int
    opponent: str
    seat: int
    step_idx: int
    hand_size: int
    state: object       # GameState at the decision (immutable, pre-action)
    value: float        # critic V at this state (what GAE uses)
    logprob: float      # log-prob of the sampled action (sampling determinism stat)
    solver: dict | None = None  # per-seat Claim-Solver certainty bits (--solver-capture)


_SOLVER_MAX_HAND = 10  # forced_claim's perf guard: chains from >10 cards are implausible


def _solver_bits(state) -> dict:
    """Claim-Solver oracle bits for all four seats of a perfect-info GameState —
    what a train-time critic COULD be fed (it may cheat freely). `go_N` = a chain
    Guaranteed Out exists from seat N's hand vs its Unseen Cards (sound,
    worst-case); `rc_N` = the softer Reclaim Out. Hands > 10 cards are skipped
    (forced_claim's perf guard) and read False."""
    from tichu_engine.claim import guaranteed_out_chain, reclaim_out, unseen_cards

    out = {}
    pub = state.public
    for s in range(_NUM_PLAYERS):
        hand = state.hands[s]
        go = rc = False
        if 0 < len(hand) <= _SOLVER_MAX_HAND:
            unseen = unseen_cards(state.private_view(s))
            chain = guaranteed_out_chain(hand, unseen)
            go = chain is not None
            rc = go or (reclaim_out(hand, unseen) is not None)
        out[f"go_{s}"] = go
        out[f"rc_{s}"] = rc
        out[f"hand_{s}"] = len(hand)
    out["out_order"] = ",".join(str(s) for s in pub.out_order)
    out["callers"] = ",".join(
        sorted(f"{s}t" for s in pub.tichu_callers) + sorted(f"{s}g" for s in pub.grand_tichu_callers))
    return out


class _RecordingPolicy:
    """Wraps a BatchedCoTrainPolicy: forwards everything, records each Play
    Decision's (game_state, seat, V, logprob). Used one-game-at-a-time so the
    records of a call sequence all belong to that game in temporal order."""

    def __init__(self, inner: BatchedCoTrainPolicy):
        self._inner = inner
        self.play_records: list[tuple[object, int, float, float]] = []
        # rollout._supports_wish / _supports_calls probe with getattr(...)
        if getattr(inner, "act_wish_batch", None) is None:
            self.act_wish_batch = None

    def act_play_batch(self, decisions):
        choices = self._inner.act_play_batch(decisions)
        for (seat, _ps, gs), ch in zip(decisions, choices):
            self.play_records.append((gs, seat, float(ch.value), float(ch.logprob)))
        return choices

    def act_schupfen_batch(self, decisions):
        return self._inner.act_schupfen_batch(decisions)

    def act_wish_batch(self, decisions):
        return self._inner.act_wish_batch(decisions)

    def act_call_batch(self, kind, decisions):
        return self._inner.act_call_batch(kind, decisions)


def _team0_outcome(state) -> float:
    final = state.public.scores
    return float(final[0] - final[1])


def _asked_tichu_from_state(state) -> set[int]:
    """A seat has been solicited for Tichu iff it has made a non-Pass Play, i.e. it
    holds fewer than the post-Schupfen 14 cards (Pass removes none; every non-Pass
    Play removes >= 1). Exact for mid-round states after Schupfen."""
    return {s for s in range(_NUM_PLAYERS) if len(state.hands[s]) < 14}


def _var_decomposition(rewards_by_key: dict, values_by_key: dict) -> dict:
    """Unbiased decomposition over keys with K replicas each.

    within    = E_s[Var(R|s)]                (aleatoric — no V(s) can remove)
    between   = Var_s(E[R|s])                (deal/state luck — a perfect V removes it)
    epistemic = E_s[(V(s) - E[R|s])^2]       (fixable critic error)
    """
    within_terms, mean_terms, epi_terms, mse_terms = [], [], [], []
    for key, rs in rewards_by_key.items():
        rs = np.asarray(rs, dtype=np.float64)
        k = len(rs)
        if k < 2:
            continue
        m, v = rs.mean(), rs.var(ddof=1)
        within_terms.append(v)
        mean_terms.append(m)
        if key in values_by_key:
            val = values_by_key[key]
            # E[(V - Rbar)^2] = (V - E[R|s])^2 + Var(R|s)/k  -> subtract the bias.
            epi_terms.append((val - m) ** 2 - v / k)
            mse_terms.append(np.mean((val - rs) ** 2))
    within = float(np.mean(within_terms))
    means = np.asarray(mean_terms)
    between = float(means.var(ddof=1) - within / len(next(iter(rewards_by_key.values()))))
    out = {
        "n_states": len(within_terms),
        "within_var_aleatoric": within,
        "within_sd": within ** 0.5,
        "between_var": between,
    }
    if epi_terms:
        epistemic = float(np.mean(epi_terms))
        out.update(
            epistemic_var=epistemic,
            critic_mse=float(np.mean(mse_terms)),
            epistemic_share=epistemic / (epistemic + within),
        )
    return out


def stage_a(config, learner, opponents, *, m, k, pool_seed, chunk=1024) -> dict:
    """Round-start decomposition: M positions x K replicas, positions alternating
    over the opponent cycle (as training does per iteration)."""
    positions = generate_full_position_pool(seed=pool_seed, n=m)
    rewards: dict[int, list[float]] = {i: [] for i in range(m)}
    v0: dict[int, float] = {}
    opp_by_pos: dict[int, str] = {}
    total = m * k
    done = 0
    for oi, (opp_name, opp_policy) in enumerate(opponents):
        pos_idx = [i for i in range(m) if i % len(opponents) == oi]
        for i in pos_idx:
            opp_by_pos[i] = opp_name
        jobs = [i for i in pos_idx for _ in range(k)]
        at = 0
        while at < len(jobs):
            batch_idx = jobs[at : at + chunk]
            batch = [positions[i] for i in batch_idx]
            trajs = collect_rollout(batch, learner, opponent_policy=opp_policy, learner_team=0)
            for g, i in enumerate(batch_idx):
                traj = trajs[2 * g]  # two learner-seat trajs per game, in order
                rewards[i].append(float(traj.reward))
                if i not in v0:
                    for s in traj.steps:
                        if s.decision_type == "schupfen":
                            v0[i] = float(s.value)
                            break
            at += len(batch)
            done += len(batch)
            print(f"  stage A: {done}/{total} rollouts", flush=True)

    out = _var_decomposition(rewards, v0)
    out["opponent_split"] = {
        name: _var_decomposition(
            {i: rs for i, rs in rewards.items() if opp_by_pos[i] == name},
            {i: v for i, v in v0.items() if opp_by_pos[i] == name},
        )
        for name in dict(opponents)
    }
    return out


def stage_b_record(config, learner_inner, opponents, *, m2, pool_seed, states_per_round, rng,
                   solver_capture=False):
    """Record M2 fresh rounds one game at a time; harvest learner-seat Play states
    stratified by phase."""
    positions = generate_full_position_pool(seed=pool_seed, n=m2)
    harvested: list[RecordedPlayState] = []
    p_chosen: list[float] = []
    for gi, pos in enumerate(positions):
        opp_name, opp_policy = opponents[gi % len(opponents)]
        rec = _RecordingPolicy(learner_inner)
        collect_rollout([pos], rec, opponent_policy=opp_policy, learner_team=0)
        # stratify this round's records by phase, pick at most one per phase
        by_phase: dict[str, list[tuple]] = {p: [] for p in _PHASES}
        for si, (gs, seat, value, logprob) in enumerate(rec.play_records):
            hand = len(gs.hands[seat])
            by_phase[_phase(hand)].append((si, gs, seat, value, logprob, hand))
            p_chosen.append(float(np.exp(logprob)))
        chosen_phases = [p for p in _PHASES if by_phase[p]][:states_per_round]
        for p in chosen_phases:
            si, gs, seat, value, logprob, hand = by_phase[p][
                rng.integers(len(by_phase[p]))
            ]
            harvested.append(
                RecordedPlayState(
                    game_idx=gi, opponent=opp_name, seat=seat, step_idx=si,
                    hand_size=hand, state=gs, value=value, logprob=logprob,
                    solver=_solver_bits(gs) if solver_capture else None,
                )
            )
        if (gi + 1) % 16 == 0:
            print(f"  stage B record: {gi + 1}/{m2} rounds, {len(harvested)} states", flush=True)
    return harvested, p_chosen


def stage_b_replay(config, learner, opponents, harvested, *, k2, chunk=1536) -> pd.DataFrame:
    """K-replay each harvested state through the real rollout driver."""
    opp_map = dict(opponents)
    rows = []
    jobs = [(h, r) for h in harvested for r in range(k2)]
    results: dict[int, list[float]] = {i: [] for i in range(len(harvested))}
    idx_of = {id(h): i for i, h in enumerate(harvested)}
    done = 0
    while done < len(jobs):
        batch = jobs[done : done + chunk]
        by_opp: dict[str, list] = {}
        for h, _r in batch:
            by_opp.setdefault(h.opponent, []).append(h)
        for opp_name, hs in by_opp.items():
            runs = [
                _GameRun(
                    state=h.state,
                    seat_agents=[RuleAgent() for _ in range(_NUM_PLAYERS)],
                    learner_team=0,
                    initial_scores=(0, 0),
                    trajs={},
                    asked_tichu=_asked_tichu_from_state(h.state),
                )
                for h in hs
            ]
            _drive(runs, learner, opp_map[opp_name], 0)
            for h, run in zip(hs, runs):
                results[idx_of[id(h)]].append(_team0_outcome(run.state))
        done += len(batch)
        print(f"  stage B replay: {done}/{len(jobs)} playouts", flush=True)

    for i, h in enumerate(harvested):
        rs = np.asarray(results[i], dtype=np.float64)
        pub = h.state.public
        rows.append(
            {
                "game_idx": h.game_idx, "opponent": h.opponent, "seat": h.seat,
                "step_idx": h.step_idx, "hand_size": h.hand_size,
                "phase": _phase(h.hand_size), "value": h.value,
                "p_chosen": float(np.exp(h.logprob)),
                # state descriptors for residual diagnostics
                "banked_diff": float(pub.scores[0] - pub.scores[1]),
                "n_out": len(pub.out_order),
                "caller_live": bool(pub.tichu_callers or pub.grand_tichu_callers),
                "caller_team0": bool(
                    {0, 2} & (set(pub.tichu_callers) | set(pub.grand_tichu_callers))
                ),
                # frozen benchmark input: score any future candidate critic against
                # (critic_features -> r_mean) with no replays needed
                "critic_features": featurize_perfect_info(h.state, h.seat).tolist(),
                **(h.solver or {}),
                "k": len(rs), "r_mean": rs.mean(),
                "r_var": rs.var(ddof=1), "r_sd": rs.std(ddof=1),
                "replays": json.dumps([round(float(x), 1) for x in rs]),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_share(sub: pd.DataFrame, iters: int = 2000, seed: int = 1) -> tuple[float, float]:
    """Bootstrap 95% CI over states for the epistemic share."""
    rng = np.random.default_rng(seed)
    epi = ((sub.value - sub.r_mean) ** 2 - sub.r_var / sub.k).to_numpy()
    ale = sub.r_var.to_numpy()
    shares = []
    n = len(sub)
    for _ in range(iters):
        idx = rng.integers(n, size=n)
        e, a = epi[idx].mean(), ale[idx].mean()
        shares.append(e / (e + a))
    return float(np.percentile(shares, 2.5)), float(np.percentile(shares, 97.5))


def summarize_stage_b(df: pd.DataFrame) -> dict:
    out = {}
    slices: list[tuple[str, pd.DataFrame]] = [(p, df[df.phase == p]) for p in _PHASES]
    slices += [
        ("all", df),
        ("caller_live", df[df.caller_live]),
        ("no_caller", df[~df.caller_live]),
        ("someone_out", df[df.n_out > 0]),
    ]
    for name, sub in slices:
        if len(sub) < 8:
            continue
        rewards = {i: json.loads(r) for i, r in zip(sub.index, sub.replays)}
        values = {i: v for i, v in zip(sub.index, sub.value)}
        d = _var_decomposition(rewards, values)
        d["epistemic_share_ci95"] = _bootstrap_share(sub)
        resid2 = ((sub.value - sub.r_mean) ** 2).to_numpy()
        d["resid2_median"] = float(np.median(resid2))
        d["resid2_p90"] = float(np.percentile(resid2, 90))
        out[name] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True, help="cotrain run dir (weights + opponents)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights", default=None, help="default: <run-dir>/_rollout_weights.pt")
    ap.add_argument("--m", type=int, default=256, help="stage A positions")
    ap.add_argument("--k", type=int, default=16, help="stage A replicas per position")
    ap.add_argument("--m2", type=int, default=96, help="stage B rounds to record")
    ap.add_argument("--k2", type=int, default=16, help="stage B replays per state")
    ap.add_argument("--states-per-round", type=int, default=3)
    ap.add_argument("--pool-seed", type=int, default=9_000_000, help="held-out (disjoint from training + gate)")
    ap.add_argument("--seed", type=int, default=0, help="sampling RNG seed")
    ap.add_argument("--stage", choices=["a", "b", "both"], default="both")
    ap.add_argument("--solver-capture", action="store_true",
                    help="capture per-seat Claim-Solver certainty bits per harvested state")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes for a fast wiring check")
    args = ap.parse_args()
    if args.smoke:
        args.m, args.k, args.m2, args.k2, args.states_per_round = 8, 4, 4, 4, 2

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    weights = args.weights or str(run_dir / "_rollout_weights.pt")

    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    learner = _load_policy(config, weights, generator=gen)
    # Rollout opponents mirror train_cotrain's gate_opp_cycle alternation
    # (iteration % len(cycle)): the rolling champion and the frozen BC.
    opponents = [
        ("champion", _load_policy(config, str(run_dir / "_champion.pt"), generator=gen)),
        ("bc", _load_policy(config, str(run_dir / "_bc_opponent.pt"), generator=gen)),
    ]

    summary: dict = {"weights": weights, "args": vars(args)}
    if args.stage in ("a", "both"):
        print(f"stage A: {args.m} positions x {args.k} replicas ...", flush=True)
        summary["stage_a"] = stage_a(
            config, learner, opponents, m=args.m, k=args.k, pool_seed=args.pool_seed,
        )
        print(json.dumps(summary["stage_a"], indent=2, default=float), flush=True)

    if args.stage in ("b", "both"):
        print(f"stage B: recording {args.m2} rounds ...", flush=True)
        harvested, p_chosen = stage_b_record(
            config, learner, opponents, m2=args.m2, pool_seed=args.pool_seed + 500_000,
            states_per_round=args.states_per_round, rng=rng,
            solver_capture=args.solver_capture,
        )
        print(f"stage B: {len(harvested)} states harvested; replaying x{args.k2} ...", flush=True)
        df = stage_b_replay(config, learner, opponents, harvested, k2=args.k2)
        df.to_parquet(out_dir / "stage_b_states.parquet", index=False)
        summary["stage_b"] = summarize_stage_b(df)
        summary["p_chosen_median_all_play"] = float(np.median(p_chosen))
        summary["p_chosen_p25_all_play"] = float(np.percentile(p_chosen, 25))
        print(json.dumps(summary["stage_b"], indent=2, default=float), flush=True)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwritten: {out_dir}")

    a = summary.get("stage_a", {})
    if a:
        print("\n=== READ ===")
        print(f"round-start: aleatoric Var(R|s0) = {a['within_var_aleatoric']:.0f} "
              f"(sd {a['within_sd']:.0f}), deal-luck between-var = {a['between_var']:.0f}")
        if "epistemic_share" in a:
            print(f"critic@s0: MSE = {a['critic_mse']:.0f}, epistemic = {a['epistemic_var']:.0f} "
                  f"-> epistemic share = {a['epistemic_share']:.1%}")
            print("LOW share (<~15%) -> critic done at s0; HIGH (>~30%) -> fixable headroom.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
