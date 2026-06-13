"""pMCPA: per-Round run-time policy adaptation (ADR-0036).

Suphx's parametric Monte-Carlo Policy Adaptation, rebuilt on this project's
paired-advantage estimator instead of its high-variance importance-weighted
gradient. At a Round's first Play Decision the acting Player:

  1. samples K Determinized Worlds consistent with its own Hand (belief baked
     into the sampling, no explicit belief state);
  2. plays each world out with the FROZEN offline policy theta_o on all four
     seats, recording its OWN Play Decisions, and computes within-world
     luck-cancelled paired advantages (the vine estimator, ADR-0035);
  3. takes a few policy-gradient steps -- contained by a KL-anchor back to
     theta_o -- to a throwaway adapted play net theta_a;
  4. plays the real Round with theta_a, then DISCARDS it and restores theta_o.

The frozen-theta_o field is load-bearing: it never moves across Rounds, so the
self-play co-drift that killed re-anchored vine (ADR-0035, -23.59) is structurally
impossible here. And because the correction is applied to the Round it was computed
on and never distilled, pMCPA sidesteps vine's cross-Round generalization failure.

Two dangers hide in "small-K overfit": (a) gradient variance -- killed by the
paired estimator; (b) world-sampling bias -- the K worlds are a biased draw and
theta_a is applied to the one true world not in the sample. The paired estimator
does nothing for (b); only more worlds, belief-weighting, or a tighter mid-round
re-sample help. (b) is what the K-sweep actually probes.
"""

import copy
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch

from tichu_engine.engine import step
from tichu_engine.legality import Pass, legal_actions_for
from tichu_eval.play_full import FullRoundResult, _calls, _with_callers
from tichu_ml.agent import Agent
from tichu_training.ppo.update import (
    clipped_policy_loss,
    kl_anchor_loss,
    _masked_logp_at,
)
from tichu_training.ppo.vine import _vine_rows, vine_net_batch
from tichu_training.search.blunder_miner import RecordedDecision, team_relative
from tichu_training.search.determinize import sample_determinized_world

_MAX_STEPS = 10_000

# The iter_06225 (theta_o) architecture — the dims its recovered .bin weights were
# trained at (configs/cotrain_vine_v1.yaml). The launcher builds raw BCModel /
# standalone modules at this arch and loads the .bin state dicts into them; the
# TorchScript .pt export cannot be Adam-stepped, so pMCPA needs the raw nn.Modules.
_ITER06225_ARCH = {
    "model": dict(skill_dim=64, trunk_hidden=1024, trunk_depth=4,
                  trunk_out_dim=512, head_hidden=256),
    "schupfen_model": {"skill_dim": 64, "hidden": 256},
    "call_model": {"skill_dim": 64, "hidden": 256},
}
_NET_FILES = {"play": "play", "schupfen": "schupfen", "tichu": "tichu", "grand": "grand"}


def sample_worlds(root_view, *, worlds: int, seed: int, belief=None) -> list:
    """Sample `worlds` Determinized Worlds from the actor's mid-Round PrivateState
    `root_view` — each a perfect-information `GameState` keeping the actor's Hand
    and a constraint-respecting assignment of the Unseen Cards to the opponents.
    A thin reproducible wrapper over the search-stack sampler (belief-off when
    `belief is None`, the program default; belief-on is a danger-(b) rescue)."""
    rng = random.Random(seed)
    return [sample_determinized_world(root_view, belief, rng) for _ in range(worlds)]


def record_from(agents, state, actor_seat, *, asked_tichu=frozenset(),
                initial_scores=None):
    """Resume a Round from a mid-Round `GameState` and play it to completion with
    `agents`, recording every Play Decision of `actor_seat` (and only that seat).

    Mirrors `blunder_miner.playout_from`'s loop -- the Tichu-ask fires at a seat's
    first non-Pass Play (grand callers / already-asked seats skipped) -- but
    appends a `RecordedDecision` (the pre-action `GameState`, the chosen action,
    the asked set, the baseline scores) for the actor, so each can later be
    replayed with a forced alternative. Returns `(team_deltas, [decision])`, the
    deltas team-relative to `initial_scores` (pass ROUND-START scores; banked
    trick points accumulate into `public.scores`). Paired advantages difference
    branches from the same point, so any constant `initial_scores` offset cancels.
    """
    asked = set(asked_tichu)
    tichu_callers = set(state.public.tichu_callers)
    grand_callers = state.public.grand_tichu_callers
    initial = state.public.scores if initial_scores is None else initial_scores
    decisions: list[RecordedDecision] = []

    for _ in range(_MAX_STEPS):
        current = state.public.current_player
        private = state.private_view(current)
        action = agents[current].act(private)
        is_play = state.public.pending_decision is None
        if is_play and current == actor_seat:
            decisions.append(
                RecordedDecision(
                    seat=current, turn=len(decisions), state=state, chosen=action,
                    asked_tichu=frozenset(asked), initial_scores=initial,
                )
            )
        if (
            is_play
            and not isinstance(action, Pass)
            and current not in asked
            and current not in grand_callers
        ):
            asked.add(current)
            if _calls(agents[current], private, "tichu"):
                tichu_callers.add(current)
                state = _with_callers(state, tichu=frozenset(tichu_callers))
        state, _, done, _ = step(state, action)
        if done:
            final = state.public.scores
            return (final[0] - initial[0], final[1] - initial[1]), decisions
    raise RuntimeError(
        f"record_from exceeded {_MAX_STEPS} steps without resolving — "
        "likely an infinite loop in agent or engine."
    )


def _play_params(play_net, *, adapt_trunk: bool):
    """The parameters a Round's adaptation is allowed to move. Default: the play
    Head only (robust to the biased K-world sample, leaves the shared Trunk that
    Wish / Dragon ride untouched). `adapt_trunk` adds the Trunk — the capacity
    escalation for a flat result (ADR-0036 Decision D)."""
    head = list(play_net.heads["play"].parameters())
    return list(play_net.trunk.parameters()) + head if adapt_trunk else head


def adapt_play_net(play_a, play_o, rows: list[dict], *, steps: int, lr: float,
                   kl_coef: float, skill_decile: int, clip_eps: float = 0.1,
                   adapt_trunk: bool = False) -> None:
    """Take `steps` paired-advantage policy-gradient steps on `play_a` in place,
    contained by a KL-anchor to the frozen `play_o`. The clipped surrogate +
    KL-anchor are the in-Round guard against a biased-sample gradient cratering
    the Round (no cross-Round anchor — theta_a is discarded). No-op if `rows` is
    empty."""
    if not rows:
        return
    nb = vine_net_batch(rows, skill_decile=skill_decile)
    adv = nb.advantages
    if adv.shape[0] > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    with torch.no_grad():
        anchor_logits = play_o(nb.features, nb.skill)["play"]
    optimizer = torch.optim.Adam(_play_params(play_a, adapt_trunk=adapt_trunk), lr=lr)
    for _ in range(steps):
        logits = play_a(nb.features, nb.skill)["play"]
        new_logp = _masked_logp_at(logits, nb.masks, nb.actions)
        policy = clipped_policy_loss(new_logp, nb.old_logp, adv, clip_eps=clip_eps)
        kl = kl_anchor_loss(logits, anchor_logits, nb.masks)
        loss = policy + kl_coef * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class PMCPAAgent(Agent):
    """An Agent that adapts its play net once per Round, per seat, then resets
    (ADR-0036).

    The frozen offline nets `models` (theta_o) are the rollout field, the
    KL-anchor, and the reset source — never mutated. Adaptation produces a
    throwaway play net theta_a that plays only the Round it was computed on.

    **Per-seat state.** The Tournament places ONE agent instance at both team-0
    seats (`(a, b, a, b)`), so a single shared theta_a would let seat 2's
    adaptation clobber seat 0's. This agent therefore keeps theta_a (and its
    serving MLAgent) keyed by absolute seat — each seat adapts independently on
    its own Hand, exactly the deployed product. Schupfen / calls / wish / dragon
    ride theta_o's (frozen) heads; with the default play-head-only scope the
    shared Trunk is untouched, so wish/dragon are identical to theta_o.
    """

    def __init__(self, models: dict, *, skill_decile: int, worlds: int,
                 decisions_per_world: int, branches: int, steps: int, lr: float,
                 kl_coef: float, clip_eps: float = 0.1, belief=None,
                 emit_branches: bool = False, min_abs_advantage: float = 0.0,
                 adapt_trunk: bool = False, seed: int = 0,
                 partner_trick_guard: bool = True) -> None:
        self._models = models
        self._play_o = models["play"]
        self._skill_decile = int(skill_decile)
        self._partner_trick_guard = bool(partner_trick_guard)
        self._cfg = dict(
            worlds=worlds, decisions_per_world=decisions_per_world, branches=branches,
            steps=steps, lr=lr, kl_coef=kl_coef, clip_eps=clip_eps, belief=belief,
            emit_branches=emit_branches, min_abs_advantage=min_abs_advantage,
            adapt_trunk=adapt_trunk,
        )
        self._rng = random.Random(seed)
        self._seat_play: dict[int, object] = {}   # seat -> theta_a play module
        self._seat_agent: dict[int, object] = {}   # seat -> MLAgent over theta_a

    def _agent_for(self, seat: int):
        """The per-seat serving view, lazily a fresh theta_o clone (so an
        un-adapted seat plays exactly theta_o)."""
        if seat not in self._seat_agent:
            from tichu_inference.ml_agent import MLAgent

            play_a = copy.deepcopy(self._play_o)
            self._seat_play[seat] = play_a
            self._seat_agent[seat] = MLAgent.from_loaded(
                play_a, schupfen=self._models["schupfen"],
                tichu_call=self._models["tichu"], grand_call=self._models["grand"],
                skill_decile=self._skill_decile,
                partner_trick_guard=self._partner_trick_guard,
            )
        return self._seat_agent[seat]

    def restore(self) -> None:
        """Discard every seat's theta_a, restoring theta_o exactly (bit-identical
        state dict) — the per-Round/seat discard discipline."""
        for play_a in self._seat_play.values():
            play_a.load_state_dict(self._play_o.state_dict())

    def _adapt_seat(self, seat: int, root_view, *, worlds_override=None) -> None:
        """Reset `seat` to theta_o, collect paired rows (from sampled worlds, or
        `worlds_override` when given — the oracle path), and adapt to theta_a."""
        self._agent_for(seat)  # ensure the per-seat clone exists
        self._seat_play[seat].load_state_dict(self._play_o.state_dict())
        rows = collect_pmcpa_rows(
            self._models, root_view,
            worlds=self._cfg["worlds"],
            decisions_per_world=self._cfg["decisions_per_world"],
            branches=self._cfg["branches"], skill_decile=self._skill_decile,
            seed=self._rng.randrange(2**31), belief=self._cfg["belief"],
            emit_branches=self._cfg["emit_branches"],
            min_abs_advantage=self._cfg["min_abs_advantage"],
            worlds_override=worlds_override,
        )
        adapt_play_net(
            self._seat_play[seat], self._play_o, rows, steps=self._cfg["steps"],
            lr=self._cfg["lr"], kl_coef=self._cfg["kl_coef"],
            skill_decile=self._skill_decile, clip_eps=self._cfg["clip_eps"],
            adapt_trunk=self._cfg["adapt_trunk"],
        )

    def on_round_start(self, private_view) -> None:
        """Per-Round adaptation for the seat at `private_view`: reset that seat to
        theta_o, then adapt to theta_a on paired rows sampled from the view.
        Fired once per seat at its first Play Decision."""
        self._adapt_seat(private_view.player, private_view)

    def act(self, private_state):
        return self._agent_for(private_state.public.current_player).act(private_state)

    def should_call(self, private_state, kind: str) -> bool:
        seat = private_state.public.current_player
        return self._agent_for(seat).should_call(private_state, kind)

    def rank_actions(self, private_state):
        return self._agent_for(private_state.public.current_player).rank_actions(private_state)


class PMCPAOracleAgent(PMCPAAgent):
    """DIAGNOSTIC ONLY (ADR-0036): adapts on the TRUE world instead of sampled
    Determinized Worlds — the ceiling of any world-quality fix (danger-(b) = 0).

    It cheats: `on_round_start_oracle` receives the full `GameState` (all four
    Hands), so the paired advantages are computed in the *actual* deal rather than
    a biased sample. If even this hurts h2h, the mechanism is broken regardless of
    world quality; if it helps, world-sampling bias is the whole gap. Not a
    servable Agent — the play path (`act`) is unchanged (info-set only); only the
    adaptation peeks at ground truth, via the runner's oracle hook."""

    def on_round_start_oracle(self, game_state) -> None:
        seat = game_state.public.current_player
        self._adapt_seat(seat, game_state.private_view(seat),
                         worlds_override=[game_state])


def collect_pmcpa_rows(models: dict, root_view, *, worlds: int,
                       decisions_per_world: int, branches: int, skill_decile: int,
                       seed: int, asked_tichu=frozenset(), initial_scores=None,
                       belief=None, emit_branches: bool = False,
                       min_abs_advantage: float = 0.0,
                       partner_trick_guard: bool = True,
                       worlds_override=None) -> list[dict]:
    """Sample `worlds` Determinized Worlds from `root_view` (the actor's mid-Round
    PrivateState), play each out with the frozen `models` on all seats, and return
    paired-advantage rows `{features, action, mask, old_logp, advantage}` for the
    actor's Play Decisions only -- the same row schema `vine_net_batch` consumes.

    Each world costs one record playout (its outcome is every recorded decision's
    chosen-branch return, by determinism) plus `branches-1` alternative playouts
    per sampled decision; the chosen branch needs no extra playout.
    """
    from tichu_inference.ml_agent import MLAgent

    agent = MLAgent.from_loaded(
        models["play"], schupfen=models["schupfen"],
        tichu_call=models["tichu"], grand_call=models["grand"],
        skill_decile=skill_decile, partner_trick_guard=partner_trick_guard,
    )
    agents = [agent] * 4
    rng = random.Random(seed)
    actor = root_view.player
    # Oracle path: adapt on the given world(s) (e.g. the TRUE GameState) instead
    # of sampling — danger-(b) = 0, the ceiling diagnostic (PMCPAOracleAgent).
    world_iter = (worlds_override if worlds_override is not None
                  else (sample_determinized_world(root_view, belief, rng)
                        for _ in range(worlds)))
    rows: list[dict] = []
    for world in world_iter:
        total, decisions = record_from(
            agents, world, actor, asked_tichu=asked_tichu,
            initial_scores=initial_scores,
        )
        result = FullRoundResult(total=total, call_bonus=(0, 0))
        eligible = [
            d for d in decisions
            if len(legal_actions_for(d.state.private_view(actor))) > 1
        ]
        rng.shuffle(eligible)
        for d in eligible[:decisions_per_world]:
            rows.extend(_vine_rows(models, agents, agent, d, result,
                                   branches=branches, skill_decile=skill_decile,
                                   emit_branches=emit_branches))
    if min_abs_advantage > 0.0:
        rows = [r for r in rows if abs(r["advantage"]) >= min_abs_advantage]
    return rows


# ---------------------------------------------------------------------------
# Launcher: raw-delta sharded A/B (ADR-0036). Module-level builders so spawn
# workers can unpickle them; the shard runner is resumable (skip-if-exists) and
# persists per-Position deltas so shards pool into one exact bootstrap.
# ---------------------------------------------------------------------------

def load_offline_models(theta_o_dir, *, arch: dict | None = None,
                        tag: str = "iter06225") -> dict:
    """Build raw BCModel / standalone modules at the iter_06225 arch and load the
    recovered `.bin` theta_o state dicts (`{net}_{tag}.bin`) into them — the
    trainable nn.Modules pMCPA adapts (the TorchScript export cannot be stepped)."""
    from tichu_training.bc.training import load_checkpoint
    from tichu_training.cli.train_cotrain import _build_models

    arch = arch or _ITER06225_ARCH
    d = Path(theta_o_dir)
    models = _build_models(arch)
    for net, stem in _NET_FILES.items():
        load_checkpoint(str(d / f"{stem}_{tag}.bin"), models[net])
    return models


def build_pmcpa_agent(*, theta_o_dir, arch: dict | None = None, tag: str = "iter06225",
                      skill_decile: int = 9, worlds: int = 128,
                      decisions_per_world: int = 4, branches: int = 4, steps: int = 3,
                      lr: float = 0.01, kl_coef: float = 1.0, clip_eps: float = 0.1,
                      emit_branches: bool = True, min_abs_advantage: float = 0.0,
                      adapt_trunk: bool = False, seed: int = 0,
                      oracle: bool = False) -> "PMCPAAgent":
    """Module-level builder (a picklable Pool target) for the adapting agent.
    `oracle=True` builds the diagnostic true-world variant (ADR-0036)."""
    models = load_offline_models(theta_o_dir, arch=arch, tag=tag)
    cls = PMCPAOracleAgent if oracle else PMCPAAgent
    return cls(
        models, skill_decile=skill_decile, worlds=worlds,
        decisions_per_world=decisions_per_world, branches=branches, steps=steps,
        lr=lr, kl_coef=kl_coef, clip_eps=clip_eps, emit_branches=emit_branches,
        min_abs_advantage=min_abs_advantage, adapt_trunk=adapt_trunk, seed=seed,
    )


def build_offline_agent(*, theta_o_dir, arch: dict | None = None,
                        tag: str = "iter06225", skill_decile: int = 9):
    """The non-adapted theta_o control: a plain MLAgent over the SAME .bin weights
    pMCPA adapts from (the clean A in adapted-vs-non-adapted)."""
    from tichu_inference.ml_agent import MLAgent

    models = load_offline_models(theta_o_dir, arch=arch, tag=tag)
    return MLAgent.from_loaded(
        models["play"], schupfen=models["schupfen"],
        tichu_call=models["tichu"], grand_call=models["grand"],
        skill_decile=skill_decile,
    )


def build_export_agent(*, export_dir, skill_decile: int = 9):
    """The ship-bar opponent: a plain MLAgent over a TorchScript export dir
    (iter_06225's `policy.pt` + standalone nets)."""
    from tichu_inference.ml_agent import MLAgent

    d = Path(export_dir)
    return MLAgent(
        d / "policy.pt", skill_decile=skill_decile,
        schupfen_path=d / "schupfen.pt", tichu_call_path=d / "tichu_call.pt",
        grand_call_path=d / "grand_tichu_call.pt",
    )


def run_pmcpa_ab_shard(*, theta_o_dir, opponent: str, positions, out_path,
                       workers: int = 1, export_dir=None, skill_decile: int = 9,
                       arch: dict | None = None, tag: str = "iter06225",
                       hp: dict | None = None, progress=None) -> bool:
    """One resumable shard of the adapted-vs-opponent A/B over `positions`.

    `opponent` is "offline" (theta_o control) or "export" (iter_06225 ship bar).
    Writes per-Position seat-swap deltas to `out_path` (.npz: totals + bonuses).
    Returns False without recomputing if `out_path` already exists — the unit of
    resume: a crashed/killed run re-runs only the missing shards."""
    out_path = Path(out_path)
    if out_path.exists():
        return False
    hp = dict(hp or {})
    builder_a = partial(build_pmcpa_agent, theta_o_dir=theta_o_dir, arch=arch,
                        tag=tag, skill_decile=skill_decile, **hp)
    if opponent == "offline":
        builder_b = partial(build_offline_agent, theta_o_dir=theta_o_dir, arch=arch,
                            tag=tag, skill_decile=skill_decile)
    elif opponent == "export":
        if export_dir is None:
            raise ValueError("opponent='export' requires export_dir")
        builder_b = partial(build_export_agent, export_dir=export_dir,
                            skill_decile=skill_decile)
    else:
        raise ValueError(f"unknown opponent {opponent!r} (expected 'offline'|'export')")

    from tichu_eval.tournament import collect_pair_deltas

    # Fine chunks (~6 per worker) so the progress callback fires throughout the
    # shard, not in one burst at the end — the intra-shard liveness/ETA signal.
    max_chunk = max(1, len(positions) // (workers * 6)) if workers > 1 else None
    totals, bonuses = collect_pair_deltas(builder_a, builder_b, positions,
                                          workers=workers, progress=progress,
                                          max_chunk=max_chunk)
    # np.savez appends ".npz" unless the name already ends in it, so the tmp name
    # must end in ".npz" or the rename target won't exist.
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, totals=totals, bonuses=bonuses)
    tmp.replace(out_path)  # atomic — a half-written shard never looks complete
    return True
