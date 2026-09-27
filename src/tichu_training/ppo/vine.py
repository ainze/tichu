"""Vine / paired-advantage collection for the play head (ADR-0035).

The ADR-0034 closure diagnosed the plateau as advantage noise: `R − V(s)` carries
the round's full outcome variance, drowning per-decision EV differences. The
blunder-miner (2026-06-11) demonstrated the antidote at evaluation time: compare
actions by playing both out IN THE SAME WORLD with deterministic continuation —
the deal-luck term cancels exactly, and with argmax agents each branch return is
a constant, not an estimate. The distill premise test then showed the mined signal
has learnable structure but must not be delivered as hard CE labels (−18 h2h);
this module delivers it as PPO advantages inside the trust region instead.

Per iteration, a few dedicated VINE GAMES are played with the current nets in
argmax mode (the policy as it would be served, guard off — we train the net).
At sampled play decisions, the top-(k−1) alternatives are forced and played out
deterministically (`playout_from`, parity-tested); the chosen branch's return is
the recorded game itself (determinism — no extra playout). The play head's PPO
sub-update then trains ONLY on these steps:

    A(s, chosen) = R_det(chosen) − mean(R_det over all k branches)

exact within the world, luck-free across branches. Schupfen / calls / wish keep
their GAE path on the main stochastic rollout, and the critic keeps pooling all
main-rollout steps (it no longer feeds play advantages, but still feeds GAE for
the other heads).

Off-policy note: branch actions are argmax choices, not samples from the
stochastic policy. `old_logp` is the policy's own masked log-prob of that argmax
action, so the PPO ratio starts at 1 and the clip + KL-to-anchor bound the
mismatch — the same containment that holds the rest of the stack.

Enrichment (the v1 autopsy, 2026-06-12): chosen-only rows delivered the mined
signal ~10x too slowly (2.1% blunder fix-rate after 2,400 iters — flat at full
tournament power because ~+0.2/round is invisible at ±2). Two dials fix the
density without new playouts:
  * `emit_branches` — emit rows for the POSITIVE-advantage alternatives next to
    the chosen row. The returns are already computed; these rows carry the
    corrective direction the chosen-only estimator lacks (at a blunder the
    better branch gets a positive advantage pushing its probability up
    directly, instead of only pushing the chosen action down and letting the
    mass renormalize blindly across all legal actions). Negative-advantage
    alternatives are NOT emitted: they push down actions that are already
    low-probability — an unbounded burn-down with no opposing force that blew
    the logit gaps apart in vine v2's first run (play KL 0.01 -> 0.158 over
    ~170 iterations, then exp(new_logp - old_logp) overflowed and NaN'd the
    weights). The chosen row keeps both signs: its probability is high, and
    pushing a blundered choice down IS the signal.
  * `min_abs_advantage` — drop near-tie rows; within a decision the branch
    advantages sum to zero, so ties contribute only dilution to the per-net
    advantage normalization.
"""

import random

import torch

from tichu_engine.legality import legal_actions_for
from tichu_training.action_space import legal_mask
from tichu_training.featurizer import _combination_to_action_index, featurize
from tichu_training.bc.heads import forward_play_model
from tichu_training.ppo.cotrain import NetBatch
from tichu_training.search.blunder_miner import (
    candidate_alternatives,
    playout_from,
    record_round,
    team_relative,
)


def collect_vine_rows(models: dict, positions, *, decisions_per_game: int,
                      branches: int, skill_decile: int, seed: int,
                      emit_branches: bool = False,
                      min_abs_advantage: float = 0.0,
                      reference_models: dict | None = None,
                      stratify: bool = False) -> list[dict]:
    """Play `positions` as vine games with the current nets in argmax mode and
    return rows `{features, action, mask, old_logp, advantage}` (all picklable):
    one per sampled play decision, plus one per POSITIVE-advantage alternative
    branch when `emit_branches` (same playouts, denser corrective signal). Rows
    with `|advantage| < min_abs_advantage` are dropped.

    `reference_models` (ADR-0040, vine v3) is the **Reference Field**: a frozen
    second net set that plays ALL FOUR seats of every branch continuation —
    including a replay of the chosen action, because the v2 parity trick
    (chosen return := trunk result) is only valid when field == trunk policy.
    The trunk game and the branch RANKING stay the learner's:
    states are where the learner actually goes, `old_logp` is the learner's
    logprob (containment unchanged). `None` = v2 behavior, byte-identical.

    `stratify` (ADR-0040) spends the row budget where the recoverable pool
    lives: decisions with hand <= 10 or a live Tichu/Grand caller are picked
    first (shuffled), the rest fill any remaining budget — v1's uniform
    sampling burned ~3/4 of its rows on near-tie open states."""
    from tichu_inference.ml_agent import MLAgent

    agent = MLAgent.from_loaded(
        models["play"],
        schupfen=models["schupfen"],
        tichu_call=models["tichu"],
        grand_call=models["grand"],
        skill_decile=skill_decile,
    )
    agents = [agent] * 4
    if reference_models is not None:
        ref_agent = MLAgent.from_loaded(
            reference_models["play"],
            schupfen=reference_models["schupfen"],
            tichu_call=reference_models["tichu"],
            grand_call=reference_models["grand"],
            skill_decile=skill_decile,
        )
        field_agents = [ref_agent] * 4
    else:
        field_agents = None  # v2: branches continue with the learner itself
    rng = random.Random(seed)
    rows: list[dict] = []
    for position in positions:
        result, decisions = record_round(agents, position)
        eligible = [
            d for d in decisions
            if len(legal_actions_for(d.state.private_view(d.seat))) > 1
        ]
        if stratify:
            preferred = [d for d in eligible if _is_pool_state(d)]
            rest = [d for d in eligible if not _is_pool_state(d)]
            rng.shuffle(preferred)
            rng.shuffle(rest)
            eligible = preferred + rest
        else:
            rng.shuffle(eligible)
        for d in eligible[:decisions_per_game]:
            rows.extend(_vine_rows(models, agents, agent, d, result,
                                   branches=branches, skill_decile=skill_decile,
                                   emit_branches=emit_branches,
                                   field_agents=field_agents))
    if min_abs_advantage > 0.0:
        rows = [r for r in rows if abs(r["advantage"]) >= min_abs_advantage]
    return rows


def _is_pool_state(decision) -> bool:
    """True iff this decision sits where the recoverable pool lives (ADR-0040):
    the actor holds <= 10 cards, or any Tichu/Grand call is live."""
    pub = decision.state.public
    return (
        len(decision.state.hands[decision.seat]) <= 10
        or bool(pub.tichu_callers)
        or bool(pub.grand_tichu_callers)
    )


def _vine_rows(models, agents, agent, decision, result, *, branches: int,
               skill_decile: int, emit_branches: bool,
               field_agents=None) -> list[dict]:
    pv = decision.state.private_view(decision.seat)
    action_idx = _combination_to_action_index(decision.chosen)
    mask = legal_mask("play", decision.state, decision.seat)
    if action_idx is None or not bool(mask[action_idx]):
        return []
    alts = candidate_alternatives(agent, pv, decision.chosen, top_k=branches - 1)
    alts = [a for a in alts if _combination_to_action_index(a) is not None]
    if not alts:
        return []

    def _branch_rel(forced) -> float:
        return team_relative(
            playout_from(field_agents or agents, decision.state,
                         forced_action=forced,
                         asked_tichu=decision.asked_tichu,
                         initial_scores=decision.initial_scores),
            decision.seat,
        )

    # v2 (no Reference Field): the chosen branch IS the recorded game
    # (deterministic agents — the blunder-miner's parity invariant), so only
    # alternatives need playouts. v3: field != trunk policy, so the chosen
    # action is REPLAYED under the field like every other branch.
    if field_agents is None:
        chosen_rel = team_relative(result.total, decision.seat)
    else:
        chosen_rel = _branch_rel(decision.chosen)
    alt_rels = [_branch_rel(alt) for alt in alts]
    baseline = (chosen_rel + sum(alt_rels)) / (1 + len(alt_rels))

    features = featurize(pv)
    with torch.no_grad():
        logits = forward_play_model(
            models["play"],
            torch.from_numpy(features).unsqueeze(0),
            torch.tensor([skill_decile], dtype=torch.long),
            torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0),
        )["play"][0]
        masked = logits.masked_fill(~torch.as_tensor(mask, dtype=torch.bool),
                                    float("-inf"))
        logp = torch.log_softmax(masked, dim=0)

    def _row(idx: int, rel: float) -> dict:
        return {
            "features": features,
            "action": int(idx),
            "mask": mask,
            "old_logp": float(logp[idx]),
            "advantage": float(rel - baseline),
        }

    rows = [_row(action_idx, chosen_rel)]
    if emit_branches:
        # Positive-advantage alternatives only — see the module docstring.
        rows.extend(_row(_combination_to_action_index(alt), rel)
                    for alt, rel in zip(alts, alt_rels) if rel > baseline)
    return rows


def vine_net_batch(rows: list[dict], *, skill_decile: int) -> NetBatch:
    """Assemble vine rows into the play head's `NetBatch` (drop-in replacement for
    the GAE play group in a `CoTrainBatch`)."""
    features = torch.stack([torch.as_tensor(r["features"], dtype=torch.float32)
                            for r in rows])
    return NetBatch(
        features=features,
        skill=torch.full((features.shape[0],), int(skill_decile), dtype=torch.long),
        actions=torch.tensor([r["action"] for r in rows], dtype=torch.long),
        masks=torch.stack([torch.as_tensor(r["mask"], dtype=torch.bool)
                           for r in rows]),
        old_logp=torch.tensor([r["old_logp"] for r in rows], dtype=torch.float32),
        advantages=torch.tensor([r["advantage"] for r in rows], dtype=torch.float32),
    )
