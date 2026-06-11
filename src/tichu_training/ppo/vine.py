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
"""

import random

import torch

from tichu_engine.legality import legal_actions_for
from tichu_training.action_space import legal_mask
from tichu_training.featurizer import _combination_to_action_index, featurize
from tichu_training.ppo.cotrain import NetBatch
from tichu_training.search.blunder_miner import (
    candidate_alternatives,
    playout_from,
    record_round,
    team_relative,
)


def collect_vine_rows(models: dict, positions, *, decisions_per_game: int,
                      branches: int, skill_decile: int, seed: int) -> list[dict]:
    """Play `positions` as vine games with the current nets in argmax mode and
    return one row per sampled play decision:
    `{features, action, mask, old_logp, advantage}` (all picklable)."""
    from tichu_inference.ml_agent import MLAgent

    agent = MLAgent.from_loaded(
        models["play"],
        schupfen=models["schupfen"],
        tichu_call=models["tichu"],
        grand_call=models["grand"],
        skill_decile=skill_decile,
        partner_trick_guard=False,  # train the NET; the guard rides at serving only
    )
    agents = [agent] * 4
    rng = random.Random(seed)
    rows: list[dict] = []
    for position in positions:
        result, decisions = record_round(agents, position)
        eligible = [
            d for d in decisions
            if len(legal_actions_for(d.state.private_view(d.seat))) > 1
        ]
        rng.shuffle(eligible)
        for d in eligible[:decisions_per_game]:
            row = _vine_row(models, agents, agent, d, result,
                            branches=branches, skill_decile=skill_decile)
            if row is not None:
                rows.append(row)
    return rows


def _vine_row(models, agents, agent, decision, result, *, branches: int,
              skill_decile: int):
    pv = decision.state.private_view(decision.seat)
    action_idx = _combination_to_action_index(decision.chosen)
    mask = legal_mask("play", decision.state, decision.seat)
    if action_idx is None or not bool(mask[action_idx]):
        return None
    alts = candidate_alternatives(agent, pv, decision.chosen, top_k=branches - 1)
    alts = [a for a in alts if _combination_to_action_index(a) is not None]
    if not alts:
        return None

    # The chosen branch IS the recorded game (deterministic agents — the
    # blunder-miner's parity invariant), so only alternatives need playouts.
    chosen_rel = team_relative(result.total, decision.seat)
    alt_rels = [
        team_relative(
            playout_from(agents, decision.state, forced_action=alt,
                         asked_tichu=decision.asked_tichu,
                         initial_scores=decision.initial_scores),
            decision.seat,
        )
        for alt in alts
    ]
    baseline = (chosen_rel + sum(alt_rels)) / (1 + len(alt_rels))

    features = featurize(pv)
    with torch.no_grad():
        logits = models["play"](
            torch.from_numpy(features).unsqueeze(0),
            torch.tensor([skill_decile], dtype=torch.long),
        )["play"][0]
        masked = logits.masked_fill(~torch.as_tensor(mask, dtype=torch.bool),
                                    float("-inf"))
        old_logp = float(torch.log_softmax(masked, dim=0)[action_idx])
    return {
        "features": features,
        "action": int(action_idx),
        "mask": mask,
        "old_logp": old_logp,
        "advantage": float(chosen_rel - baseline),
    }


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
