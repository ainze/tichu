"""Search distillation — teach the policy from played-out move comparisons
(test build).

At a sampled non-forced Play Decision of a greedy self-play round, the chosen
move and the policy's top alternatives are each played to the end in the deal
as it really is. The true deal is one fair draw of what the deciding seat
cannot see, so each outcome is an unbiased (if noisy) sample of that move's
value given the seat's own view. Averaged over many decisions by a loss that is
linear in those outcomes, the policy learns which of its near-top moves does
better — the "policy net on search games" half of expert iteration, with the
search reduced to one exact playout per move.
"""

from __future__ import annotations

import random

import numpy as np

from tichu_engine.legality import legal_actions_for
from tichu_eval.play_full import play_full_round
from tichu_training.search.belief_chooser import asked_tichu_from_public
from tichu_training.search.blunder_miner import candidate_alternatives
from tichu_training.search.full_state_value import (
    _phase,
    resume_round,
    round_totals,
    team_points,
)

MAX_CANDIDATES = 4
_SCALE = 100.0


def emit_decision_targets(agent, position, rng: random.Random, *, decisions: int = 6,
                          top_k: int = 3) -> list[dict]:
    """Move comparisons from one deal, with `agent` (an MLAgent) on every seat.

    Each item: `features` (the policy's own input for the deciding seat), `mask`
    (its legal-action mask over the action space), `cand` (action-space indices,
    candidate 0 the chosen move, -1 padded), `q` (exact round outcome in
    points / 100 for the decider's team after each candidate, NaN padded) and
    `phase`. Candidates that share an action index with an earlier one are
    dropped (the policy cannot tell them apart).
    """
    from tichu_inference.ml_agent import _combination_to_action_index, _trunk_legal_mask

    initial = position.state.public.scores
    base: list[tuple] = []
    result = play_full_round([agent] * 4, position.state, position.grand_prefixes,
                             state_observer=lambda s, g, a: base.append((s, g, a)))
    open_turns = [i for i, (seat, gs, _a) in enumerate(base)
                  if len(legal_actions_for(gs.private_view(seat))) > 1]
    out: list[dict] = []
    for turn in sorted(rng.sample(open_turns, min(decisions, len(open_turns)))):
        seat, gs, chosen = base[turn]
        private = gs.private_view(seat)
        chosen_idx = _combination_to_action_index(chosen)
        if chosen_idx is None:
            continue
        cand = [chosen_idx]
        q = [team_points(result.total, seat) / _SCALE]
        for alt in candidate_alternatives(agent, private, chosen, top_k=top_k):
            idx = _combination_to_action_index(alt)
            if idx is None or idx in cand:
                continue
            end_state, done = resume_round(
                [agent] * 4, gs, forced_action=alt,
                asked_tichu=asked_tichu_from_public(gs.public))
            assert done
            cand.append(idx)
            q.append(team_points(round_totals(end_state.public.scores, initial), seat)
                     / _SCALE)
        if len(cand) < 2:
            continue
        features, _skill = agent._inputs(private)
        mask = _trunk_legal_mask(private, is_play_decision=True)
        pad = MAX_CANDIDATES - len(cand)
        out.append({
            "features": features[0].numpy().astype(np.float16),
            "mask": mask[0].numpy() > 0,
            "cand": np.asarray(cand + [-1] * pad, dtype=np.int16),
            "q": np.asarray(q + [np.nan] * pad, dtype=np.float32),
            "phase": _phase(gs, seat),
        })
    return out


class OnTurnPlayOnly:
    """`tuned` decides on-turn Play Decisions; `base` decides everything else
    (Bomb Window offers, Wish, Dragon give, Schupfen, calls). Isolates what the
    fine-tune was trained on from what it only moved as a side effect."""

    def __init__(self, tuned, base) -> None:
        self._tuned, self._base = tuned, base

    def should_call(self, private_state, kind: str) -> bool:
        return self._base.should_call(private_state, kind)

    def rank_actions(self, private_state):
        return self._pick(private_state).rank_actions(private_state)

    def act(self, private_state):
        return self._pick(private_state).act(private_state)

    def _pick(self, private_state):
        public = private_state.public
        on_turn = (public.pending_decision is None
                   and public.current_player == private_state.player)
        return self._tuned if on_turn else self._base


def build_on_turn_play_only(*, tuned_dir, base_dir, skill_decile: int = 9):
    from tichu_training.ppo.pmcpa import build_export_agent

    return OnTurnPlayOnly(
        build_export_agent(export_dir=tuned_dir, skill_decile=skill_decile),
        build_export_agent(export_dir=base_dir, skill_decile=skill_decile))
