"""Value Search — multi-world search with a full-state value leaf (test build).

At each Play Decision the agent takes its policy's choice and the policy's top
alternatives, samples Determinized Worlds from its own observation, plays each
candidate in each world, and scores where it lands with the Full-State Value
net (or the exact outcome when the round ends). Every candidate is scored in
the *same* worlds, so the comparison is paired.

It overrides the policy only when an alternative is ahead by a clear margin
that is also statistically solid across worlds. A plain argmax over noisy
per-candidate means is the optimizer's curse that cost piKL 58 points a round
(ADR-0037); the bar here plays the role of the Belief-Optimal Chooser's
deadband (ADR-0041), with a value net in place of full playouts so that many
worlds are affordable.

`depth` is how many further Play Decisions the field (the policy on every
seat) plays in each world before the leaf is scored. 0 scores the state right
after the candidate; `None` plays the round out, which is the exact-playout
reference and needs no value net.
"""

from __future__ import annotations

import json
import os
import pathlib
import random

import numpy as np

from tichu_engine.legality import legal_actions_for
from tichu_ml.agent import Agent
from tichu_training.search.belief_chooser import asked_tichu_from_public
from tichu_training.search.blunder_miner import candidate_alternatives
from tichu_training.search.determinize import sample_determinized_world
from tichu_training.search.full_state_value import (
    leaf_features,
    resume_round,
    round_totals,
    team_points,
)


def pick_action(values: np.ndarray, *, delta_threshold: float,
                t_threshold: float) -> int:
    """Index of the candidate to play, given `values[c, w]`: candidate `c`'s
    score in world `w`, with candidate 0 the policy's own choice.

    Returns 0 unless some alternative's paired advantage over candidate 0 has a
    mean of at least `delta_threshold` points and a paired t-statistic of at
    least `t_threshold`; among those that clear both, the largest mean wins.
    """
    values = np.asarray(values, dtype=np.float64)
    n_worlds = values.shape[1]
    best, best_mean = 0, None
    for c in range(1, values.shape[0]):
        deltas = values[c] - values[0]
        mean = float(deltas.mean())
        if mean < delta_threshold:
            continue
        if n_worlds > 1:
            se = float(deltas.std(ddof=1)) / np.sqrt(n_worlds)
            t = mean / se if se > 0 else float("inf")
            if t < t_threshold:
                continue
        if best_mean is None or mean > best_mean:
            best, best_mean = c, mean
    return best


def pick_action_mixed(values: np.ndarray, log_probs, *, tau: float) -> int:
    """Gumbel-AlphaZero-style choice without the noise: the candidate with the
    largest `log pi(a) + mean_value(a) / tau`. `tau` is how many points of value
    buy one nat of policy preference, so a move the policy strongly dislikes
    needs a large value edge, and near-ties in the policy go to the value."""
    values = np.asarray(values, dtype=np.float64)
    score = np.asarray(log_probs, dtype=np.float64) + values.mean(axis=1) / tau
    return int(np.argmax(score))


class ValueSearchAgent(Agent):
    """The policy, with its Play Decisions re-checked by Value Search. Schupfen,
    calls, Wish and Dragon give stay the policy's.

    `round_start_scores` is what a terminal outcome is measured against. The
    tournament starts every round at 0-0; a served agent would have to track it.
    """

    def __init__(self, policy, value, *, field=None, worlds: int = 20, top_k: int = 4,
                 depth: int | None = 0, delta_threshold: float = 10.0,
                 t_threshold: float = 2.5, mix_tau: float | None = None, seed: int = 0,
                 round_start_scores=(0, 0), stats_dir=None) -> None:
        if value is None and depth is not None:
            raise ValueError("a value net is required unless depth is None")
        self._policy = policy
        self._value = value
        self._field = field if field is not None else [policy] * 4
        self._worlds = int(worlds)
        self._top_k = int(top_k)
        self._depth = depth
        self._delta = float(delta_threshold)
        self._t = float(t_threshold)
        self._tau = mix_tau
        self._rng = random.Random(seed)
        self._start = tuple(round_start_scores)
        self._stats_path = (
            None if stats_dir is None
            else pathlib.Path(stats_dir) / f"stats_{os.getpid()}.json")
        self.decisions = 0
        self.deviations = 0
        self.gain_sum = 0.0

    def should_call(self, private_state, kind: str) -> bool:
        should = getattr(self._policy, "should_call", None)
        return bool(should(private_state, kind)) if should else False

    def rank_actions(self, private_state):
        return self._policy.rank_actions(private_state)

    def act(self, private_state):
        chosen = self._policy.act(private_state)
        if private_state.public.pending_decision is not None:
            return chosen
        if private_state.public.current_player != private_state.player:
            return chosen  # a Bomb Window offer, not an on-turn Play Decision
        if len(legal_actions_for(private_state)) <= 1:
            return chosen
        alternatives = candidate_alternatives(
            self._policy, private_state, chosen, top_k=self._top_k)
        if not alternatives or self._worlds <= 0:
            return chosen
        candidates = [chosen, *alternatives]
        values = self.candidate_values(private_state, candidates)
        if self._tau is None:
            pick = pick_action(values, delta_threshold=self._delta, t_threshold=self._t)
        else:
            scored = self._policy.play_action_scores(private_state)
            log_probs = [
                np.log(max(next((p for a, p in scored if a == cand), 0.0), 1e-9))
                for cand in candidates]
            pick = pick_action_mixed(values, log_probs, tau=self._tau)
        self.decisions += 1
        if pick != 0:
            self.deviations += 1
            self.gain_sum += float((values[pick] - values[0]).mean())
        if self._stats_path is not None and self.decisions % 100 == 0:
            self._write_stats()
        return candidates[pick]

    def candidate_values(self, private_state, candidates) -> np.ndarray:
        """`values[c, w]` in points for the acting seat's team: each candidate
        played in each of the shared worlds."""
        seat = private_state.player
        asked = asked_tichu_from_public(private_state.public)
        worlds = [sample_determinized_world(private_state, None, self._rng)
                  for _ in range(self._worlds)]
        values = np.zeros((len(candidates), len(worlds)), dtype=np.float64)
        leaves: list[np.ndarray] = []
        where: list[tuple[int, int, int]] = []  # (candidate, world, sign)
        for w, world in enumerate(worlds):
            for c, candidate in enumerate(candidates):
                state, done = resume_round(
                    self._field, world, forced_action=candidate, asked_tichu=asked,
                    max_play_decisions=self._depth)
                if done:
                    totals = round_totals(state.public.scores, self._start)
                    values[c, w] = team_points(totals, seat)
                else:
                    feats, leaf_seat = leaf_features(state)
                    leaves.append(feats)
                    where.append((c, w, 1 if leaf_seat % 2 == seat % 2 else -1))
        if leaves:
            preds = self._value.predict(np.stack(leaves))
            for (c, w, sign), pred in zip(where, preds):
                values[c, w] = sign * float(pred)
        return values

    def _write_stats(self) -> None:
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._stats_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "decisions": self.decisions, "deviations": self.deviations,
            "gain_sum": self.gain_sum}))
        tmp.replace(self._stats_path)


def build_value_search_agent(*, export_dir, value_path=None, skill_decile: int = 9,
                             **hp) -> ValueSearchAgent:
    """Module-level builder (a picklable Pool target): an MLAgent over a
    TorchScript export as the policy and the field, wrapped in Value Search."""
    from tichu_training.ppo.pmcpa import build_export_agent
    from tichu_training.search.full_state_value import load_full_state_value

    policy = build_export_agent(export_dir=export_dir, skill_decile=skill_decile)
    value = None if value_path is None else load_full_state_value(value_path)
    return ValueSearchAgent(policy, value, **hp)
