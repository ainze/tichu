"""Belief-Optimal Chooser (ADR-0041) — the information-ceiling instrument.

One seat replaces its Policy Network's argmax with the action that wins across
**Determinized Worlds** drawn from its **Observation**. Not **PIMC**: there is no
tree, no PUCT, no per-world argmax — a single action is committed across all K
worlds, so strategy fusion is absent by construction.

The deviation rule is the blunder-miner's validated tier-2 criterion rather than
a raw argmax over the K-world mean. A K-world mean is a noisy `Q`, and argmax
over a noisy `Q` is the optimizer's curse — the measured mechanism behind piKL's
-58.66/round ([ADR-0037](../../../docs/adr/0037-pikl-inference-time-anchored-search.md)).
"""

from __future__ import annotations

import random

from tichu_engine.legality import Pass, legal_actions_for
from tichu_training.search.blunder_miner import (
    candidate_alternatives,
    playout_from,
    team_relative,
)
from tichu_training.search.determinize import sample_determinized_world


def asked_tichu_from_public(public) -> frozenset[int]:
    """The seats the runner has already offered a Tichu call to.

    ADR-0018's rule is "first non-Pass play, grand callers skipped", which the
    blunder-miner reconstructs from a recorded prefix. At inference there is no
    prefix — but Featurizer v6 added `played_cards_by_player` to **PublicState**
    (ADR-0038), so "has made a non-Pass play" is now directly readable and the
    reconstruction is exact rather than approximate.
    """
    return frozenset(
        seat for seat in range(4)
        if public.played_cards_by_player[seat]
        and seat not in public.grand_tichu_callers
    )


class BeliefOptimalChooser:
    """The **Belief-Optimal Chooser** as an Agent, for exactly ONE seat.

    Every other seat — including this seat's own later turns inside a playout —
    is the frozen champion, so the improved seat faces a fixed MDP and this is one
    exact step of policy iteration. Upgrading a second seat would make it a joint
    policy change with no such guarantee (ADR-0041).

    `marginals_fn(private_state) -> (3, 56) | None` selects the arm: `None`
    (the default) is **belief-off** — `sample_determinized_world`'s uniform
    card-counting sampler — and a **Belief Model** forward is **belief-on**.
    """

    def __init__(
        self,
        policy,
        playout_agents,
        *,
        marginals_fn=None,
        worlds: int = 24,
        top_k: int = 4,
        win_threshold: float = 0.70,
        delta_threshold: float = 15.0,
        seed: int = 0,
    ) -> None:
        self._policy = policy
        self._playout_agents = playout_agents
        self._marginals_fn = marginals_fn
        self._worlds = int(worlds)
        self._top_k = int(top_k)
        self._win = float(win_threshold)
        self._delta = float(delta_threshold)
        self._rng = random.Random(seed)
        self.deviations = 0
        self.decisions = 0

    # The Chooser only re-decides Play; Schupfen / calls / wish stay frozen.
    def should_call(self, private_state, kind: str) -> bool:
        # `should_call` is optional on an Agent — `play_full._calls` treats a
        # missing method as "declines", so mirror that instead of assuming it.
        should = getattr(self._policy, "should_call", None)
        return bool(should(private_state, kind)) if should else False

    def rank_actions(self, private_state):
        return self._policy.rank_actions(private_state)

    def act(self, private_state):
        chosen = self._policy.act(private_state)
        if private_state.public.pending_decision is not None:
            return chosen
        if len(list(legal_actions_for(private_state))) <= 1:
            return chosen

        candidates = candidate_alternatives(
            self._policy, private_state, chosen, top_k=self._top_k,
        )
        if not candidates or self._worlds <= 0:
            return chosen

        self.decisions += 1
        deltas = self._paired_deltas(private_state, chosen, candidates)
        action = choose_with_deadband(
            chosen, deltas,
            win_threshold=self._win, delta_threshold=self._delta,
        )
        if action is not chosen:
            self.deviations += 1
        return action

    def _paired_deltas(self, private_state, chosen, candidates) -> dict:
        """`{candidate_index: [delta per world]}` with **common random numbers**:
        every candidate is replayed in the *same* K worlds as `chosen`, so
        world-level variance cancels in the difference. piKL's estimator lacked
        this and its argmax was 32% unstable at N=10."""
        seat = private_state.player
        asked = asked_tichu_from_public(private_state.public)
        out: dict = {i: [] for i in range(len(candidates))}
        for _ in range(self._worlds):
            belief = (
                None if self._marginals_fn is None
                else self._marginals_fn(private_state)
            )
            world = sample_determinized_world(private_state, belief, self._rng)
            base = team_relative(
                playout_from(
                    self._playout_agents, world, forced_action=chosen,
                    asked_tichu=asked,
                ),
                seat,
            )
            for i, candidate in enumerate(candidates):
                alt = team_relative(
                    playout_from(
                        self._playout_agents, world, forced_action=candidate,
                        asked_tichu=asked,
                    ),
                    seat,
                )
                out[i].append(alt - base)
        return {candidates[i]: d for i, d in out.items()}


def choose_with_deadband(
    chosen,
    deltas_by_candidate: dict,
    *,
    win_threshold: float,
    delta_threshold: float,
):
    """The champion's `chosen` action unless some candidate clears **both** the
    world-win-rate and the mean paired-delta bar; the best clearing candidate by
    mean delta otherwise.

    Both bars are load-bearing. Win-rate alone admits a candidate that wins
    everywhere by a rounding error; mean alone admits a hindsight lottery that
    loses in most worlds and wins enormously in one — the failure mode that made
    96-99% of the miner's true-world gains evaporate under resampling.
    """
    best, best_mean = chosen, None
    for candidate, deltas in deltas_by_candidate.items():
        if not deltas:
            continue
        mean = sum(deltas) / len(deltas)
        win_rate = sum(1 for d in deltas if d > 0) / len(deltas)
        if win_rate < win_threshold or mean < delta_threshold:
            continue
        if best_mean is None or mean > best_mean:
            best, best_mean = candidate, mean
    return best
