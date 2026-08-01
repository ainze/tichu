"""Blunder-miner: counterfactual replay with a hindsight -> robustness funnel.

Self-play gives the true deal for free, and the served agents are deterministic
(argmax), so "what if it had played X here instead?" is a single exact playout, not
a Monte-Carlo estimate. The funnel:

  Tier 1 (hindsight, broad):   at every recorded Play Decision, force each of the
      top-k alternatives in the TRUE world and play the round out; record
      `delta = alt_outcome - chosen_outcome` (team-relative to the actor). The
      chosen outcome is the original round's outcome (determinism), so tier 1
      costs k playouts per decision.
  Tier 2 (robustness, narrow): hindsight winners may just be luck. For each
      candidate, resample K Determinized Worlds consistent with the actor's
      OBSERVATION (`sample_determinized_world`, belief-off) and replay chosen vs
      alternative in each. Only an alternative that wins across most consistent
      worlds is a TRUE observable mistake — the agent should have known better.
      This is what kills hindsight bias / the strategy-fusion trap.

Surviving blunders are clustered into hypotheses; a fat cluster is adjudicated by
a forced-action probe at tournament scale (the ADR-0031 harness), NOT by this
miner — playout deltas here are policy-relative evidence, not a ship verdict.

The mid-round resume (`playout_from`) mirrors `play_full_round` exactly — the
tichu-ask rule (first non-Pass per seat, grand callers skipped) is reconstructed
from the recorded prefix — and is parity-tested from every recorded decision.
"""

import hashlib
from dataclasses import dataclass

from tichu_engine.combinations import FourOfAKindBomb, StraightFlushBomb
from tichu_engine.engine import step
from tichu_engine.legality import Pass, legal_actions_for
from tichu_eval.play_full import FullRoundResult, _calls, _with_callers, play_full_round
from tichu_training.search.determinize import sample_determinized_world

_MAX_STEPS = 10_000
_BOMB_TYPES = (FourOfAKindBomb, StraightFlushBomb)

# Premium specials for context classification (rank-free identity check by name —
# avoids importing card singletons here).
_PREMIUM_NAMES = ("DRAGON", "PHOENIX")


@dataclass(frozen=True)
class RecordedDecision:
    """One Play Decision captured during a recorded self-play round: the full
    perfect-info `GameState` (immutable, held by reference), the action the agent
    chose, and the runner-side bookkeeping the `PublicState` does not carry —
    the tichu-ask set and the ROUND-START scores (`public.scores` accumulates
    banked trick points mid-round, so a resume must difference against the round
    start, not the resume point)."""

    seat: int
    turn: int
    state: object  # GameState
    chosen: object  # ConcreteAction
    asked_tichu: frozenset
    initial_scores: tuple = (0, 0)


def record_round(agents, position):
    """Play one full round with the trusted runner, capturing every Play Decision
    via the `state_observer` hook. Returns `(FullRoundResult, [RecordedDecision])`.

    `asked_tichu` reconstruction: a seat has been asked exactly when it has made a
    non-Pass play earlier in the round and is not a grand caller (ADR-0018's
    first-non-Pass rule, mirroring `play_full_round`'s own bookkeeping)."""
    decisions: list[RecordedDecision] = []
    asked: set[int] = set()
    initial_scores = position.state.public.scores

    def observer(seat, game_state, action):
        decisions.append(
            RecordedDecision(
                seat=seat,
                turn=len(decisions),
                state=game_state,
                chosen=action,
                asked_tichu=frozenset(asked),
                initial_scores=initial_scores,
            )
        )
        if not isinstance(action, Pass) and seat not in game_state.public.grand_tichu_callers:
            asked.add(seat)

    result = play_full_round(
        agents, position.state, position.grand_prefixes, state_observer=observer
    )
    return result, decisions


def playout_from(agents, state, *, forced_action=None, asked_tichu=frozenset(),
                 initial_scores=None):
    """Resume a round from a mid-round `GameState` and play it to completion,
    forcing `forced_action` as the current seat's action. Returns the team score
    deltas `(team0, team1)` relative to `initial_scores` — pass the ROUND-START
    scores (banked trick points accumulate into `public.scores` mid-round, so
    differencing against the resume point would drop them). Defaults to the
    resume state's scores, which is only correct at the round start.

    Mirrors `play_full_round`'s loop: pending decisions route through the agents,
    and the tichu-ask fires at a seat's first non-Pass play (grand callers and
    already-asked seats — `asked_tichu` — are skipped)."""
    asked = set(asked_tichu)
    tichu_callers = set(state.public.tichu_callers)
    grand_callers = state.public.grand_tichu_callers
    initial = state.public.scores if initial_scores is None else initial_scores
    forced = forced_action

    for _ in range(_MAX_STEPS):
        current = state.public.current_player
        private = state.private_view(current)
        if forced is not None:
            action = forced
            forced = None
        else:
            action = agents[current].act(private)
        if (
            state.public.pending_decision is None
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
            return (final[0] - initial[0], final[1] - initial[1])
    raise RuntimeError(
        f"playout_from exceeded {_MAX_STEPS} steps without resolving — "
        "likely an infinite loop in agent or engine."
    )


def team_relative(totals, seat: int) -> float:
    """Team-relative outcome for the actor: own team total minus opponents'."""
    own = totals[seat % 2]
    other = totals[1 - seat % 2]
    return float(own - other)


def candidate_alternatives(agent, private_state, chosen, *, top_k: int):
    """The actor's top-k legal alternatives to `chosen`, in the agent's own
    preference order when it exposes `rank_actions` (MLAgent does), else legal
    order. Decisions with one legal action yield no candidates."""
    legal = list(legal_actions_for(private_state))
    rank = getattr(agent, "rank_actions", None)
    ranked = (rank(private_state) if rank is not None else None) or legal
    return [a for a in ranked if a != chosen][:top_k]


def _combo_kind(action) -> str:
    if isinstance(action, Pass):
        return "Pass"
    name = type(action).__name__
    if isinstance(action, _BOMB_TYPES):
        return f"Bomb:{name}"
    cards = getattr(action, "cards", None)
    has_premium = False
    try:
        from tichu_engine.legality import _cards_in

        has_premium = any(
            getattr(c, "name", "") in _PREMIUM_NAMES for c in _cards_in(action)
        )
    except Exception:
        pass
    return f"{name}{'+premium' if has_premium else ''}"


def decision_context(state, seat: int, chosen, alt) -> dict:
    """Clusterable context for one decision (the `mine_divergences` altitude:
    role, phase, caller context, action kinds — hypothesis keys, not verdicts)."""
    pub = state.public
    leader = pub.trick.leader
    partner = (seat + 2) % 4
    callers = set(pub.tichu_callers) | set(pub.grand_tichu_callers)
    if seat in callers:
        caller_ctx = "self"
    elif partner in callers:
        caller_ctx = "partner"
    elif callers:
        caller_ctx = "opp"
    else:
        caller_ctx = "none"
    hand = len(state.hands[seat])
    phase = "open" if hand >= 11 else ("mid" if hand >= 6 else "late")
    return {
        "role": "lead" if pub.trick.top_combination is None else "follow",
        "phase": phase,
        "caller_ctx": caller_ctx,
        "partner_leads": leader == partner,
        "chosen_kind": _combo_kind(chosen),
        "alt_kind": _combo_kind(alt),
    }


def mine_round(agents, position, *, round_idx: int = 0, top_k: int = 4):
    """Tier 1 for one round: hindsight deltas for every alternative at every Play
    Decision (in the TRUE world). One row per (decision, alternative)."""
    result, decisions = record_round(agents, position)
    rows: list[dict] = []
    for d in decisions:
        private = d.state.private_view(d.seat)
        legal = legal_actions_for(private)
        if len(legal) <= 1:
            continue
        chosen_rel = team_relative(result.total, d.seat)
        for alt in candidate_alternatives(agents[d.seat], private, d.chosen, top_k=top_k):
            totals = playout_from(
                agents, d.state, forced_action=alt, asked_tichu=d.asked_tichu,
                initial_scores=d.initial_scores,
            )
            ctx = decision_context(d.state, d.seat, d.chosen, alt)
            rows.append(
                {
                    "round_idx": round_idx,
                    "turn": d.turn,
                    "seat": d.seat,
                    "n_legal": len(legal),
                    "chosen": repr(d.chosen),
                    "alt": repr(alt),
                    "delta": team_relative(totals, d.seat) - chosen_rel,
                    "chosen_rel": chosen_rel,
                    **ctx,
                }
            )
    return rows


def candidate_seed(round_idx: int, turn: int, alt_repr: str) -> int:
    """Process-stable rng seed for one tier-2 candidate.

    Deliberately NOT builtin `hash()`: hashing a `str` is salted by PYTHONHASHSEED,
    which is unset in this repo and re-drawn in every `spawn` worker — so the same
    candidate drew different Determinized Worlds on every run, and tier-2 verdicts
    were not reproducible even within one run.
    """
    key = f"{round_idx}|{turn}|{alt_repr}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def verify_candidate(agents, decision: RecordedDecision, alternative, *, worlds: int, rng):
    """Tier 2: replay chosen vs `alternative` across `worlds` Determinized Worlds
    consistent with the actor's observation (belief-off). The actor's hand and the
    public state are identical in every world, so both actions stay legal; only the
    hidden hands move. Returns the robust read for this candidate."""
    own_view = decision.state.private_view(decision.seat)
    deltas: list[float] = []
    wins = 0
    for _ in range(worlds):
        world = sample_determinized_world(own_view, None, rng)
        chosen_totals = playout_from(
            agents, world, forced_action=decision.chosen,
            asked_tichu=decision.asked_tichu, initial_scores=decision.initial_scores,
        )
        alt_totals = playout_from(
            agents, world, forced_action=alternative,
            asked_tichu=decision.asked_tichu, initial_scores=decision.initial_scores,
        )
        delta = team_relative(alt_totals, decision.seat) - team_relative(
            chosen_totals, decision.seat
        )
        deltas.append(delta)
        if delta > 0:
            wins += 1
    return {
        "worlds": worlds,
        "alt_win_rate": wins / worlds,
        "mean_delta": sum(deltas) / len(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
    }
