"""Behavioral Drift Benchmark — the human reference, from replayed BSW games.

Each human Round is replayed through the engine (`replay_round`) and every
Decision is logged by the SAME `decision_row` the agents use, so a human rate has
exactly the agents' definition. The result is a `DriftLog` with its own `arm`
name, shaped like the BC arm: every measured team is one `half` (half = team),
so every metric runs on it unchanged.

It is a **reference, not an arm**: humans play different deals against human
opponents, so there is no pairing, no Δ and no BH — `drift_stats.summarise_reference`
gives each rate a CI bootstrapped by Game (the same players recur within one).
Known gaps versus the agent arms, stated in the report:

* out-of-turn Bomb interrupts are not logged (agents only act on their turn);
* Trick attribution is unavailable from the replay (`tricks_won` is None);
* humans see the game score, so they are not score-blind.

The replay pre-populates every Tichu caller from the start of the Round. A
recorder reading that directly would show a partner as "called" before they
called, so callers are re-added in the order their call lines appear.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import pandas as pd

from tichu_engine.legality import BombInterrupt, Pass
from tichu_engine.state import GameState, PublicState, Trick
from tichu_eval.drift_arms import DriftLog
from tichu_eval.drift_recorder import RoundLog, RoundTracker, decision_row, round_row
from tichu_eval.play_full import _call_bonus, _decision_kind
from tichu_training.bsw.replay import replay_round

_TARGET = 1000


def record_human_round(parsed, *, deal: int, game, subject_teams) -> list[RoundLog] | None:
    """Replay one parsed human Round and log it once per team in `subject_teams`
    (half = team). None when the Round does not replay cleanly."""
    result = replay_round(parsed)
    if result.illegal_action is not None:
        return None
    grand = frozenset(parsed.grand_tichu_callers)
    tracker = RoundTracker()
    rows: list[tuple] = []   # (seat, kind, state, action)

    grand_state = _grand_state(parsed.pre_deal_hands)
    for seat in range(4):
        rows.append((seat, "grand", grand_state, seat in grand))

    called: set[int] = set()
    asked_tichu: set[int] = set()
    for (parsed_action, action), state in zip(result.decisions, result.pre_decision_states):
        if parsed_action.kind in ("tichu", "grand_tichu"):
            if parsed_action.kind == "tichu":
                called.add(parsed_action.player)
            continue
        if state is None:                          # a phantom BSW pass line
            continue
        if isinstance(action, BombInterrupt):      # out of turn: agents never do this
            tracker.observe("play", action.bomb)
            continue
        view = _with_callers(state, grand=grand, tichu=called)
        kind = _decision_kind(state.public.pending_decision)
        seat = parsed_action.player
        if (kind == "play" and not isinstance(action, Pass)
                and seat not in asked_tichu and seat not in grand):
            # The agents are asked for Tichu once, at their first non-Pass Play.
            asked_tichu.add(seat)
            before_own = _with_callers(state, grand=grand, tichu=called - {seat})
            rows.append((seat, "tichu", before_own, seat in parsed.tichu_callers))
        tracker.observe(kind, action)
        rows.append((seat, kind, view, action))

    out_order = _out_order(parsed.plays)
    first_out = out_order[0] if out_order else None
    bonus = _call_bonus(grand, frozenset(parsed.tichu_callers), first_out)
    logs = []
    for team in subject_teams:
        subject = frozenset((team, team + 2))
        log = RoundLog()
        log.decisions = [decision_row(seat, kind, st, a, deal=deal, half=team, subject=subject)
                         for seat, kind, st, a in rows]
        log.round = round_row(
            deal=deal, half=team, subject_team=team, totals=parsed.ergebnis, call_bonus=bonus,
            grand_callers=sorted(grand), tichu_callers=sorted(parsed.tichu_callers),
            out_order=out_order, tricks_won=None, tricks_total=None, tracker=tracker,
        )
        logs.append(log)
    return logs


def human_log(games, *, name: str, n_rounds: int, decile_of: dict | None = None,
              min_decile: int | None = None) -> DriftLog:
    """Log up to `n_rounds` human Rounds from an iterable of `ParsedGame`s.

    With `min_decile`, a team is measured only when BOTH partners' Skill Decile
    (`decile_of[handle]`; anonymous seats never qualify) is at least that — the
    filter runs on handles before any replay, so a rare stratum stays cheap.
    Every row carries the Game it came from (`game`, `round_index`) and each
    Round the Game's real length in Rounds (`game_length`, NaN if the Game never
    reached 1000) for the Rounds-per-Game reference."""
    decisions: list[dict] = []
    rounds: list[dict] = []
    deal = 0
    for game in games:
        length = _game_length(game)
        for parsed in game.rounds:
            teams = _teams(parsed, decile_of, min_decile)
            if not teams:
                continue
            logs = record_human_round(parsed, deal=deal, game=game.game_id, subject_teams=teams)
            if logs is None:
                continue
            extra = {"arm": name, "game": game.game_id, "round_index": parsed.round_index}
            for log in logs:
                decisions.extend({**d, **extra} for d in log.decisions)
                rounds.append({**log.round, **extra, "game_length": length})
            deal += 1
            if deal >= n_rounds:
                return _to_log(decisions, rounds)
    return _to_log(decisions, rounds)


def _to_log(decisions, rounds) -> DriftLog:
    return DriftLog(decisions=pd.DataFrame(decisions), rounds=pd.DataFrame(rounds))


def _teams(parsed, decile_of, min_decile) -> tuple[int, ...]:
    if min_decile is None:
        return (0, 1)
    def strong(seat):
        handle = parsed.handles[seat]
        return bool(handle) and (decile_of or {}).get(handle, -1) >= min_decile
    return tuple(t for t in (0, 1) if strong(t) and strong(t + 2))


def _grand_state(pre_deal_hands) -> GameState:
    """The `(8,8,8,8)` state the Grand-Tichu Call is decided on — the same shape
    `play_full_round` asks the agents on."""
    hands = tuple(frozenset(h) for h in pre_deal_hands)
    public = PublicState(current_player=0, hand_sizes=tuple(len(h) for h in hands),
                         scores=(0, 0), trick=Trick.empty())
    return GameState(hands=hands, public=public)


def _with_callers(state: GameState, *, grand, tichu) -> GameState:
    public = dataclasses.replace(state.public, grand_tichu_callers=frozenset(grand),
                                 tichu_callers=frozenset(tichu))
    return dataclasses.replace(state, public=public)


def _out_order(plays) -> tuple[int, ...]:
    """Seats in the order they emptied their 14-card Hand (Bomb interrupts count)."""
    played: Counter = Counter()
    out: list[int] = []
    for a in plays:
        if a.kind == "play":
            played[a.player] += len(a.cards)
            if played[a.player] >= 14 and a.player not in out:
                out.append(a.player)
    return tuple(out)


def _game_length(game) -> float:
    t0 = t1 = 0
    for i, r in enumerate(game.rounds, start=1):
        t0 += r.ergebnis[0]
        t1 += r.ergebnis[1]
        if max(t0, t1) >= _TARGET and t0 != t1:
            return float(i)
    return float("nan")
