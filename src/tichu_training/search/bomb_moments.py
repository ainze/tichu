"""Probe for issue #110: what are the bomb moments the Bomb Window contract
leaves out worth? (test build, nothing here changes the engine)

The round driver asks `bomb_window_queue` who gets a Bomb Window. This module
swaps in a wider queue *inside the probe's own processes only*:

- `after_pass`: after a Pass, with a top on the table, every seat that holds a
  beating bomb and is neither the passer nor the seat now to act is asked;
- `empty`: when the Trick is empty (after the Dog, a resolved Trick, a Dragon
  give), every seat that holds a bomb and is not the seat about to lead is asked.

Both teams play that wider game. One team answers the extra windows from its
play head; the other is wrapped in `ExtraWindows(decline=True)` and always
passes in them. The seat-swap margin is what using the extra moments is worth
to the current net, with no retraining — a lower bound on their value.
"""

from __future__ import annotations

from collections import Counter

from tichu_engine.legality import (
    PASS,
    Pass,
    bomb_window_queue,
    legal_bomb_interrupts,
)

NUM_PLAYERS = 4
MODE = {"after_pass": False, "empty": False}
COUNTS: Counter = Counter()
_extra: str | None = None  # kind of the window queue now being asked, if extra


def extended_queue(state, action, actor) -> tuple[int, ...]:
    global _extra
    _extra = None
    base = bomb_window_queue(state, action, actor)
    public = state.public
    if base or public.pending_decision is not None:
        return base
    current = public.current_player
    top = public.trick.top_combination
    if top is not None and isinstance(action, Pass) and MODE["after_pass"]:
        kind, skip = "after_pass", {actor, current}
    elif top is None and MODE["empty"]:
        kind, skip = "empty", {current}
    else:
        return ()
    seats = tuple(
        seat for seat in ((current + off) % NUM_PLAYERS for off in range(1, NUM_PLAYERS))
        if seat not in skip and legal_bomb_interrupts(state, seat))
    if seats:
        _extra = kind
        COUNTS[f"{kind}_queues"] += 1
    return seats


def install(*, after_pass: bool, empty: bool) -> None:
    """Point the full-round driver at the wider queue (this process only)."""
    import tichu_eval.play_full as play_full

    MODE["after_pass"], MODE["empty"] = after_pass, empty
    play_full.bomb_window_queue = extended_queue


class ExtraWindows:
    """Counts extra-window answers; with `decline`, always passes in them."""

    def __init__(self, base, *, decline: bool) -> None:
        self._base, self._decline = base, decline

    def __getattr__(self, name):
        return getattr(self._base, name)

    def should_call(self, private_state, kind: str) -> bool:
        return self._base.should_call(private_state, kind)

    def rank_actions(self, private_state):
        return self._base.rank_actions(private_state)

    def act(self, private_state):
        public = private_state.public
        off_turn = (public.pending_decision is None
                    and public.current_player != private_state.player)
        if not (off_turn and _extra):
            return self._base.act(private_state)
        COUNTS[f"{_extra}_asked"] += 1
        if self._decline:
            return PASS
        action = self._base.act(private_state)
        if not isinstance(action, Pass):
            COUNTS[f"{_extra}_bombed"] += 1
            holder = public.trick.leader
            if holder is not None:
                same = holder % 2 == private_state.player % 2
                COUNTS[f"{_extra}_bombed_{'partner' if same else 'opponent'}_top"] += 1
        return action


def build_user(*, export_dir, after_pass: bool, empty: bool, skill_decile: int = 9):
    from tichu_training.ppo.pmcpa import build_export_agent

    install(after_pass=after_pass, empty=empty)
    return ExtraWindows(build_export_agent(export_dir=export_dir, skill_decile=skill_decile),
                        decline=False)


def build_decliner(*, export_dir, after_pass: bool, empty: bool, skill_decile: int = 9):
    from tichu_training.ppo.pmcpa import build_export_agent

    install(after_pass=after_pass, empty=empty)
    return ExtraWindows(build_export_agent(export_dir=export_dir, skill_decile=skill_decile),
                        decline=True)
