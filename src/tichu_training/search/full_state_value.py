"""Full-State Value — a value net that sees all four hands (test build).

The observable critic misjudges late-round positions because the outcome there
is decided by exactly which cards each seat holds. Inside a Determinized World
every hand is known, so a search leaf can be scored by a net that takes the
whole `GameState`. This module holds the three pieces that net needs:

- `resume_round`: continue a round from a mid-round `GameState`, mirroring
  `play_full_round` (Tichu ask at the first non-Pass play, Bomb Windows), with an
  optional cap on how many further on-turn Play Decisions are taken.
- `emit_deal`: one deal of greedy self-play plus one counterfactual branch, as
  `(features, z)` rows. The served agents are deterministic, so the label of a
  full state is an exact number, not a noisy sample.
- `FullStateValue`: the trained net behind a batched `predict`.

Every labelled state is an on-turn Play Decision, featurized from the seat to
act. A leaf must be evaluated at the same kind of state (`leaf_features`).
"""

from __future__ import annotations

import pathlib
import random

import numpy as np

from tichu_engine.engine import step
from tichu_engine.legality import Pass, bomb_window_queue, legal_actions_for
from tichu_eval.play_full import _calls, _with_callers, play_full_round
from tichu_training.featurizer import FEATURIZER_VERSION
from tichu_training.perfect_info import PERFECT_INFO_DIM, featurize_perfect_info
from tichu_training.search.belief_chooser import asked_tichu_from_public
from tichu_training.search.blunder_miner import candidate_alternatives

_MAX_STEPS = 10_000
_SCALE = 100.0  # value targets are round-outcome points / 100


def resume_round(agents, state, *, forced_action=None, asked_tichu=frozenset(),
                 max_play_decisions=None, state_observer=None):
    """Continue a round from `state` (an on-turn Play Decision). Returns
    `(state, done)`.

    `forced_action`, if given, is played by the seat to act instead of asking its
    agent. `max_play_decisions` caps how many further on-turn Play Decisions the
    agents take: the round stops at the next one and returns `done=False`. With 0
    the forced action is applied, any Wish / Dragon give / Bomb Window it opens is
    resolved, and the state handed back is the next seat's Play Decision.

    `state_observer(seat, game_state, action)` sees every on-turn Play Decision
    the agents take (not the forced one), as in `play_full_round`.
    """
    asked = set(asked_tichu)
    tichu_callers = set(state.public.tichu_callers)
    grand_callers = state.public.grand_tichu_callers
    forced = forced_action
    taken = 0
    window_queue: list[int] = []
    for _ in range(_MAX_STEPS):
        in_window = bool(window_queue)
        actor = window_queue.pop(0) if in_window else state.public.current_player
        on_turn_play = state.public.pending_decision is None and not in_window
        was_forced = forced is not None
        if on_turn_play and not was_forced:
            if max_play_decisions is not None and taken >= max_play_decisions:
                return state, False
            taken += 1
        private = state.private_view(actor)
        if was_forced:
            action, forced = forced, None
        else:
            action = agents[actor].act(private)
            if state_observer is not None and on_turn_play:
                state_observer(actor, state, action)
        if (
            state.public.pending_decision is None
            and not isinstance(action, Pass)
            and actor not in asked
            and actor not in grand_callers
        ):
            asked.add(actor)
            if _calls(agents[actor], private, "tichu"):
                tichu_callers.add(actor)
                state = _with_callers(state, tichu=frozenset(tichu_callers))
        if in_window and isinstance(action, Pass):
            continue
        state, _, done, _ = step(state, action)
        if done:
            return state, True
        window_queue = list(bomb_window_queue(state, action, actor))
    raise RuntimeError(f"resume_round exceeded {_MAX_STEPS} steps")


def team_points(totals, seat: int) -> float:
    """Round outcome in points for `seat`'s team minus the other team's, from the
    `(team0, team1)` round totals."""
    return float(totals[seat % 2] - totals[1 - seat % 2])


def round_totals(final_scores, initial_scores) -> tuple[int, int]:
    return (final_scores[0] - initial_scores[0], final_scores[1] - initial_scores[1])


def leaf_features(state) -> tuple[np.ndarray, int]:
    """Features of a leaf `state` (an on-turn Play Decision) and the seat whose
    team the prediction is relative to — the seat to act, as in training."""
    seat = state.public.current_player
    return featurize_perfect_info(state, seat), seat


def _phase(state, seat: int) -> int:
    """0 = open, 1 = mid, 2 = late, by the acting seat's hand size (the miner's
    bands)."""
    n = len(state.hands[seat])
    return 0 if n >= 11 else (1 if n >= 6 else 2)


def emit_deal(agents, position, rng: random.Random, *, rows_per_traj: int = 12,
              top_k: int = 4) -> dict:
    """One deal as value rows: the greedy round, plus one branch that forces a
    random top-k alternative at a random non-forced decision and plays greedy
    from there.

    Returns `rows` (list of `(features, z, traj, turn, phase, call_live)`), where
    `z` is the exact outcome under greedy continuation, relative to the acting
    seat's team, in points / 100; and `pair`, the branch as a same-state action
    comparison: the indices into `rows` of the two successor states (or -1 where
    the action ended the round) and both outcomes relative to the deciding seat.
    `pair` is None when the round has no non-forced decision.
    """
    initial = position.state.public.scores
    base: list[tuple] = []

    def observe(seat, game_state, action):
        base.append((seat, game_state, action))

    result = play_full_round(agents, position.state, position.grand_prefixes,
                             state_observer=observe)
    if not base:
        return {"rows": [], "pair": None}
    base_final = result.total

    rows: list[tuple] = []

    def add_row(seat, game_state, final, traj, turn) -> int:
        z = team_points(final, seat) / _SCALE
        live = bool(game_state.public.tichu_callers
                    or game_state.public.grand_tichu_callers)
        rows.append((featurize_perfect_info(game_state, seat).astype(np.float16),
                     z, traj, turn, _phase(game_state, seat), live))
        return len(rows) - 1

    # The branch: a non-forced decision and one of the agent's own alternatives.
    open_turns = [
        i for i, (seat, gs, _a) in enumerate(base)
        if len(legal_actions_for(gs.private_view(seat))) > 1
    ]
    pair = None
    branch: list[tuple] = []
    branch_final = None
    turn = None
    if open_turns:
        turn = rng.choice(open_turns)
        seat, gs, chosen = base[turn]
        alts = candidate_alternatives(agents[seat], gs.private_view(seat), chosen,
                                      top_k=top_k)
        if alts:
            alt = rng.choice(alts)

            def observe_branch(s, game_state, action):
                branch.append((s, game_state, action))

            end_state, done = resume_round(
                agents, gs, forced_action=alt,
                asked_tichu=asked_tichu_from_public(gs.public),
                state_observer=observe_branch)
            assert done
            branch_final = round_totals(end_state.public.scores, initial)

    keep = set(rng.sample(range(len(base)), min(rows_per_traj, len(base))))
    if branch_final is not None and turn + 1 < len(base):
        keep.add(turn + 1)
    base_row = {}
    for i in sorted(keep):
        seat, gs, _a = base[i]
        base_row[i] = add_row(seat, gs, base_final, 0, i)

    if branch_final is not None:
        keep_b = set(rng.sample(range(len(branch)), min(rows_per_traj, len(branch))))
        if branch:
            keep_b.add(0)
        branch_row = {}
        for j in sorted(keep_b):
            seat_b, gs_b, _a = branch[j]
            branch_row[j] = add_row(seat_b, gs_b, branch_final, 1, turn + 1 + j)
        seat, gs, _chosen = base[turn]
        pair = {
            "base_row": base_row.get(turn + 1, -1),
            "alt_row": branch_row.get(0, -1),
            "z_base": team_points(base_final, seat) / _SCALE,
            "z_alt": team_points(branch_final, seat) / _SCALE,
            # +1 when the successor state's acting seat is on the decider's team.
            "base_sign": (1 if base[turn + 1][0] % 2 == seat % 2 else -1)
            if turn + 1 < len(base) else 0,
            "alt_sign": (1 if branch[0][0] % 2 == seat % 2 else -1) if branch else 0,
            "phase": _phase(gs, seat),
        }
    return {"rows": rows, "pair": pair}


def emit_siblings(agents, position, rng: random.Random, *, decisions: int = 4,
                  top_k: int = 3, plies: int = 5) -> dict:
    """One deal as same-state move comparisons: at a few non-forced decisions of
    the greedy round, the chosen move and each of the agent's top alternatives
    are played out greedy to the end in the true deal.

    Returns `rows`: `(features, z, decision, candidate, ply, sign, phase)` — the
    on-turn Play Decision reached `ply` decisions after the candidate (ply 0 is
    the depth-0 search leaf), `z` relative to that state's acting seat, `sign`
    +1 when that seat is on the decider's team; and `cands`:
    `(decision, candidate, zd, n_rows, phase)` with `zd` the exact outcome
    relative to the decider. Candidate 0 is the chosen move. Units: points / 100.
    """
    initial = position.state.public.scores
    base: list[tuple] = []
    result = play_full_round(agents, position.state, position.grand_prefixes,
                             state_observer=lambda s, g, a: base.append((s, g, a)))
    rows: list[tuple] = []
    cands: list[tuple] = []
    open_turns = [i for i, (seat, gs, _a) in enumerate(base)
                  if len(legal_actions_for(gs.private_view(seat))) > 1]
    picked = sorted(rng.sample(open_turns, min(decisions, len(open_turns))))
    for d, turn in enumerate(picked):
        seat, gs, chosen = base[turn]
        alts = candidate_alternatives(agents[seat], gs.private_view(seat), chosen,
                                      top_k=top_k)
        if not alts:
            continue
        phase = _phase(gs, seat)
        lines = [(base[turn + 1: turn + 1 + plies], result.total)]
        for alt in alts:
            branch: list[tuple] = []
            end_state, done = resume_round(
                agents, gs, forced_action=alt,
                asked_tichu=asked_tichu_from_public(gs.public),
                state_observer=lambda s, g, a, b=branch: b.append((s, g, a)))
            assert done
            lines.append((branch[:plies], round_totals(end_state.public.scores, initial)))
        for c, (states, final) in enumerate(lines):
            zd = team_points(final, seat) / _SCALE
            cands.append((d, c, zd, len(states), phase))
            for ply, (s, g, _a) in enumerate(states):
                sign = 1 if s % 2 == seat % 2 else -1
                rows.append((featurize_perfect_info(g, s).astype(np.float16),
                             sign * zd, d, c, ply, sign, phase))
    return {"rows": rows, "cands": cands}


class FullStateValue:
    """A trained full-state value net behind `predict(features) -> points`."""

    def __init__(self, model, *, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        self._device = device
        self._model = model.to(device).eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        """`(N, PERFECT_INFO_DIM)` features -> `(N,)` predicted round outcome in
        points, relative to the team of the seat the features were built for."""
        torch = self._torch
        x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self._device)
        with torch.no_grad():
            return (self._model(x) * _SCALE).cpu().numpy()


def save_full_state_value(model, path, *, hidden: int, depth: int,
                          input_dim: int = PERFECT_INFO_DIM, note: str = "") -> None:
    import torch

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": int(input_dim), "hidden": int(hidden), "depth": int(depth),
        "featurizer_version": FEATURIZER_VERSION, "note": note,
    }, path)


def load_full_state_value(path, *, device: str = "cpu") -> FullStateValue:
    import torch

    from tichu_training.awr.value_baseline import ValueBaseline

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("featurizer_version") != FEATURIZER_VERSION:
        raise ValueError(
            f"full-state value featurizer_version {blob.get('featurizer_version')!r} "
            f"!= harness {FEATURIZER_VERSION!r}")
    if blob["input_dim"] != PERFECT_INFO_DIM:
        raise ValueError(
            f"full-state value input_dim {blob['input_dim']} != {PERFECT_INFO_DIM}")
    model = ValueBaseline(blob["input_dim"], hidden=blob["hidden"], depth=blob["depth"])
    model.load_state_dict(blob["state_dict"])
    return FullStateValue(model, device=device)
