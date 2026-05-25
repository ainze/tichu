"""Held-out move-prediction evaluator.

Replays held-out human games and scores the agent's policy against the
recorded action at every decision state. Reports top-1 (always) and
top-5 (when the agent overrides `rank_actions`) accuracy bucketed by
decision type.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator

from tichu_engine.legality import ConcreteAction
from tichu_engine.state import PrivateState
from tichu_ml.agent import Agent
from tichu_training.bsw.records import ParsedAction, ParsedGame
from tichu_training.bsw.replay import replay_round


_PARSED_KIND_TO_DECISION_TYPE: dict[str, str] = {
    "play": "play",
    "pass": "play",
    "schupfen": "schupfen",
    "wish": "wish",
    "dragon_give": "dragon_assignment",
}


_TOP_K = 5


@dataclass(frozen=True)
class EvalDecision:
    """One human decision state for move-prediction evaluation.

    `decision_type` is one of 'play', 'schupfen', 'wish', 'dragon_assignment' —
    matching the BC model's heads.
    """

    decision_type: str
    private_state: PrivateState
    human_action: ConcreteAction


def evaluate_move_prediction(
    agent: Agent,
    decisions: Iterable[EvalDecision],
    *,
    max_decisions: int | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Run `agent` on each decision; aggregate top-1 (and top-5 if supported)
    accuracy by `decision_type`.

    Returns a dict shaped `{decision_type: {"top1": float, "top5": float | None, "n": int}}`.
    `top5` is `None` for a decision type whenever ANY decision under that bucket
    had a non-ranking agent response, so the caller can distinguish 0% from
    "ranking not implemented".
    """
    n_by_type: dict[str, int] = defaultdict(int)
    top1_hits: dict[str, int] = defaultdict(int)
    top5_hits: dict[str, int] = defaultdict(int)
    has_ranking: dict[str, bool] = defaultdict(lambda: True)

    for i, decision in enumerate(decisions):
        if max_decisions is not None and i >= max_decisions:
            break
        n_by_type[decision.decision_type] += 1

        chosen = agent.act(decision.private_state)
        if chosen == decision.human_action:
            top1_hits[decision.decision_type] += 1

        ranking = agent.rank_actions(decision.private_state)
        if ranking is None:
            has_ranking[decision.decision_type] = False
        else:
            if decision.human_action in ranking[:_TOP_K]:
                top5_hits[decision.decision_type] += 1

    out: dict[str, dict[str, float | int | None]] = {}
    for dt, n in n_by_type.items():
        top1 = top1_hits[dt] / n if n else 0.0
        top5: float | None
        if has_ranking[dt]:
            top5 = top5_hits[dt] / n if n else 0.0
        else:
            top5 = None
        out[dt] = {"top1": top1, "top5": top5, "n": n}
    return out


def decisions_from_game(parsed_game: ParsedGame) -> Iterator[EvalDecision]:
    """Walk one parsed game and yield one `EvalDecision` per real engine
    decision (skipping Tichu/Grand-Tichu calls and phantom passes the parser
    emits but the engine doesn't step through).
    """
    for parsed_round in parsed_game.rounds:
        result = replay_round(parsed_round)
        for (parsed, engine_action), state in zip(
            result.decisions, result.pre_decision_states
        ):
            if state is None:
                continue
            decision_type = _PARSED_KIND_TO_DECISION_TYPE.get(parsed.kind)
            if decision_type is None:
                continue
            yield EvalDecision(
                decision_type=decision_type,
                private_state=state.private_view(parsed.player),
                human_action=engine_action,
            )

