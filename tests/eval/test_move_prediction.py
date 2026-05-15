"""Move-prediction evaluator + Agent.rank_actions extension."""

from dataclasses import dataclass

from tichu_engine.legality import PASS, Action
from tichu_engine.state import PrivateState

from tichu_eval.move_prediction import EvalDecision, evaluate_move_prediction
from tichu_ml.agent import Agent
from tichu_ml.random_agent import RandomAgent


# A stand-in state — the evaluator never inspects the state directly,
# it just hands it to the agent. We use a sentinel to avoid building real
# PrivateStates in unit tests focused on the aggregator.
@dataclass(frozen=True)
class _DummyState:
    label: int


class _ConstantAgent(Agent):
    """Always returns `action`. Implements ranking by reporting `top_k`."""

    def __init__(self, *, action: Action, top_k: list[Action] | None = None):
        self._action = action
        self._top_k = top_k

    def act(self, private_state):
        return self._action

    def rank_actions(self, private_state):
        return self._top_k


def test_agent_default_rank_actions_is_none():
    # The base Agent contract: rank_actions defaults to None (not implemented).
    assert RandomAgent(seed=0).rank_actions(None) is None


def test_top1_perfect_when_agent_matches_human():
    agent = _ConstantAgent(action=PASS)
    decisions = [
        EvalDecision("play", _DummyState(i), PASS) for i in range(10)
    ]
    out = evaluate_move_prediction(agent, decisions)
    assert out["play"]["top1"] == 1.0
    assert out["play"]["n"] == 10


def test_top1_zero_when_agent_never_matches():
    agent = _ConstantAgent(action=PASS)
    other_action = "OTHER"  # type: ignore[assignment]
    decisions = [
        EvalDecision("play", _DummyState(i), other_action) for i in range(4)
    ]
    out = evaluate_move_prediction(agent, decisions)
    assert out["play"]["top1"] == 0.0
    assert out["play"]["n"] == 4


def test_top5_is_none_when_agent_does_not_override_ranking():
    agent = _ConstantAgent(action=PASS, top_k=None)
    decisions = [EvalDecision("play", _DummyState(0), PASS)]
    out = evaluate_move_prediction(agent, decisions)
    assert out["play"]["top5"] is None


def test_top5_reports_accuracy_when_agent_provides_ranking():
    sentinel = "TARGET"
    agent = _ConstantAgent(action=PASS, top_k=[PASS, "X", sentinel, "Y", "Z"])
    decisions = [EvalDecision("play", _DummyState(0), sentinel)]
    out = evaluate_move_prediction(agent, decisions)
    # top1 fails (agent picks PASS, human picked sentinel).
    assert out["play"]["top1"] == 0.0
    # top5 succeeds — sentinel is in the top 5.
    assert out["play"]["top5"] == 1.0


def test_top5_misses_when_target_outside_top_five():
    sentinel = "OUTSIDER"
    agent = _ConstantAgent(action=PASS, top_k=["A", "B", "C", "D", "E", sentinel])
    decisions = [EvalDecision("play", _DummyState(0), sentinel)]
    out = evaluate_move_prediction(agent, decisions)
    assert out["play"]["top5"] == 0.0


def test_aggregates_by_decision_type():
    agent = _ConstantAgent(action=PASS)
    decisions = [
        EvalDecision("play", _DummyState(0), PASS),
        EvalDecision("play", _DummyState(1), "X"),
        EvalDecision("wish_rank", _DummyState(2), PASS),
    ]
    out = evaluate_move_prediction(agent, decisions)
    assert out["play"]["n"] == 2
    assert out["play"]["top1"] == 0.5
    assert out["wish_rank"]["n"] == 1
    assert out["wish_rank"]["top1"] == 1.0


def test_max_decisions_caps_evaluation():
    agent = _ConstantAgent(action=PASS)
    decisions = [EvalDecision("play", _DummyState(i), PASS) for i in range(100)]
    out = evaluate_move_prediction(agent, decisions, max_decisions=10)
    assert out["play"]["n"] == 10


def test_decisions_from_game_yields_decisions_from_a_sample_game():
    """End-to-end: parsed BSW game → walked decisions → evaluator."""
    from pathlib import Path

    from tichu_eval.move_prediction import decisions_from_game
    from tichu_training.bsw.parser import parse_tch

    sample = (
        Path(__file__).resolve().parents[2] / "sample" / "2417500.tch"
    )
    game = parse_tch(sample.read_text(encoding="utf-8"), game_id="2417500")
    decisions = list(decisions_from_game(game))
    assert len(decisions) > 50  # plenty of decisions per multi-round game

    seen_types = {d.decision_type for d in decisions}
    # At minimum every game has play decisions and schupfen pass_card decisions.
    assert "play" in seen_types
    assert "pass_card" in seen_types

    # RuleAgent runs end-to-end without crashing on every decision.
    from tichu_ml.rule_agent import RuleAgent
    agent = RuleAgent()
    out = evaluate_move_prediction(agent, decisions, max_decisions=200)
    assert "play" in out
    assert 0.0 <= out["play"]["top1"] <= 1.0
    # RuleAgent doesn't implement rank_actions → top5 is None.
    assert out["play"]["top5"] is None
