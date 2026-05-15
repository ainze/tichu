---
id: "011e"
title: "Held-out move prediction mode (top-1 + top-5 accuracy vs human action)"
type: AFK
parent: "011"
blocked_by: ["011b"]
---

## What to build

A second eval mode independent of self-play tournaments: replay held-out human games and score the agent's policy against the recorded human action at every decision state.

- Extend `Agent` with an optional `rank_actions(private_state) -> list[Action]` method. Default implementation returns `[self.act(private_state)]` (top-1 only). Learned agents that have a head-logit distribution override this to return all legal actions sorted by predicted probability.
- `evaluate_move_prediction(agent: Agent, held_out_games: Iterable[ParsedGame], *, max_decisions: int | None = None) -> dict[str, dict[str, float]]` — for each decision state in each game, ask the agent to rank, compare to the human's recorded action, and aggregate:
    - top-1 accuracy per `decision_type` (play / pass_card / wish_rank / dragon_give / tichu / grand_tichu where applicable)
    - top-5 accuracy per `decision_type` (only counted for agents that override `rank_actions`; otherwise reported as `None`)
- The held-out set is identified by a flag on the parser CLI or a separate directory; not in this slice's scope to define, but the evaluator must accept any iterable of parsed games.
- CLI subcommand (or new flag on `eval_matrix`): `eval_matrix --mode move_prediction --held-out <dir>` writes a CSV with rows `agent, decision_type, top1, top5, n`.

## Acceptance criteria

- [ ] `evaluate_move_prediction(RuleAgent(), small_held_out_set)` returns a dict with at least the `play` decision type and a non-negative `top1` accuracy.
- [ ] When the agent's `rank_actions` falls back to `[act]`, the reported `top5` for that decision type is `None`.
- [ ] Synthetic-fixture test where the agent is hand-crafted to always pick the recorded human action reports `top1 == 1.0`.
- [ ] CLI mode writes the CSV with the documented columns.

## Blocked by

- [#011b Single-deal runner](011b-play-deal.md) — for the shared decision-loop driver.
