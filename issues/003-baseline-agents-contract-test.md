---
id: "003"
title: "Baseline agents + agent-interface contract test"
type: AFK
blocked_by: ["002"]
stories: [41, 8]
---

## What to build

Define the `Agent` abstract interface and implement the two baseline agents that serve as permanent reference points for all evaluation runs.

**Agent interface.** A single abstract base class (or protocol) in `tichu_ml` with one required method: `act(private_state) -> action`. Every concrete agent — learned or rule-based — implements this interface. The interface must not carry any training-time state; it is inference-only.

**Random agent.** Selects uniformly at random from the legal-actions mask on each call. Used as the floor baseline in all tournament runs.

**Rule-based agent.** A hand-coded heuristic agent strong enough to beat Random reliably. Minimum viable heuristics: lead with singles or pairs, play bombs only when behind, call Tichu only with very strong hands, follow suit for passing. This agent does not need to be strong — it needs to be clearly stronger than Random so the tournament harness has a meaningful reference point, and it must never produce an illegal action.

**Contract test.** A parametrized test that runs every concrete `Agent` subclass through a complete random game and asserts that every action it returns is legal in the engine's state at that step. This is the safety net that ensures new agents don't silently produce illegal moves.

## Acceptance criteria

- [ ] `Agent` abstract interface is defined and documented
- [ ] `RandomAgent` uniformly samples legal actions and never returns an illegal one
- [ ] `RuleAgent` beats `RandomAgent` by a positive average score margin over 1,000 deals
- [ ] Agent contract test runs both agents through a full game without illegal actions
- [ ] Both agents are registered with the eval harness's agent registry (even if the harness itself isn't built yet — just the registration mechanism)

## Blocked by

- [#002 Rules engine](002-rules-engine.md)
