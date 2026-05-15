---
id: "011b"
title: "Single-deal runner (play a deal to round-end with seated agents)"
type: AFK
parent: "011"
blocked_by: ["011a"]
---

## What to build

Driver that runs one deal from a starting `GameState` until the round terminates.

- `play_deal(agents: tuple[Agent, Agent, Agent, Agent], initial_state: GameState) -> tuple[int, int]` — repeatedly calls `agents[current_player].act(private_view)` and applies `engine.step` until the round resolves. Returns the team-0 and team-1 score deltas relative to the initial scores.
- BombInterrupts are out of scope at this slice (the engine supports them, but legality enumeration in `legal_actions_for` only returns in-turn actions). Document as a known limitation.
- Tichu/Grand Tichu *calling* is out of scope at this slice (separate networks per #010). Each deal is played with empty caller sets.

## Acceptance criteria

- [ ] Round of 4 `RandomAgent` instances (seeded) plays to completion without error and returns a 2-tuple of ints.
- [ ] Same seeded agents + same `initial_state` ⇒ identical score deltas across two runs (determinism).
- [ ] `RuleAgent` x4 plays to completion on at least 3 deals from a small pool without raising.
- [ ] Score deltas sum to the engine's expected total (card points + any bonuses).

## Blocked by

- [#011a Fixed seeded deal pool](011a-deal-pool.md)
