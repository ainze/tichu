---
id: "002"
title: "Rules engine — card types, combinations, legal-action mask"
type: AFK
blocked_by: ["001"]
stories: [9, 10, 11, 12, 13, 14, 15, 16]
---

## What to build

Implement the full Tichu rules engine inside `tichu_engine`. This is the most load-bearing component in the project: every other slice either imports it directly or calls it over HTTP. Correctness here is the only thing that matters.

**Card and combination representation.** All card and combination types are immutable typed objects (dataclasses or named tuples). Combination types to implement: single, pair, triple, full house, straight (length ≥ 5), pair-step (consecutive pairs), four-of-a-kind bomb, straight-flush bomb. Special cards — Phoenix (wild, value 0.5 above the card it substitutes), Dragon (highest single, scores 25 points, given to an opponent), Mahjong (lowest single, holder leads, may name a wish rank), Dog (passed directly to partner, no points) — must be modelled with their exact Tichu semantics, not simplified away.

**State separation.** `PublicState` and `PrivateState` are distinct types. The engine returns the correct view for the acting player at each step. This separation is the contract between the engine and everything upstream; getting it wrong now costs more later.

**Legal-action mask.** Every game state exposes a bitmask over the full canonical action space. All agents — learned, rule-based, or human — check legality through this mask, never through ad-hoc logic. The mask must correctly handle: out-of-turn bombs interrupting a trick, Mahjong wish fulfillment forcing a legal play if possible, Phoenix substitution legality, Dog-to-partner semantics, Dragon-give-direction prompt after winning a trick with the Dragon.

**Step interface.** `engine.step(state, action) -> (next_state, reward, done, info)` — one action per call, deterministic, no I/O. Replaying a logged game is mechanically the same as running a rollout.

**Tests.** Property-based tests using Hypothesis over the combination enumerator: any dealt 14-card hand must enumerate combinations satisfying no-duplicates, all-legal, rank-comparable invariants. Canonical-example tests for every special-card rule cited in the Tichu rulebook: Phoenix-in-straight, Phoenix-in-pair-step, Mahjong wish forcing, Dog pass, Dragon-give, out-of-turn bomb interrupting a pass cycle. Determinism test: same inputs → byte-identical state outputs.

The engine must never import PyTorch or any ML dependency.

## Acceptance criteria

- [ ] All card types and all combination types are implemented as immutable objects
- [ ] `PublicState` and `PrivateState` are distinct types; the engine returns the correct view for the acting player
- [ ] `legal_actions_mask` is exposed on every state; mask is correct for all special-card rules
- [ ] `engine.step` is deterministic and stateless (no global mutable state)
- [ ] A full 4-player game can be run to completion without crash or illegal move
- [ ] Hypothesis property-based tests pass over the combination enumerator
- [ ] Canonical rulebook-case tests pass (Phoenix-in-straight, Mahjong wish, Dog pass, Dragon-give, out-of-turn bomb)
- [ ] `tichu_engine` has zero ML-library imports

## Blocked by

- [#001 Project scaffolding & CI skeleton](001-project-scaffolding-ci.md)
