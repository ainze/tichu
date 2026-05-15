---
id: "015a"
title: "JSON codec for PrivateState"
type: AFK
parent: "015"
blocked_by: []
---

## What to build

`tichu_inference.codec` — full round-trip JSON serializer/deserializer for `PrivateState`. The HTTP `POST /act` request body is exactly this shape.

- Cards serialize as integer IDs in `[0, 56)` using the existing deal-pool table (`fresh_deck()` order).
- Combinations serialize as `{kind, cards, ...}` with kind-specific extras (e.g. Phoenix `as_rank` for Single; `phoenix_as_rank` for Straight).
- `Trick` carries its `plays` (each a `(player, combination)`), `leader`, and `passes`.
- `PendingDecision`: a tagged union `{kind: "schupfen" | "dragon_give" | "mahjong_wish", ...}`. Schupfen carries the per-player `submitted` tuple; the others carry their payload.
- `PublicState` covers everything: current_player, hand_sizes, scores, trick, mahjong_wish, pending_decision, round_points_by_player, out_order, tichu_callers, grand_tichu_callers.

Functions:
- `private_state_to_json(ps: PrivateState) -> dict`
- `private_state_from_json(payload: dict) -> PrivateState`
- `action_to_json(action: Action) -> dict` (output side of the HTTP response)

## Acceptance criteria

- [ ] Round-trip: `from_json(to_json(ps)) == ps` for a deal-initial state (no trick, no pending) and for at least one mid-trick state with a non-trivial trick.
- [ ] Card encoding stable across runs.
- [ ] Pending-decision encodings: all three kinds covered with tests.
- [ ] Helpful error on malformed payload (missing required key → `ValueError`).

## Blocked by

- None — can start immediately.
