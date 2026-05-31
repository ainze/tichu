# ADR-0023: Tichu / Grand-Tichu calls are served via a separate `/call` endpoint, not `act()`

- **Status:** Accepted
- **Date:** 2026-05-31
- **Related:** [ADR-0007](0007-calls-are-standalone-networks.md), [ADR-0012](0012-schupfen-is-a-standalone-network.md)

## Context

Play, schupfen, mahjong-wish and dragon-give are all engine **pending
decisions** — they surface through `legal_actions_for` / the `pending_decision`
field, so the `Agent.act(PrivateState) -> ConcreteAction` interface (and the
`POST /act` endpoint) serves them uniformly. Tichu / Grand-Tichu **calls are
not** pending decisions: there is no `CallPending` in `tichu_engine.state`
(only Dragon/Wish/Schupfen), and calls are stamped into
`tichu_callers` / `grand_tichu_callers` *before* the engine step loop. So
`act()` is never asked for a call, and the call networks (`tichu_final.bin` /
`grand_final.bin`, ADR-0007) had no inference hook.

## Decision

Expose the call decision as a **separate, binary endpoint** —
`POST /call {difficulty, kind: "tichu"|"grand", private_state} -> {call: bool}`
— backed by a new `MLAgent.should_call(private_state, kind) -> bool` that runs
the matching Call Network. The external UI calls it at the two trained moments:
Grand-Tichu at the 8-card state, Tichu at the player's first non-pass play
(ADR-0018). `act()` is left unchanged and continues to serve only engine
pending decisions.

## Rationale / rejected alternatives

- **Overloading `/act` with a synthesised call-pending** (the UI fakes a
  pending-decision in `private_state`, `/act` returns `CallTichu`/decline):
  rejected — it requires inventing a pending-decision type the engine does not
  have and couples the wire format to a fiction the engine can't validate.
- **Making calls a real engine pending decision**: rejected for now — a much
  larger engine change (new state, new step transitions, replay/featurizer
  impact) than the inference feature warranted.

## Consequences

- The call contract is its own thing the UI binds to; changing it is a
  breaking change for the client (hence this record).
- Baseline tiers (`RuleAgent`, and any ML tier without a call net wired) have
  no `should_call` and **decline** — declining is always legal and never
  stalls, mirroring the `/act` random-legal fallback contract.
- Calls remain outside the eval harness: `play_round` still plays with empty
  caller sets, so the tournament measures play strength, not call quality.
