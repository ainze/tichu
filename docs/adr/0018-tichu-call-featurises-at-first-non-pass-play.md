# ADR-0018: Tichu Call featurises at the seat's first non-Pass Play state

- **Status:** Accepted
- **Date:** 2026-05-30
- **Related:** [ADR-0007](0007-calls-are-standalone-networks.md), [ADR-0011](0011-bc-training-replay-on-the-fly.md)

## Context

[ADR-0007](0007-calls-are-standalone-networks.md) §rationale 1 states that
the Tichu Call Network sees "the full 14-card Hand but no Trick or
public-play history (calls happen before the first Play)". Implementing
the parquet Call-Network adapter (the deferred follow-up from ADR-0007)
exposed that this is underspecified for the negative-example side:

- BSW rules let a player call Tichu at any moment **before they play
  their first card** — including after seeing one or more opponents Play.
  A Pass doesn't burn the Tichu window because no card has hit the table.
- Positive examples (seats that called) carry the parsed `tichu` action
  somewhere in the play sequence; its exact engine-state context varies
  per call.
- Negative examples (seats that didn't call) have no parsed-call moment
  at all. They need an explicitly chosen featurise state.

For the binary classifier's targets to be coherent, positives and
negatives must be featurised at states drawn from the same distribution.

## Decision

For each seat, the Tichu Call is featurised at the engine state
immediately **before that seat's first non-Pass Play** in the round.
The rule applies symmetrically to positive and negative examples. A
seat that never made a non-Pass Play in the round contributes **no**
Tichu Call example.

Concretely (per `ParquetCallDataset.__iter__` in
`tichu_training/bc/call_training.py`):

1. Call `replay_round(parsed_round)`.
2. For each seat 0..3, scan `zip(replay.decisions,
   replay.pre_decision_states)` for the first index where
   `parsed_action.player == seat`, `parsed_action.kind == "play"`, and
   `pre_state is not None`.
3. Snapshot `pre_state`. `pre_state.private_view(seat)` → `featurize` →
   `CallExample` with `target = (seat in parsed_round.tichu_callers)`.
4. If no such index exists, drop the seat (no example for this round).

Grand-Tichu's featurise moment is unaffected: a synthetic deal-time
`GameState` built from `parsed_round.pre_deal_hands` with
`hand_sizes=(8,8,8,8)`, no Trick, no pending decision. Per seat,
`private_view` and featurise.

## Rationale

1. **Matches BSW's rule.** "Played a card" excludes Passes — the
   Tichu window stays open through in-trick Passes. The rule
   "first non-Pass Play" captures this exactly.
2. **Symmetric positives ↔ negatives.** Non-callers have no logged
   moment-they-would-have-called. Binding both halves of the binary
   to the same engine-defined moment keeps the target distribution
   uniform across labels.
3. **Reuses `replay_round`.** The first-non-Pass-Play pre-state is
   already in `replay.pre_decision_states`, alongside the legal-action
   cache. No new replay surface, no extra engine cost beyond what BC
   already pays per [ADR-0011](0011-bc-training-replay-on-the-fly.md).
4. **Sharpens ADR-0007 §rationale 1.** Tichu features can include up to
   ~3 prior Plays of public history (the seat's predecessors in
   trick 1) — not the frozen pre-trick state ADR-0007's prose implied.

## Consequences

- ADR-0007 §rationale 1's "no Trick or public-play history" should be
  read as the **upper bound at deal time**, not the actual featurise
  state. Grand-Tichu honours it; Tichu can see up to three prior Plays.
- BSW players who logged their Tichu call after their own first non-Pass
  Play (rule-breaking; rare) are featurised as if they had decided
  earlier. Pragmatic; the classifier still learns a coherent target.
- A seat that never made a non-Pass Play in a round (partner swept
  before their turn — vanishingly rare) contributes no Tichu example.
  Should be tracked in the adapter's drop counter when one is added.
- Loss vs. ADR-0007's literal reading: small. The mid-trick-1 context
  visible to a late-position Tichu caller is exactly the public
  information a real BSW caller would weigh.

## Rejected alternatives

- **Uniform post-Schupfen state for all four seats.** Symmetric and
  simple, but doesn't match BSW's window — a real caller absorbs public
  information up to their first card. This would systematically drop
  signal that the featurizer already encodes.
- **Asymmetric featurise rule** (positives at the literal parsed
  `tichu` action state; negatives at first-non-Pass-Play). Most
  faithful to BSW timing for positives, but breaks distribution
  symmetry: the two halves of the binary target are drawn from
  different state distributions, and the classifier learns whatever
  selection bias separates them rather than the call decision itself.
- **First time the seat is `current_player`** (the narrower C variant).
  Closes the Tichu window the first time a seat could Pass — even
  though BSW would let them call after passing. Loses positives
  unnecessarily.
