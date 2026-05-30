# ADR-0007: Tichu and Grand-Tichu Calls are standalone networks, not Heads on the shared Trunk

- **Status:** Accepted
- **Date:** 2026-05-25
- **Related:** [ADR-0001](0001-trunk-architecture.md), [ADR-0003](0003-easy-difficulty-is-a-baseline.md), [ADR-0018](0018-tichu-call-featurises-at-first-non-pass-play.md) (precise Tichu featurise moment)

## Context

The PRD describes a "multi-head behavioral cloning model — sharing a trunk
across the structurally distinct decisions of grand-tichu call,
regular-tichu call, card passing, in-trick play, mahjong wish, and
dragon-give-direction." Read literally, that's six Heads on one Trunk.

The implementation ships only **four Heads** on the shared Trunk
(`tichu_training/bc/heads.py:17-20` — play / schupfen / wish /
dragon_assignment) and **two standalone Call Networks** for Tichu Call and
Grand-Tichu Call (`tichu_training/bc/call_model.py`, trained by
`train_calls`).

## Decision

**Tichu Call and Grand-Tichu Call are trained as standalone networks,
not as Heads sharing the BC Trunk.** An ML Agent loads three Checkpoints:
one Policy Network Checkpoint (Trunk + 4 Heads) plus one Tichu Call
Checkpoint plus one Grand-Tichu Call Checkpoint.

## Rationale

1. **Call decisions happen at frozen moments with frozen inputs.**
   Grand-Tichu Call sees only the first 8 cards of the player's Hand and
   nothing else. Tichu Call sees the full 14-card Hand but no Trick or
   public-play history (calls happen before the first Play). The Play /
   Wish / Dragon-Assignment Heads see rich mid-Round state. The trunk
   representation optimised for mid-Round state is not the representation
   a Call needs — most of the 16,568-float Feature Vector is zeroed at
   call time anyway.
2. **Call aggressiveness is a tunable product knob.** Calling too rarely
   is boring; calling too often is reckless. Separating Calls from the
   Policy Network lets the calling threshold be tuned (or even swapped
   out for a calibrated rule) without retraining the Trunk or risking
   any regression in mid-Round play.
3. **Call training data is sparse.** A Round has at most two Tichu Call
   Decisions and four Grand-Tichu Call Decisions — versus dozens of Play
   Decisions. Co-training Calls on the Trunk would mean the Call loss is
   gradient-starved relative to Play, OR requires careful per-Decision
   loss weighting to compensate. A standalone network sidesteps this.
4. **Different training cadence and stopping criteria.** A Call Network
   may converge in a fraction of the BC epochs needed for the Policy
   Network. Separating them means each can have its own training schedule
   without one bottlenecking the other.
5. **Inference latency budget.** Calls happen at most twice per Round
   per Player; the cost of an independent forward pass at those moments
   is irrelevant to the 99p latency target (PRD #48). The forward pass
   cost saved by sharing the Trunk does not justify the coupling.

## Consequences

- The ML Agent loads **three** Checkpoints, not one. The Difficulty Spec
  must accept three Checkpoint paths per ML tier (or a directory
  convention that bundles them).
- A Refined Checkpoint refines **only** the Policy Network. The Call
  Networks are not refined by AWR in the current pipeline. If Call
  quality plateaus, a separate AWR-for-calls pipeline is the next step,
  but it does not block Policy refinement.
- Call Networks have their own training CLI (`train_calls`) separate
  from `train_bc`. CI must build both before any ML tier can be
  promoted.
- The Featurizer and Action Space versions still pin Call Network
  Checkpoints, even though the Calls do not use the full Feature Vector
  — this keeps the load-time version checks uniform across all
  Checkpoint kinds (consistent with [ADR-0004](0004-payload-agnostic-checkpoint.md)).
- The PRD's "six-head shared trunk" framing should be read as "six
  Decision types served by one trained system", not as "six Heads on
  one literal Trunk".

## Rejected alternatives

- **Six Heads on the shared Trunk.** Rejected for reasons 1–4 above.
  The trunk representation is dominated by mid-Round inputs; Calls would
  either be under-trained or distort the trunk.
- **One standalone Call Network covering both Tichu and Grand-Tichu.**
  Rejected because the inputs differ (8 vs 14 cards) and the strategic
  decisions are different in kind. Two small networks beat one network
  with conditional inputs.
