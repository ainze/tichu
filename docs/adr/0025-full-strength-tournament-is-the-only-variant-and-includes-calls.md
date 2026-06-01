# ADR-0025: The Full-strength Tournament is the only variant and exercises the complete model stack (play + schupfen + both calls)

- **Status:** Accepted — supersedes [ADR-0006](0006-tournament-play-strength-vs-full-strength.md)
- **Date:** 2026-06-01
- **Related:** [ADR-0007](0007-calls-are-standalone-networks.md), [ADR-0012](0012-schupfen-is-a-standalone-network.md), [ADR-0018](0018-tichu-call-featurises-at-first-non-pass-play.md), [ADR-0023](0023-calls-served-via-separate-call-endpoint.md), [ADR-0024](0024-master-tier-conditions-on-top-skill-decile.md)

## Context

[ADR-0006](0006-tournament-play-strength-vs-full-strength.md) defined **two**
Tournament variants: Play-strength (Schupfen skipped) and Full-strength
(Schupfen played). Neither exercised the Tichu / Grand-Tichu Call Networks —
`tichu_eval/play.py` explicitly deferred calls (*"Tichu / Grand Tichu calling is
out of scope at this slice… composing call networks with a play agent is a
follow-up"*), and the Starting-Position Pool skipped Schupfen entirely
(`deal_initial_state`). So in practice every Tournament measured Play / Wish /
Dragon only; Schupfen and both Calls were never under test against each other.

We now want to measure the **strength of the models combined** — the actual
shipping product, which calls Grand-Tichu on its first 8 cards, Schupfens, calls
Tichu, and plays. The `MLAgent` already runs that full stack (it has
`should_call` and a Schupfen path); the gap was entirely in the eval harness.

## Decision

**There is one Tournament variant: Full-strength.** It drives the complete
product lifecycle per Round — **Grand-Tichu Call → Schupfen → Tichu Call → Play
→ Wish → Dragon** — each Decision served by the Agent's own networks. The
Play-strength variant is retired as a default.

Concretely:

1. **Single round runner `tichu_eval/play_full.py::play_full_round`**, separate
   from the play-only `play_round` (which stays in-tree, deprecated). It
   orchestrates the two out-of-band Call phases the engine does not drive
   (the engine only reads `tichu_callers` / `grand_tichu_callers` at
   `_finalise_round`):
   - **Grand-Tichu:** build the synthetic `(8,8,8,8)` deal-time state from each
     seat's **Grand-Tichu Prefix** (first 8 cards in deal order) and ask each
     seat `should_call(ps, "grand")` independently — no prior callers visible,
     matching `grand_tichu_examples_for_round`.
   - **Schupfen:** engine-driven from a `deal_for_schupfen` start state.
   - **Tichu:** the first time a non-grand-caller seat makes a non-Pass Play,
     ask `should_call(ps, "tichu")` on that pre-play state (ADR-0018) and inject
     into `tichu_callers` before applying the Play.
   Agents without `should_call` (RuleAgent, RandomAgent) decline all Calls.

2. **Starting Position is redefined as the pre-Schupfen deal-time state.**
   Generation preserves deal order so the Grand-Tichu Prefix is a deterministic,
   genuine prefix of the same 14 cards played that Round; the Pool file stores
   each seat's 14-card hand **and its 8-card prefix explicitly** (the 8/6 split
   survives without persisting raw order). New file
   `full_position_pool_s{seed}_n{n}.parquet`, `(seed, n)` identity unchanged.

3. **Per-agent network bundles in the Tournament config.** Each agent spec names
   its own `checkpoint_path` / `schupfen_path` / `tichu_call_path` /
   `grand_call_path` / `skill_decile`, freely mixing training runs — **at a
   single fixed featurizer version** (all exports load against the harness's
   global `FEATURIZER_VERSION`; cross-featurizer comparison requires separate
   runs). An ablation agent omits the schupfen/call paths.

4. **`eval_matrix` gains a `variant` key (`play_strength | full_strength`),
   defaulting to `full_strength`.** New configs are Full-strength with no
   ceremony; the existing `eval_100k_v5.yaml` and `eval_skill_ab_100k_v5.yaml`
   (and ADR-0024's reproduce command) keep working by adding `variant:
   play_strength`.

5. **Larger default `n` and a call-bonus breakdown.** Calls add ±100/±200
   per-Round variance, so Full-strength Pools default to ~2000 Starting
   Positions. `play_full_round` returns the Call-bonus component of each Round's
   delta alongside the total, and the matrix reports both — partially recovering
   the play/schupfen/call attribution lost by retiring Play-strength.

## Consequences

- A Full-strength regression no longer tells you **which** axis moved without
  the call-bonus breakdown (and, for play-only debugging, re-running the
  deprecated Play-strength path).
- The Grand-Tichu Prefix uses a deck-order 8/6 split, not a faithful BSW dealing
  sequence — acceptable because the split is fixed/reproducible and the
  Grand-Tichu Call Network only ever saw 8-card states regardless of order.
- ADR-0006's `play_strength_matrix` / `full_strength_matrix` dual-file naming is
  moot; there is one `matrix.parquet`, now with call-bonus columns.

## Rejected alternatives

- **Keep both variants, add calls only to Full-strength.** Leaves two pools and
  two runners to maintain for a Play-strength number we no longer ship on.
- **Hard-replace the tournament path (delete Play-strength).** Silently breaks
  ADR-0024's documented one-command re-measure; the additive `variant` key costs
  almost nothing and preserves it.
- **A third separately-named "Full-stack" variant.** Leaves "Full-strength"
  permanently misnamed (call-silent); better to make it mean *actually full*.
