# ADR-0008: Every parsed BSW game is validated by full engine replay

- **Status:** Accepted
- **Date:** 2026-05-25
- **Related:** [ADR-0002](0002-intent-level-action-space.md)

## Context

The BSW corpus is ~2.4 million games of Tichu played on Brettspielwelt.
Two failure modes are catastrophic for training:

1. **Parser bugs.** A subtly mis-parsed action turns into a wrong
   training label and silently corrupts the BC distribution.
2. **Engine bugs.** A rule the engine implements incorrectly (e.g. Phoenix
   substitution, Mahjong Wish enforcement, Dog-passes-to-partner) means
   the engine's view of legality disagrees with BSW's — and the model
   ends up learning a policy for the wrong rules.

Both classes of bug can be present without producing any visible
parse-time error: the parser produces a plausible record, the engine
produces a plausible state, and only the final score reveals that
something diverged mid-Round.

## Decision

**Every parsed BSW game is replayed action-by-action through the
rules engine, and the engine's final scores are asserted to equal the
scores BSW recorded.** Games where the scores diverge are reported as
`GameValidationResult.score_mismatch` and excluded from training. Games
where the engine refuses an action (illegal play, missing Wish target,
etc.) are reported as replay errors and excluded.

Implementation: `tichu_training/bsw/validate.py` calls `replay_round`
from `tichu_training/bsw/replay.py` for every parsed Round, compares
`replay.final_state.public.scores` against the BSW-reported scores, and
aggregates results across the corpus via `validate_corpus`.

## Rationale

1. **Catches parser bugs and engine bugs symmetrically.** Both classes
   of bug surface as score mismatches. Neither can be detected by
   examining the parser or engine in isolation. The replay is the only
   place the two meet.
2. **The 2.4M-game corpus is the ultimate fuzz test.** Hand-written
   tests cover known rules edge cases; the corpus covers the edge
   cases nobody thought to write. Phoenix-in-pair-step with a Mahjong
   Wish active and an out-of-turn Bomb interrupt — that scenario is in
   the corpus somewhere.
3. **The cost is paid once.** Validation runs at parse time, before any
   model is trained. Training reads pre-validated parquet shards and
   does not re-replay.
4. **Silently corrupted training data is the worst failure.** A model
   trained on subtly wrong labels is harder to debug than a model that
   refuses to train. Loud rejection at parse time is strictly better
   than silent acceptance.
5. **It defines "ground truth" precisely.** The engine + BSW agreement
   is the project's operational definition of "correct Tichu". Anything
   that passes replay is considered playable; anything that fails is
   excluded. This frees downstream code from speculative defensive
   checks.

## Consequences

- Adding a new Action shape (e.g. a future Tichu variant) requires
  matching parser support AND engine support AND BSW examples to
  validate against. No one of the three is enough.
- Engine rule changes are gated on a corpus revalidation pass. A change
  that reduces the score-match rate is a regression even if every
  unit test passes.
- Games excluded by replay-validation must be logged with reason so
  that the exclusion rate is monitorable. A sudden spike in the
  exclusion rate is a signal that either the parser or the engine
  regressed.
- The training data scale is "corpus size minus exclusion count" —
  not "corpus size". Exclusion rate is a published number per parse
  run.
- A parsed Round that the engine cannot replay is excluded entirely;
  it is **not** truncated to the prefix that did replay. Truncated
  Rounds would bias the Decision distribution toward early-Round
  positions.
- Validation throughput is a real constraint at 2.4M games. The
  validate step must be parallelisable (per-game independence holds —
  no cross-game state).

## Rejected alternatives

- **Trust the parser; skip replay.** Rejected — silent label
  corruption is the worst failure mode. The cost of replay is bounded
  and one-time; the cost of debugging a corruption-trained model is
  unbounded.
- **Replay only a sample of games.** Rejected — sampling catches
  high-frequency bugs but misses rare-rule-interaction bugs, which are
  exactly the ones that matter. The corpus is large enough that
  full-replay is operationally feasible.
- **Replay only legality, not score.** Rejected — legality replay
  catches engine refusals but not subtle scoring bugs (e.g. wrong
  Dragon-trick assignment, mis-credited round-end bonuses). Score
  match is the stronger invariant.
