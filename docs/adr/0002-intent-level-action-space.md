# ADR-0002: Intent-level action space with heuristic resolver

- **Status:** Accepted
- **Date:** 2026-05-25
- **Supersedes:** —
- **Related:** [ADR-0001](0001-trunk-architecture.md)

## Context

The policy network must output an action distribution over every legal play.
Tichu's full set of concrete plays — distinct combinations of specific cards
including suit identities and Phoenix substitution positions — is on the order
of ~10,000 entries per Decision. Most of those are exchangeable from a
strategy standpoint: choosing "the red 7 + the black 7" vs. "the green 7 + the
blue 7" for a pair-of-7s does not change expected value in any meaningful way.

Two candidates were considered:

1. **Concrete action space (~10k entries).** One slot per distinct suit-aware
   play. The policy directly emits a card-specific action.
2. **Intent-level action space (1,809 entries).** One slot per *intent* —
   "pair of 7s with Phoenix" or "straight 5..9 with Phoenix at position 2".
   A separate Resolver picks concrete suits at inference time using a
   deterministic heuristic.

## Decision

**Use an intent-level Action Space of 1,809 entries, with a heuristic
Resolver that maps Intent + Hand → Concrete Action.**

The Action Space is enumerated once in `tichu_training/action_space.py` in a
fixed deterministic order, persisted via `ACTION_SPACE_VERSION = "v1"`. Every
Checkpoint stamps this version; loading a Checkpoint against a different
Action Space is a hard error.

The Resolver lives in `tichu_inference/ml_agent.py`. Given an Intent and the
player's Hand, it picks specific cards (which two 7s for a pair-of-7s; which
suit holds a substituted Phoenix). Picks are heuristic.

## Rationale

1. **5× smaller policy head.** 1,809 outputs vs ~10,000. Less parameters,
   less softmax cost, less label noise in the BSW training signal (the
   corpus does not consistently identify which-suit-of-a-pair was played).
2. **Suit identity is strategically irrelevant in Tichu.** Tichu has no
   trump suit and no suit-based scoring. Suits exist only for Straight
   Flush bombs, which the Action Space handles explicitly. Folding
   suit-equivalence into the Action Space removes a degree of freedom the
   model would otherwise have to relearn.
3. **Phoenix substitution is enumerated explicitly.** Phoenix-in-position-k
   variants are distinct Intents so that the model can express a preference
   for *where* to substitute the Phoenix in a straight or pair-step. The
   Resolver does not re-choose this.
4. **Cheaper masking.** The legal-actions mask is computed over 1,809 slots
   per decision step, not 10,000.

## Consequences

- The Resolver heuristic becomes a real design surface. Two different
  heuristics over the same Intent distribution can produce systematically
  different concrete plays. Two known choice points exist today:
  (a) which specific cards realise a multi-card Intent (e.g. *which* two 7s),
  and (b) which suit a substituted Phoenix "becomes" for the purpose of
  trick-taking. If a Resolver change ever measurably moves eval results,
  it deserves its own ADR.
- The Action Space is hard to change. Any reordering or membership change
  forces an `ACTION_SPACE_VERSION` bump, which invalidates every existing
  Checkpoint. The Action Space is treated as immutable for the lifetime of
  v1.
- The BSW Parser must canonicalise concrete plays from the corpus into
  Intents during the parse step. That mapping (concrete → intent) is the
  inverse of the Resolver and must be exact — there is no heuristic on the
  training side.
- The Belief model and any future search-based policy in Phase 2 must
  agree on the same Intent vocabulary, or pay a translation cost.

## Rejected alternative

**Concrete action space (~10k entries).** Considered and rejected. Higher
parameter count, label noise from the BSW corpus's inconsistent suit
recording, and no strategic benefit. Adopt only if a future Decision becomes
suit-dependent — none currently is.
