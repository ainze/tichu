# 2026-06-07 — `legal_actions` speedup: memoise the hand, don't operate on the engine

## TL;DR

A prior session flagged `legal_actions` as ~30% of a co-train iteration (the
config comment puts rules-engine legal-action enumeration at *~60% of an iter*)
and proposed **invasive engine surgery**: faster card hashing, memoised
straight/pair enumeration. Measurement overturned that plan.

**~50% of all `_enumerate_all` calls are immediate redundant recomputation, and
~69% are eliminable by a cache** — because the hot path enumerates the *same
hand twice per Play Decision* and re-enumerates unchanged hands on every Pass.
The surgery only attacks the *unique 31%*. The cache attacks the redundant 69%.

The fix is a 1-line `@functools.lru_cache(maxsize=4096)` on `_enumerate_all`.
Measured **1.79× on the engine-bound self-play proxy, byte-identical output**.

## The redundancy

Every Play Decision enumerates the actor's hand twice:

1. `BatchedPolicy.act_play_batch` / BC `legal_mask` →
   `legal_actions_for(private_state)` → `legal_actions` → `_enumerate_all(hand)`
   — to build the legal-Intent mask.
2. `engine.step` → `action not in legal_actions(state)` → `_enumerate_all(hand)`
   **again on the same hand** — to validate the action the policy already
   sampled from the legal set.

On a Pass, the player's hand is unchanged, so the next time they are asked the
enumeration is byte-identical yet recomputed from scratch.

## What was measured

Loop: RuleAgent self-play via `play_full_round` over a seeded
`generate_full_position_pool`. This reproduces the exact `policy-enumerate →
step-enumerate` structure of the co-train rollout without needing torch.
`_enumerate_all` was wrapped to record every `hand` argument.

40 rounds, 6,618 `_enumerate_all` calls (165/round):

| Cache strategy | hit rate |
| --- | ---: |
| 1-slot memo (immediate repeat) | **50.0%** |
| LRU(64) | 67.8% |
| LRU(256) | 68.2% |
| LRU(4096) | **69.3%** |
| distinct hands (cache ceiling) | 30.9% unique → 69.1% reusable |

The clean 50.0% immediate-repeat is the policy→step double-enumeration; the extra
~19% up to the ceiling is Passes and cross-decision repeats.

### Timing (60 rounds, engine-only proxy)

| | wall | |
| --- | ---: | --- |
| baseline | 2.100 s | |
| `lru_cache(maxsize=4096)` | 1.176 s | **1.79×**, 69.3% hits |
| results identical | ✓ | `res_base == res_memo` |

## The fix

```python
# src/tichu_engine/legality.py
@functools.lru_cache(maxsize=4096)
def _enumerate_all(hand) -> frozenset[Combination]:
    ...  # body unchanged
```

## Why it's safe

- **Pure function of `hand`.** `_enumerate_all` calls 8 enumerators that each take
  only `hand`. No other state — the cache never needs invalidation.
- **Key soundness.** `hand` is a `frozenset[CardOrSpecial]`: hashable, value-based
  hash. The policy and `step` pass value-equal but distinct-identity frozensets →
  same entry. The 69% hit rate is itself proof of cross-identity hits.
- **Immutable result.** The returned frozenset is never mutated — every caller
  copies it (`set(all_combos)`, `{c for c in all_combos if …}`). Safe to share.
- **Per-process.** The co-train rollout fans across spawn workers; each has its own
  cache. `maxsize=4096` ≫ the ~51-game working set per worker (M=512 / 10 workers).
- **Determinism.** Returning a stable cached frozenset cannot reorder anything a
  recompute wouldn't; sampling is over the fixed action-space index mask regardless.
  Verified byte-identical.

## Regression test

`tests/engine/test_legality.py`:
- `test_enumerate_all_hits_cache_on_distinct_identity_equal_hands` — pins that a
  value-equal, distinct-identity second call is a real cache *hit* (the hot-path
  pattern) and returns the identical result.
- `test_legal_actions_repeated_calls_are_stable_under_memo` — `legal_actions` is
  byte-identical across repeated and distinct-identity equal states.

`pytest tests/engine tests/training/search` → 323 passed.

## Relationship to prior work

- Complementary to the [`enumerate_straights` spike](2026-05-29-enumerate-straights-spike.md):
  that fix (`Straight._from_canonical_cards`) speeds the cache-**miss** path (the
  unique 31%); the memo eliminates the redundant 69%. They stack.
- The spike's flagged follow-ups (`StraightFlushBomb`/`PairStep`/`FullHouse`
  trusted factories) further speed only the miss path — lower leverage now that the
  redundant majority is cached, but still free wins on the unique tail.

## Not done (lower-leverage, more invasive)

A `validate=False` fast-path on `step` for the trusted rollout would skip the
validation enumeration entirely. The memo already collapses it to an O(1) hash
lookup while keeping the engine's safety contract for BSW replay / tests, so this
is marginal extra gain at higher cost. Not pursued.
