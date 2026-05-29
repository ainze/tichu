# 2026-05-29 — `enumerate_straights` perf spike

## TL;DR

`enumerate_straights` was identified as ~10% of parse_bsw worker CPU
across 7 distinct flamegraph lines (priority #1 from the post-v3
wrap-up). Drilling in:

- The real cost was **`Straight.__post_init__` validation**, not the
  Python-level enumeration loop.
- Pre-canonicalising cards at the enumeration site + bypassing
  `__post_init__` via a new `Straight._from_canonical_cards()` factory
  yields **2.5× – 3.6× speedup** on `enumerate_straights` and
  **1.8× speedup on `_enumerate_all`** for the full-ladder phoenix hands.
- 599 tests pass — no engine, BC, AWR, or inference regressions.

The two conservative micro-hoists tried first (constant tuple
extraction, deduplicated `ranks_to_fill` computation) were within
noise. Confirms the lesson from the BC investigation: **profile, then
fix the cost dominator, not the visible loop**.

## What was measured

`scripts/bench_enumeration.py` runs each enumerator against five
representative hands. The hand mix exercises:

- `ladder_naturals_dup2`: ranks 2-14 in JADE plus a SWORD-2. Triggers
  the "no phoenix" branch with a few inner-product expansions.
- `ladder_with_phoenix`: ranks 2-14 in JADE + PHOENIX. Triggers the
  phoenix-substitution branch on every window.
- `ladder_phoenix_mahjong`: same plus MAHJONG. Adds the Mahjong-led
  starting window.
- `dense_suits_3_to_12`: 40 cards (ranks 3-12 × 4 suits). Upper-bound
  stress test for `product(*rank_choices)`.
- `mid_round_7_cards`: a plausible 7-card hand mid-round. Realistic
  workload distribution.

## Baseline numbers (per call)

| Enumerator | naturals_dup2 | with_phoenix | phoenix+mahjong | mid_round |
| --- | ---: | ---: | ---: | ---: |
| `enumerate_singles` | 6.0 µs | 5.9 µs | 6.3 µs | 3.4 µs |
| `enumerate_pairs` | 4.9 µs | 26.4 µs | 27.1 µs | 5.5 µs |
| `enumerate_triples` | 2.7 µs | 2.4 µs | 3.1 µs | 1.1 µs |
| `enumerate_full_houses` | 6.5 µs | 27.8 µs | 28.5 µs | 6.7 µs |
| **`enumerate_straights`** | **524 µs** | **3,693 µs** | **4,619 µs** | **28 µs** |
| `enumerate_pair_steps` | 8.0 µs | 832 µs | 867 µs | 7.1 µs |
| `enumerate_four_of_a_kind_bombs` | 2.7 µs | 1.9 µs | 2.0 µs | 1.0 µs |
| `enumerate_straight_flush_bombs` | 367 µs | 219 µs | 218 µs | 2.1 µs |
| **`_enumerate_all`** | **950 µs** | **4,819 µs** | **6,309 µs** | **59 µs** |

Phoenix turns `enumerate_straights` from 524 µs into 3,693 µs — a **7×
explosion**. The phoenix branch generates ~7× the number of `Straight`
objects (390 vs 54) and each construction pays full validation.

## What didn't work (hoists)

Attempt 1 was three "safe" micro-hoists:

1. Hoist `extra = (MAHJONG,) if mahjong_in_window else ()` out of the
   inner `for picked` loop (only depends on `mahjong_in_window`).
2. Hoist `extra_with_phoenix = (PHOENIX,) + extra` to per-window.
3. De-duplicate `ranks_to_fill` computation between Case 1 and Case 2.
4. Drop redundant `tuple(picked)` re-wrap (`picked` is already a tuple
   from `product`).

Result: all within run-to-run noise (590 vs 524, 4108 vs 3693, 5929 vs
4619). Each iteration's hoisted work was ≤100 ns × ~400 iterations =
40 µs saved on a 4 ms call — invisible against the dominator.

**Lesson**: Don't bother with Python-loop hoists when the inner-loop
call sits behind a `frozen=True` dataclass `__post_init__`.

## What did work — bypass `__post_init__`

`Straight.__post_init__` does substantial per-construction work:

```python
def __post_init__(self) -> None:
    if len(self.cards) < 5:                         # length check
        raise ValueError(...)
    if len(set(self.cards)) != len(self.cards):     # set alloc, distinctness
        raise ValueError(...)
    for c in self.cards:                            # type loop
        if isinstance(c, SpecialCard) and ...:
            raise ValueError(...)
    has_phoenix = PHOENIX in self.cards             # membership scan
    has_mahjong = MAHJONG in self.cards             # membership scan
    if has_phoenix and self.phoenix_as_rank is None:
        raise ValueError(...)
    # ...
    normal_ranks = [c.rank for c in self.cards if isinstance(c, Card)]  # list comp
    all_ranks = sorted(normal_ranks)                # sort
    if has_phoenix:
        all_ranks = sorted(all_ranks + [self.phoenix_as_rank])  # 2nd sort
    if has_mahjong:
        all_ranks = sorted(all_ranks + [1])         # 3rd sort
    for i in range(1, len(all_ranks)):              # consecutive-check loop
        if all_ranks[i] != all_ranks[i - 1] + 1:
            raise ValueError(...)
    sorted_cards = self._canonical_card_order(all_ranks)  # dict build + tuple
    object.__setattr__(self, "cards", sorted_cards)
```

Roughly 10 µs per construction at this hand-size. For
`ladder_with_phoenix` we construct ~400 Straights → ~4 ms in
`__post_init__` alone.

**`enumerate_straights` already produces all of these invariants
correctly by construction.** Cards come from sorted `ranks_to_fill`,
distinctness is guaranteed by `_group_by_rank`, ranks are consecutive
by the window construction, Mahjong/Phoenix placement is explicit. The
validation is verifying things the caller already established.

## The fix

Added a trusted factory on `Straight`:

```python
@classmethod
def _from_canonical_cards(
    cls,
    cards: tuple,
    phoenix_as_rank: int | None,
) -> "Straight":
    obj = object.__new__(cls)
    object.__setattr__(obj, "cards", cards)
    object.__setattr__(obj, "phoenix_as_rank", phoenix_as_rank)
    return obj
```

Updated `enumerate_straights` to produce cards in canonical order
directly (small refactor — splice PHOENIX at the right slot index)
and call `_from_canonical_cards()` instead of `Straight()`.

## Post-fix numbers

| Hand | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| `ladder_naturals_dup2` | 524 µs | **147 µs** | **3.6×** |
| `ladder_with_phoenix` | 3,693 µs | **1,283 µs** | **2.9×** |
| `ladder_phoenix_mahjong` | 4,619 µs | **1,862 µs** | **2.5×** |
| `mid_round_7_cards` | 28 µs | 18.5 µs | 1.5× |
| `_enumerate_all` (with_phoenix) | 4,819 µs | **2,686 µs** | **1.8×** |

## Translating to wall-clock

Worker CPU breakdown from the parse_bsw flamegraph was:
- `replay_round` (parent): ~75%
- `legal_actions` (nested): ~44%
- `_enumerate_all` (nested): ~21%
- `enumerate_straights` (nested): ~10%

A 1.8× speedup on `_enumerate_all` saves ~9% of worker CPU. Projection
for parse_bsw at the current 12 games/sec: ≈ **13.2 games/sec** (~10%
throughput lift). For BC materialise: same factor, since both
pipelines share `replay_round` / `legal_actions`.

The bench delta is bigger than the wall-clock delta because not every
decision evaluates a 14-card phoenix hand — the `mid_round_7_cards`
benchmark shows a 1.5× speedup, more representative of average hands.

## What did not get done in this spike (follow-ups)

The same pattern is applicable to other enumerator + frozen-dataclass
pairs. Each contributes a smaller share but together meaningfully add
up:

- `StraightFlushBomb._from_canonical_cards`: ~5% of worker CPU.
  Same shape — pre-canonical cards from the enumeration site, skip
  validation.
- `PairStep._from_validated_pairs`: ~3% on phoenix hands (832 µs).
  `__post_init__` sorts pairs and validates the chain.
- `FullHouse._from_validated_triple_pair`: ~1%. Validation is mostly
  the triple/pair rank-difference check; small but free.
- `Triple` / `Pair`: tiny share, probably not worth touching.
- `FourOfAKindBomb`: ~1 µs/call already, leave alone.

Stretch target: same pattern across all 4 dataclasses would deliver
~30-40% of worker CPU back, projecting parse_bsw to ~16-17 games/sec
(from 12). Not pursued here to keep this spike scoped.

Also untried but flagged:
- Dedupe canonical tuples in a `set` BEFORE constructing `Straight`
  instances. For phoenix hands, ~30-50% of generated `Straight`s are
  duplicates (different suit choices over the same rank pattern that
  canonicalise identically). A second-level dedupe before construction
  would compound on top of `_from_canonical_cards`.

## Files touched

- `src/tichu_engine/combinations.py`: added `Straight._from_canonical_cards`.
- `src/tichu_engine/enumeration.py`: rewrote `enumerate_straights` to
  emit canonical-order cards and use the trusted factory. Hoisted
  invariant module-level constants `_MAHJONG_TUPLE`, `_PHOENIX_TUPLE`,
  `_EMPTY_TUPLE`.
- `scripts/bench_enumeration.py`: new microbench (used to characterise
  baselines and validate the fix). Kept for future regression runs.

## Artefacts referenced

- parse_bsw worker flamegraph: `C:\Temp\parse_bsw_worker.svg` (2,988
  samples over 30s on a 10-worker run against the user's archive).
- parse_bsw main flamegraph: `C:\Temp\parse_bsw_main.svg` (120 samples
  over 20s — ~6% on-CPU; mostly blocked on workers).
