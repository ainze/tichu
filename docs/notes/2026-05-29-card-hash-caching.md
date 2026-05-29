# 2026-05-29 — Card / SpecialCard `__hash__` caching

## TL;DR

py-spy on a parse_bsw worker attributed ~14% of worker CPU to
auto-generated dataclass `__hash__` paths (the `<string>:23/24`
frames in the flamegraph). A microbench identified `Card` as the
leverage point: every combination class hashes a tuple of cards
recursively, so every Pair hash is ~2×Card hash, every Triple is 3×,
etc.

Two-step change:

1. Override `Card.__hash__` and `SpecialCard.__hash__` with **perfect
   hashes** that bypass tuple construction.
2. **Cache the hash on the Card instance** via `__post_init__` so
   repeated hashes are pure attribute reads.

Combined effect:

| Type | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| `Card.__hash__` | 143 ns | **49 ns** | **2.9×** |
| `Pair.__hash__` | 319 ns | **149 ns** | **2.1×** |
| `Triple.__hash__` | 432 ns | **179 ns** | **2.4×** |
| `FourOfAKindBomb.__hash__` | 548 ns | **222 ns** | **2.5×** |
| `enumerate_straights` (ladder+phoenix) | 1,283 µs *post-spike-1* | **790 µs** | 1.6× more |
| `_enumerate_all` (ladder+phoenix) | 2,686 µs *post-spike-1* | **1,897 µs** | 1.4× more |

Stacked with the earlier `Straight._from_canonical_cards` change,
total `_enumerate_all` speedup is **2.5×** vs original (4,819 µs →
1,897 µs).

## Microbench baselines

```
type                             ns/hash
----------------------------------------
Card                              143.4    ← was the dominator
SpecialCard (DRAGON)               84.3
Single(card)                      208.1
Pair                              319.0    ← 2×Card + tuple
Triple                            432.0    ← 3×Card + tuple
Straight (5-card)                  92.4    ← tuple-of-cards optimised
FourOfAKindBomb                   547.9    ← 4×Card + tuple
StraightFlushBomb (5)              86.2
Suit.JADE                          73.6
(suit, rank) tuple                 27.9    ← native tuple hash floor
```

Pair = 2×Card + tuple overhead (314 ≈ 319 ✓).
Triple = 3×Card + tuple (429 ≈ 432 ✓).
FourOfAKindBomb = 4×Card + tuple (572 ≈ 548 ✓ — slight slack from
inlining).

**Card hash compounds.** Cutting it in half cuts every combination
class hash in proportion.

## Why the auto-generated dataclass `__hash__` is slow

`@dataclass(frozen=True)` emits:

```python
def __hash__(self):
    return hash((self.suit, self.rank))
```

Each call does:
1. Attribute access on `self.suit` and `self.rank`.
2. Tuple allocation of `(suit, rank)`.
3. `hash(tuple)` → recursive: `hash(Suit.JADE)` (Enum → hash of
   `_name_` string) + `hash(rank)` (native int identity).
4. Tuple hash combine.

The tuple build + Enum recursion dominates. Native tuple hash for a
2-tuple of cached-hash elements is ~28 ns, but the full path for a
Card is ~143 ns.

## What was done

### Step 1: perfect hashes (one method override per class)

```python
_SUIT_HASH_BASE: dict[Suit, int] = {
    Suit.JADE: 0, Suit.SWORD: 16, Suit.PAGODA: 32, Suit.STAR: 48,
}

def __hash__(self) -> int:
    return _SUIT_HASH_BASE[self.suit] | self.rank
```

Card hash: 143 → 111 ns. Pair: 319 → 287. Modest — the Python method
dispatch + attribute access floor is ~80 ns; the actual hash work was
the cheap part.

### Step 2: precompute + cache on instance

```python
def __post_init__(self) -> None:
    object.__setattr__(
        self, "_hash", _SUIT_HASH_BASE[self.suit] | self.rank,
    )

def __hash__(self) -> int:
    return self._hash
```

Card hash drops to 49 ns — pure attribute read. The compute happens
once at construction; every subsequent hash is amortised. Workload
must hash a Card more than once for the cache to pay off, and Cards
are heavily reused (frozenset(hand) operations, tuple-of-cards
recursion inside Pair/Triple/etc).

`object.__setattr__` is the standard escape hatch for setting an
attribute on a `frozen=True` dataclass; it bypasses the frozen guard
that the dataclass-generated `__setattr__` raises from. Pickle
roundtrip recomputes `_hash` via `__post_init__` on the new instance
— stable across processes.

### SpecialCard

Singleton-by-design (DRAGON / PHOENIX / MAHJONG / DOG; `__reduce__`
maps unpickling back to the singletons). Perfect hash by name:

```python
_SPECIAL_HASH_BY_NAME = {
    "dragon": 200, "phoenix": 201, "mahjong": 202, "dog": 203,
}

def __hash__(self) -> int:
    return _SPECIAL_HASH_BY_NAME.get(self.name, hash(self.name))
```

Bench moved 84 → 78 ns. SpecialCard sees fewer hashes than Card
(only 4 singletons; mostly reused via `is`-comparison rather than
hashed lookups), so the change is small but free.

## Wall-clock projection

Translating to parse_bsw / BC materialise: the flamegraph attributed
~14% to `__hash__`. With combined Card+combinations 2-3× faster, we
save ~7-9% of worker CPU. Stacked on the earlier `enumerate_straights`
spike (~9% saved), parse_bsw is projected at **~15-17% total wall-clock
lift**, or 12 → ~14 games/sec.

Realistic mid-round 7-card hand sees a smaller speedup
(`_enumerate_all`: 59 → 55 µs, ~7%). Full-ladder hands see big wins
but they're a minority of replay calls. Real wall-clock will land
between the two — bench follow-up against actual parse_bsw is the
honest validation.

## What was NOT done (follow-ups)

- **Cached hash on combination classes**. Pair / Triple / FullHouse /
  PairStep / FourOfAKindBomb / StraightFlushBomb all have
  auto-generated `__hash__` that hashes a tuple of cards. With Card
  already cached, the residual is ~50 ns × N_cards. A
  `__post_init__`-cached hash on each combination class would drop
  Pair from 149 → ~30 ns (5× more). Same pattern, ~10 LOC per class,
  zero ML-quality risk. **Worth doing as a stretch follow-up.**
- **PublicState / PrivateState / Trick hashing**. These are larger
  objects but hashed less often. py-spy didn't surface them as hot.
  Skip unless a future profile says otherwise.
- **`slots=True` on dataclasses**. Smaller memory + faster attribute
  access. Untouched here because the caching change already gets the
  hot path; slots is a separate refactor with broader implications
  (no `__dict__`, etc).
- **IntEnum for Suit**. Would let us skip the `_SUIT_HASH_BASE` dict
  lookup entirely (just `self.suit.value << 4 | self.rank`). But
  changes Suit's repr/serialisation contract — bigger blast radius
  for a marginal speedup.

## Files touched

- `src/tichu_engine/cards.py`:
  - `Card`: added `__hash__` override + `__post_init__` cached `_hash`.
  - `SpecialCard`: added perfect-hash `__hash__`.
  - Module constants `_SUIT_HASH_BASE` and `_SPECIAL_HASH_BY_NAME`.
- `scripts/bench_enumeration.py`: new `--hash` mode for the hash
  microbench (kept for regression).
- `docs/notes/2026-05-29-card-hash-caching.md`: this file.

## Test status

- Engine: 246 / 246 pass.
- Full suite (excluding pre-existing AWR streaming flake): 599 pass,
  1 skip. No regressions.
