---
id: "006a"
title: "Canonical intent-level action space"
type: AFK
blocked_by: ["002"]
parent: "006"
---

## Parent

[#006 Featurizer + action space](006-featurizer-action-space.md)

## What to build

A single ordered list of every action the agent can take in Tichu, enumerated once at import time, with a frozen string version constant and a round-trippable encode/decode pair.

**Granularity: intent-level.** One entry per (combination kind, defining rank(s), phoenix-substitution slot, length). Concrete suit-choice for the cards in a pair/triple/etc. is *not* part of the action — the action wrapper resolves that heuristically at submission time.

**Sections (deterministic order, never changes within a version).** The canonical list is the concatenation:

1. Play singles — every distinct single-card play: 52 natural + Mahjong + Dog + Dragon + Phoenix-as-base + 13 entries for Phoenix-following-rank-N (rank 2..A).
2. Play pairs — for each rank 2..A: natural pair, phoenix-substitute pair.
3. Play triples — for each rank 2..A: natural triple, phoenix-substitute triple.
4. Play full houses — for each (triple-rank, pair-rank) with triple-rank ≠ pair-rank: natural, phoenix-in-triple, phoenix-in-pair.
5. Play pair-steps — for each (start-rank, length) with start ∈ 2..A and length ∈ 2..(A-start+1): natural, plus one phoenix variant per pair slot.
6. Play straights — for each (start-rank, length) with start ∈ 1..A−4 and length ∈ 5..14 within bounds: natural, plus one phoenix-position variant per position (excluding the rank-1 slot when Mahjong is mandatory).
7. Play 4-of-a-kind bombs — one per rank 2..A.
8. Play straight-flush bombs — for each (suit, start-rank, length) with length ≥ 5.
9. Pass action — single entry.
10. Tichu call — single entry.
11. Grand Tichu call — single entry.
12. Schupfen — three entries per direction: "give-to-next", "give-to-partner", "give-to-previous". Concrete card is chosen heuristically by the wrapper; the policy emits a *direction-intent* per schupfen step. (Three steps per round → three independent direction picks.)
13. Wish rank — 13 entries (rank 2..A) plus one "no wish" entry.
14. Dragon-give direction — two entries: "give-to-left" / "give-to-right".

**Version constant.** `ACTION_SPACE_VERSION: str = "v1"`. Importable from `tichu_training.action_space`.

**API.** Module `tichu_training.action_space` exports:

- `ACTION_SPACE_VERSION: str`
- `CANONICAL_ACTIONS: tuple[Action, ...]` — ordered.
- `ACTION_SPACE_SIZE: int`
- `encode(action: Action) -> int`
- `decode(index: int) -> Action`
- `Action` is a sum-type (dataclasses) covering every section above.

**Import-time log.** When `tichu_training.action_space` is imported, log `INFO` line: `action space v1: N entries (singles=A, pairs=B, ...)`.

## Acceptance criteria

- [ ] `CANONICAL_ACTIONS` is non-empty and `ACTION_SPACE_SIZE == len(CANONICAL_ACTIONS)`
- [ ] `decode(encode(a)) == a` for every `a in CANONICAL_ACTIONS`
- [ ] `encode(decode(i)) == i` for every `i in range(ACTION_SPACE_SIZE)`
- [ ] Section sizes and total are logged at import time
- [ ] `ACTION_SPACE_VERSION == "v1"` and is a frozen module-level constant
- [ ] All non-play actions (pass, tichu, grand_tichu, schupfen×3, wish×14, dragon×2) are present and uniquely indexed

## Blocked by

- [#002 Rules engine](002-rules-engine.md)
