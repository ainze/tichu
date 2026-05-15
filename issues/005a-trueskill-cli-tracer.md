---
id: "005a"
title: "TrueSkill tracer: minimal compute_trueskill CLI"
type: AFK
blocked_by: ["004"]
parent: "005"
---

## Parent

[#005 TrueSkill rating computation](005-trueskill-ratings.md)

## What to build

The smallest end-to-end path: a `compute_trueskill` CLI that reads BSW games in chronological order, runs a per-game TrueSkill team update for the four seats, and writes one row per player to a Parquet table.

**Input.** `--input` accepts either a directory of `.tch` files or a `.tar.zst` archive containing them. The CLI sniffs the suffix and streams accordingly. Iteration order is sorted ascending by integer game_id (the filename stem). Games whose engine validation already flagged them as bad (per #004's `known_bad_games.txt`, if present alongside the input) can be passed through unfiltered for now — filtering can come later.

**Per-game update.** For each parsed game:

- Determine the winning team from `sum(round.ergebnis[0]) vs sum(round.ergebnis[1])` across all rounds in the game (BSW games run to 1000).
- Apply a single TrueSkill rating update treating seats 0+2 as team A and seats 1+3 as team B; draws skipped.
- Use the `trueskill` PyPI package with default parameters.

**Output.** `--output ratings.parquet` with columns:

- `player_handle: string`
- `mu: float64`
- `sigma: float64`
- `n_games: int32`

Players with zero games are not written. New players seen for the first time start at TrueSkill defaults (μ=25, σ=25/3).

**Dependency.** Add `trueskill>=0.4` to `pyproject.toml` dependencies.

## Acceptance criteria

- [ ] `compute_trueskill --input <dir|.tar.zst> --output <file.parquet>` exits 0 on the checked-in `sample/` directory
- [ ] Output Parquet has the four columns listed above with the correct types
- [ ] Each player_handle in the sample corpus has `n_games >= 1` and a `mu` distinct from the TrueSkill default (i.e. an update was actually applied)
- [ ] CLI also accepts a `.tar.zst` archive of `.tch` files and produces the same output as the equivalent extracted directory
- [ ] Games are processed in ascending game_id order (verified by feeding a deliberately out-of-order shuffle: result should match the sorted run)

## Blocked by

- [#004 BSW parser + replay validation](004-bsw-parser-replay-validation.md)
