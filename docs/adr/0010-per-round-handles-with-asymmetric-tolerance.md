# ADR-0010: Per-round handles, with asymmetric tolerance across BC and TrueSkill

- **Status:** Accepted
- **Date:** 2026-05-26
- **Related:** [ADR-0008](0008-bsw-replay-validation.md), [ADR-0009](0009-bsw-ingest-streaming-pipeline.md)

## Context

Two real BSW data shapes break the simplifying assumption that a Game has
four fixed player handles:

1. **Anonymous Seats.** Some pre-deal / start-hand lines drop the handle
   entirely (e.g. `(1) BA BD SB B10 R9 S7 S6 R3` — a seat index but no
   name). This happens at the moment a guest / freshly-joined player has
   not yet been assigned a handle by BSW, and also in occasional
   serialiser quirks where a single line drops the handle even though
   the player is named elsewhere in the round (e.g. `Drache an: (1)`).
2. **Mid-game substitutions.** The same seat can be `alejandro_styl`
   for rounds 0-3, anonymous in round 4 (the substitution boundary),
   and `pöppi69` for rounds 5+ of the same Game. The seat is stable;
   the identity is not.

Hand-traced from a parse run, ~15 of the first 16 games in the corpus
hit one of these two cases. The legacy parser raised
`ValueError: expected '(N)handle <cards>' line` and the legacy
`ParsedGame.handles` snapshot was set from round 0 only — used
downstream by `to_parquet`, the TrueSkill sweep, and the Tichu
success-rate counter as if it were the stable identity for all rounds.

That snapshot was silently mis-attributing post-substitution decisions
to the round-0 handle. Replay validation ([ADR-0008](0008-bsw-replay-validation.md))
did not catch this because score-agreement is unaffected by who's-credited.
The rejection of the *whole game* on the parse error masked the
deeper mis-attribution bug — fixing only the parser would have made
the mis-attribution silent at corpus scale.

## Decision

**Player identity is per-round, not per-game.**

1. `ParsedRound` carries `handles: tuple[str, str, str, str]`, populated
   from the round's own pre-deal / start-hand lines. An Anonymous Seat
   is recorded as the empty string `""`.
2. `ParsedGame` no longer has a `handles` field. Any consumer that
   wants identity must read it from a specific round.
3. The parser tolerates an empty handle in every `(N)…`-shaped line
   (pre-deal, start-hand, plays, passes, schupfen, Tichu / Grand-Tichu
   calls, `Drache an:`). The seat index is the canonical identity; the
   handle is metadata.

**Tolerance is asymmetric across consumers:**

- **BC ingestion is permissive.** `to_parquet` reads `parsed_round.handles[player]`,
  so substitutions and anonymous seats flow through to the parquet
  `player_handle` column. The skill-decile join on an empty handle
  returns `None`, which downstream training treats as Neutral Skill
  Decile. No training data is lost.
- **TrueSkill ingestion is strict.** `compute_ratings` skips any Game
  where (a) any round contains an Anonymous Seat, or (b) any seat's
  handle differs between rounds. The per-identified-stable-game rating
  semantics are preserved; the synthetic `""` "player" never enters
  the rating table.
- **Tichu success-rate counters** follow the same per-round attribution
  as `to_parquet`: a call is credited to the seat's *round-level*
  handle, and Anonymous Seats are skipped (no aggregation against `""`).

## Rationale

1. **Per-round identity is the only correct model.** The same seat
   playing two distinct identities in one game is real BSW data, not
   noise. Storing one handle per game is a category error.
2. **`""` is a load-bearing sentinel.** It already maps to
   `skill_lookup.get("") → None → Neutral Skill Decile`, which is the
   pre-existing concept for "identity unknown". Inventing a new sentinel
   would have created a distinction no consumer cared about.
3. **Asymmetric tolerance matches what each consumer needs.** BC wants
   signal — the cards played, the trick outcomes, the team scores —
   which an Anonymous Seat does not corrupt. TrueSkill wants identity
   — which an Anonymous Seat does corrupt. Same anomaly, two different
   correct answers.
4. **Removing `ParsedGame.handles` is the discipline.** Keeping it as
   a round-0 snapshot would have re-created the trap: every consumer
   would need to remember "use round handles, not game handles" — and
   the call sites that forgot would have looked correct.
5. **Defensive parser relaxation costs nothing.** Of the six handle-
   capturing regexes, only one (`_PLAYER_LINE_RE` in pre-deal context)
   uses the captured handle for anything. The other five throw it
   away; relaxing `\S+` to `\S*` is a one-character change per regex
   with no semantic shift.

## Consequences

- `ParsedGame.handles` is gone. Any external caller breaks loudly at
  attribute-access time — preferred over silent mis-attribution.
- Parquet shards now contain the correct per-round handle for
  substituted seats. Existing shards parsed under the legacy code
  carry the round-0 handle for *every* row of a substituted game and
  should be regenerated.
- TrueSkill exclusion rate gains a new source: any game with an
  Anonymous Seat or seat-substitution drops out. Worth monitoring — a
  spike means either the parser regressed or BSW changed its
  serialisation.
- The `""` handle never appears as a TrueSkill key. The Min-games
  Filter no longer has to filter a synthetic `""` "player" with
  pathological `n_games`.
- The Tichu Success Rate computation now matches the granularity of
  the data — a call credited to whichever player held the seat that
  round, not the round-0 holder.

## Rejected alternatives

- **Skip every game with an Anonymous Seat or substitution.** Rejected
  for BC ingestion — anonymous and substituted games still contain
  valid cards-played / trick-outcome signal, and Neutral Skill Decile
  already exists as the right fallback. Accepted for TrueSkill ingestion
  because there identity *is* the signal.
- **Per-round TrueSkill updates instead of per-game.** Considered as a
  way to salvage substituted games — apply a 2v2 TrueSkill event per
  round with that round's seating. Rejected because it changes rating
  semantics (`n_games` becomes `n_rounds`; convergence dynamics
  differ), and the CONTEXT.md `Player Rating` glossary entry would
  need re-defining. Migrate later if substitution frequency turns out
  to be high enough to matter.
- **Keep `ParsedGame.handles` as a round-0 snapshot for back-compat.**
  Rejected — re-creates the silent-mis-attribution trap. Loud breakage
  at attribute access is strictly better than quiet wrong data.
- **Introduce a distinct sentinel like `"<anon>"` instead of `""`.**
  Rejected — the two meanings (anonymous player vs. unattributed
  action) collapse to identical downstream behavior (Neutral Skill
  Decile). The distinction is monitoring-only and can be derived from
  `parsed_action.player == -1` for the unattributed case without
  baking a sentinel into the handle column.
- **Per-line handle recovery in the parser** (back-fill the missing
  handle from `parsed_round.handles[seat]` into each `ParsedAction`).
  Rejected — per-round handles are looked up at `to_parquet` time
  anyway, so the back-fill is redundant.
