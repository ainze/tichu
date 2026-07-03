# ADR-0039: Featurizer v7 — trick point-stakes + current-trick winner

- **Status:** Rejected — pre-check did not clear AND **H2's premise is false** (see "Result" + "Correction"). Not built.
- **Date:** 2026-07-01 (result 2026-07-03; H2-redundancy correction 2026-07-03)

## Correction (2026-07-03): H2 is redundant with the existing `trick_leader`

This ADR's H2 rests on the claim that `trick.leader` is the *opener* and the
current *winner* (`plays[-1].player`) is a distinct, un-featurised player. **That
is wrong for this engine.** `Trick.add_play` sets `leader = player` on every play
([state.py:53](../../src/tichu_engine/state.py)), so `Trick.leader` is always the
last player to play the top Combination — i.e. the current winner itself. There is
no separate opener. Therefore `current_trick_winner` = `(plays[-1].player - self)%4`
is **byte-identical to v6's `trick_leader[4]`** = `(leader - self)%4`, and H2 adds
nothing. Verified empirically on the pre-check cache: of 105,460 partner-winning
contested rows, **all** had partner as leader; the partner-overtook case H2 was
meant to capture **does not occur**. So the only genuinely non-derivable candidate
was H1 (dead in the pre-check), H2 is a duplicate, and H3 is derivable — the whole
v7 is unjustified. The "partner-overtake blunder" this was reopened to fix is a
**cotrain reward / credit-assignment** problem (the policy already sees partner-
winning via `trick_leader`, plus first-out via `out_order` and GT via
`grand_tichu_callers`), routed to reward-shaping / a served-time guard, not a
featurizer change.
- **Related:** [ADR-0038](0038-featurizer-v6-played-by-and-schupfen-received.md) (v6 content lever), [ADR-0019](0019-bit-pack-materialised-bundle.md) (bit-pack bundle / `CONTINUOUS_SECTIONS`), [ADR-0017](0017-featurizer-v4-compact-trick-top-combo.md) (`trick_top_combo` layout — owner-less by design), [ADR-0033](0033-perfect-info-critic-escape.md) (why critic-R² is not a gate).

## Context

An audit for EV-relevant signal *not derivable* from the v6 featurizer (591 dims) surfaced two additions the net is structurally blind to:

1. **Trick point-stakes.** The engine tracks the point value sitting in the current Trick pile (5→5, 10→10, K→10, Dragon→+25, Phoenix→−25; `engine._card_value`), but nothing in the feature path exposes it. `trick_top_combo` carries only the leading Combination's shape/rank; `played_by[4][56]` is cumulative-this-Round, so current-pile points cannot be separated from prior-Trick cards. **Non-derivable.**
2. **Current-Trick winner.** The tactically load-bearing question mid-Trick is *"who currently holds the top combo — the player I overtake if I beat it?"* = `trick.plays[-1].player`. This is **not** `trick.leader` (the *opener*, already at v6 as `trick_leader[4]`): in any Trick with a raise the opener and the current top-holder differ. The `trick_top_combo` section encodes the top Combination but **never its owner** — so partner-vs-opponent-winning is absent from v6.

A third candidate — remaining-cards-by-rank (card-counting summary) — was **dropped**: it is the complement of `own_hand[56]` + `played_by[4][56]`, both already at v6, so it is derivable and fails the "must reduce label noise the net can't already see" filter.

## Decision

**Bump `FEATURIZER_VERSION` to `"v7"`. Two additions, 591 → 596.**

1. **`trick_point_value` — 1 continuous scalar.** Sum of `_card_value` over `pub.trick.plays`, normalised **÷ 25** (Dragon = 1.0), **not clipped**. Range is `[−25, +125]` (floor: lone Phoenix; ceiling: all four 5s + four 10s + four Kings + Dragon in one pile with the Phoenix absent — the whole-deck 100 nets the Phoenix's −25 back in). The divisor only bounds the ratio; clipping would flatten exactly the high-stakes piles the feature exists to flag. Added to `CONTINUOUS_SECTIONS`.
2. **`current_trick_winner[4]` — relative-seat one-hot** `[self, next, partner, previous]` on `trick.plays[-1].player`, all-zero on an empty Trick. Indicator (bit-packable). The winner one-hot (not a lean 1-bit "partner-winning") also carries which *opponent* leads — dragon-give direction and whose points these become — and is symmetric with `trick_leader` / `played_by`.

The two are fed as **separate dims**; the H1×H2 interaction is left for the 1024×4 trunk to learn (a precomputed product would discard the components).

**No engine *state/codec* change, no Replay-Validation impact.** The only engine touch is an API-surface addition: the existing private `_trick_points` is promoted to a public `trick_point_value(trick)` so the scoring mapping has a **single source of truth** (reused, not duplicated in training — the drift class behind the team_scores/schupfen bugs). `trick.plays` and `trick.leader` already exist on `PublicState` and already round-trip through the Wire PrivateState (`codec.trick_to_json`), so `featurize` computes both H1 and H2 identically at train time and at `/act` — structurally immune to the train/infer mismatch class (team_scores, schupfen `current_player`) that has bitten this featurizer before.

## Consequences

- **Cheap rebuild, not a full re-materialise.** `trick_point_value` is *continuous* → a float column, so v7 is built by an **additive-augment pass** that appends the 1 new column to the existing v6 bit-packed bundle (the `current_trick_winner` indicators bit-pack in place). The 591 v6 columns are **not** rewritten — this is far cheaper than ADR-0038's full v6 stack rebuild.
- **The version bump and BC retrain are the irreducible floor.** Output width 591 → 596, so every v6 checkpoint is non-loadable by the pin. BC is **retrained**, warm-started from the finished v6 wishfix BC with a widened (591→596) input layer, not trained from scratch. `SECTION_DIMS` + `CONTINUOUS_SECTIONS` gain the entries; the bundle reader reconstructs 596; a frozen-golden test is cloned for v7.
- **`perfect_info.py`'s `PERFECT_INFO_DIM`** tracks the +5 observable-block growth.

## Gating (pre-committed before the number is seen)

1. **Residual corpus pre-check (before any version work).** On the **decile-9 contested-trick slice** (acting seat can legally beat the current top, points at stake), fit a predictor of the human action from v6 features alone, then from v6 + {`trick_point_value`, `current_trick_winner`}. Bar: **material held-out NLL reduction on that slice + no corpus-wide regression.** Run on-the-fly (replay sample, no bundle written). H1-alone is reported as a diagnostic, never as the go/no-go — the signal is expected to live in the H1×H2 interaction. If flat, v7 is never built.
2. **Gate A (post-retrain).** Held-out BC NLL / top-k on the decile-9 contested slice vs the v6 baseline.
3. **Gate C (truth).** EV via the paired-world / blunder-miner replay (the only loop that beats the PPO noise floor), or a greedy mini-tournament.

**No critic-R² gate.** H3 is dropped, so the only remaining channel is the BC prior (better prior → free cotrain lift). A held-out critic-R² delta is the instrument [ADR-0033](0033-perfect-info-critic-escape.md) found structurally confounded (offline held-out R² is the wrong lens for an on-distribution online critic); any critic read stays telemetry, never a gate.

## Result (2026-07-03) — pre-check did not clear, v7 NOT built

The residual pre-check (`scripts/precheck_trick_stakes.py`) ran on 20k games /
**426,804** decile-9 contested-trick rows (grab-vs-cede label, by-game split,
seed-averaged MLP). H3 was tested too (reopened despite being derivable).

**Verdict: STOP.** With a properly regularised probe (early stopping on a by-game
val split), every candidate lands **below the pre-registered 1% rel-NLL bar** at
every capacity, including the trunk's own 1024 width:

| hidden | v6+H1 (scalar) | v6+H1H2 | v6+H3 | v6+H1H2+H3 |
|---|---|---|---|---|
| 256 | ~0% | **+0.58%** | +0.34% | +0.69% |
| 1024 | ~0% | **+0.42%** | +0.40% | +0.67% |

(8 seeds, NLL std ≈ 0.0015.) **H1 (the point scalar) is inert** — all stakes signal
is H2. The effect is real but small.

**Methodology caveat that flipped the call:** a *first* run without early stopping
gave v6+H1H2 = **+1.16%** at hidden=256 and read as a GREENLIGHT. That was an
**overfitting artifact** — held-out baseline AUC fell with probe width
(0.849→0.840→0.826) and the verdict swung GREENLIGHT/STOP with the probe's hidden
size (H1H2 = +0.38 / +1.11 / −0.56% across widths). Early stopping flattened the
baseline (0.848→0.851) and collapsed all lifts below the bar. **A residual-probe
verdict is untrustworthy without early stopping + a capacity sweep.** The 1% bar and
the derivability filter both held: neither H1H2 (non-derivable) nor H3 (derivable)
earns the version bump on the binary grab/cede proxy.

The pure feature primitives (`trick_stakes.py`, `card_counting.py`) and their tests
remain in-tree, unused by `featurize` (v6 stays pinned), for whoever revisits this.
The one un-run, more-faithful proxy is the full 1809-way play head (not binary
grab/cede) — but it must clear the same **early-stopped** bar before any v7 build.

## Rejected alternatives

- **Ride `trick_leader` for H2.** Rejected: `trick.leader` is the opener, not the current top-holder; they differ after any raise, and the winner's identity is genuinely absent from v6.
- **H2 as a lean 1-bit "partner-is-winning".** Rejected: the 4-dim one-hot additionally encodes which opponent leads (dragon-give / point-destination signal) at no float cost.
- **Precompute H1×H2 as a single scalar.** Rejected: throws away the componentwise signal a deep trunk uses.
- **H3 remaining-cards-by-rank.** Rejected: derivable from `own_hand` + `played_by`; the one argument for it ("explicit-though-derivable eases the critic") is untestable on the cheap pre-check and rides the discredited critic-R² instrument.
- **Full v7 re-materialise (rewrite all 591 dims).** Rejected: unnecessary — the only new stored column is continuous and appends additively.
