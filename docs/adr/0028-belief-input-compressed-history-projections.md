# ADR-0028: Belief input = policy features + compressed history projections

- **Status:** Accepted (B-core; H4 play-time rejected by ablation) — **mechanism superseded by
  [ADR-0038](0038-featurizer-v6-played-by-and-schupfen-received.md) on 2026-07-27**: featurizer
  v6 absorbed the B-core channels, so the belief-side History block and the A / B_core / B_full
  tier machinery are deleted. See *Implementation status* below. The *decision* (Belief needs
  cross-Trick negative information) stands and is now delivered by the policy featurizer.
- **Date:** 2026-06-03
- **Supersedes (in part):** [ADR-0021](0021-belief-trains-on-a-replay-derived-bundle.md) — the
  "Belief reuses the same 224-dim Feature Vector the policy consumes" clause. **Un-superseded by
  ADR-0038:** at v6 the belief input is again exactly the policy Feature Vector — because that
  vector now *contains* B-core, not because Belief gave up the channels.
- **Ablation outcome:** B-core > A > naive on all metrics (decline channel earns its bytes);
  **B-full < B-core, so H4 (play-time) is dropped**. Absolute belief signal is weak at gate
  scale — see [docs/notes/2026-06-03-belief-history-gate.md](../notes/2026-06-03-belief-history-gate.md).
  Canonical belief input is **B-core (251 dims)**.

## Context

The **Belief Model** ([model.py](../../src/tichu_training/belief/model.py)) predicts each
opponent's per-card occupancy. Today its input is the **identical 224-dim policy Feature
Vector** ([emit.py](../../src/tichu_training/belief/emit.py)). Two problems:

1. **No information edge over the policy.** If Belief sees exactly what the policy sees, its
   only advantage is supervision (ground-truth opponent Hands as labels). It cannot exploit
   any signal the policy's input lacks — so feeding its output back into the policy can only
   re-package information the policy already had.
2. **It misses the dominant belief signal in trick-taking: negative information from
   declines.** A **Pass** on a single-King strongly implies "no free single > King" (modulo
   holding it for a Combination). The featurizer carries `trick_passes` (4) and
   `trick_top_combo` (50) for the **current Trick only** — once a Trick resolves the passes
   are cleared, and all that survives cross-Trick is the position-less `seen_cards`. The
   round-long, **per-opponent record of what each opponent declined** — the richest belief
   evidence — is not in the representation.

The naive fix — materialise the full ordered play history (a one-hot-per-play against the
~1809 **Action Space**, stacked) — is what the v1 featurizer did and what v4/v5 deleted: at
~1.25B play examples it is terabytes of disk. So "complete raw history" is out on cost.

The resolution is that **Belief does not need raw history — it needs a sufficient projection.**
Raw history is hugely redundant (consecutive Decisions in a Round share almost all of it) and
mostly belief-irrelevant. The belief-relevant signal compresses to a handful of engineered
per-opponent statistics that cost ~tens of bytes, not ~1800 dims.

## Decision

**Give the Belief Model its own input = the 224 policy features (unchanged) + a compact,
quantised History block of per-opponent projections,** materialised in the existing belief
bundle pipeline. Belief gets its own input-version stamp (`belief_input_version`) and its own
`feature_dim` / continuous-column set in the bundle manifest — the policy **Featurizer** is
untouched. The History block is accumulated as the emit loop walks `replay.decisions` in
order, indexed by absolute seat, then read out in **relative-seat order** (next / partner /
previous) at feature-build time, matching the label convention.

Ship it as a **three-tier ablation** so each channel must earn its bytes:

| Tier | Input | extra dims | purpose |
|---|---|---|---|
| **A** (floor) | 224 policy features | 0 | current design — supervision-only |
| **B-core** | A + H1 + H2 + H3 | +27 | the negative-information channels |
| **B-full** | B-core + H4 | +56 | + your positive play-order channel |

Run all three at 100k against the naive count baseline. **B-core vs A** proves whether the
decline channel earns its keep; **B-full vs B-core** proves whether play-time does. Only the
winner is materialised at full corpus.

### Feature layout (the spec)

All History values are from the **acting seat's** view; the three opponent slots are ordered
next / partner / previous. Ranks normalised as `rank/14` ∈ [0,1]; quantised to `u8` on disk.

**H1 — `declined_top` (3 opp × 6 Intent-types = 18).**
For each opponent and each non-bomb Combination-Intent type
(Single, Pair, Triple, FullHouse, PairStep, Straight): the **max primary-rank that opponent
declined to beat** — i.e. the highest `trick_top_combo` primary-rank present when that
opponent **Passed** while the top was of that type. `0` = never declined that type. (Bombs
excluded: declining to bomb carries little signal and is rare.)
*Emit:* on a Pass by seat `p`, read `pre_state.public.trick.top_combination`'s type `T` and
primary rank `r`; set `declined_top[p][T] = max(prev, r)`.

**H2 — `lead_summary` (3 opp × 2 = 6).**
Per opponent: (`tricks_led` normalised by a fixed cap, `lowest_single_lead_rank/14`). A low
lowest-single-lead signals shedding / no Combination they wanted to hold.
*Emit:* when seat `p` plays into an empty Trick (becomes leader), bump count and update the
lead-rank min for Single leads.

**H3 — `pass_pressure` (3 opp × 1 = 3).**
Per opponent: fraction of that opponent's Play Decisions so far that were Passes. High =
holding un-leadable cards / running low on beating power.

**H4 — `play_time` (56), Tier-2 only.**
Per card slot: normalised global play-index at which the card left play (`played_idx /
cap`), `0` if unplayed. Positive-ordering channel (your brainstorm). **Player-anonymous** —
attribution lives in H1–H3 — and the **lowest value-per-dim** of the four, which is exactly
why it is isolated in its own tier.

`belief_input = concat(policy_224, H1[18], H2[6], H3[3], [H4[56]])` → **251 (B-core)** or
**307 (B-full)** dims.

### Disk

Per-example bytes (history quantised to `u8`): A ≈ 71 B, B-core ≈ 98 B, B-full ≈ 154 B. At
~1.25B play examples: **A ≈ 89 GB, B-core ≈ 122 GB, B-full ≈ 192 GB** — all in the same order
as the existing BC bundle, none near the terabyte raw-history figure.

## Rationale

1. **Sufficient projection, not raw history.** The decline inference's evidence is "what top
   did each opponent decline," whose compressed statistic is ~18 numbers, not 1800. We pay for
   the signal, not the redundancy.
2. **It gives Belief a genuine information edge** the policy lacks (cross-Trick declines),
   which is the only way an explicit Belief Model beats the policy's *implicit* belief on more
   than supervision alone.
3. **Stays on the fast pre-featurised path** (ADR-0014/0019) — no return to replay-on-the-fly
   throughput costs, because the projections are cheap to store.
4. **Learned, not rule-coded.** The "...unless they're holding it for a Combination" confound
   is handled statistically by training on ground-truth Hands; we only supply the evidence.
5. **Ablation discipline** — each channel proves its value before it ships at scale; cheap to
   run at 100k.

## Consequences

- Belief diverges from the policy Feature Vector → a new `belief_input_version` stamp and a
  belief-specific `feature_dim` + continuous-column list in the bundle manifest;
  `belief_materialised.py` must stop hardcoding the policy `FEATURIZER_OUTPUT_DIM`.
- `belief_examples_for_round` grows a per-Round, per-seat `HistoryAccumulator` threaded across
  the ordered decision walk.
- `BeliefModel(feature_dim=...)` takes the wider input; no architecture change.
- A new belief bundle must be materialised (none exists on disk today); start at 100k.
- The CONTEXT.md Belief Model entry and ADR-0021's "same 224-dim vector" line are updated.

## Implementation status (2026-07-27)

Everything above shipped at v5 and produced the ablation result in the header. **Featurizer v6
([ADR-0038](0038-featurizer-v6-played-by-and-schupfen-received.md)) then folded H1/H2/H3 into
the policy Feature Vector itself** as the `declined_top` / `lead_summary` / `pass_pressure`
sections, fed by live per-seat engine accumulators — so from v6 on:

- **The belief-side History block is deleted** (`belief/history.py`). Appending it to a v6
  vector produced 674 dims in which H1/H2/H3 duplicated columns `[564:591]`. Equivalence was
  verified on the sample replays: 1,143 / 1,172 emitted decisions matched bit-for-bit, and the
  29 that did not were H3 rows where the **engine accumulators are the more faithful side**
  (the replay walk missed synthetic `PASS`es inserted by `bsw/replay.py::_sync_for`). Since
  inference has only the engine accumulators, dropping the block also closes a train/serve skew.
- **H4 `play_time` is not carried forward.** It was already rejected here by ablation
  (B-full < B-core), so v6 having no equivalent channel is not a loss.
- **The A / B_core / B_full tiers are deleted.** They existed to run an ablation that this ADR
  settled; at v6 all three would select the same channels anyway. The belief input width is
  pinned to `FEATURIZER_OUTPUT_DIM` by `tests/training/belief/test_input_spec.py`.
- **`belief_input_version` survives and does its job**: bumped `v1 -> v2`, it invalidates every
  History-block bundle independently of the policy featurizer pin — the separation this ADR
  introduced is what made the fix a version bump rather than a silent re-slice.

## Rejected alternatives

- **(C) Sequence model over the raw play log.** Most expressive; not blocked by latency for a
  Phase-2 model — but *materialising* its input is the terabyte explosion. If ever pursued, it
  would run replay-on-the-fly (compute the sequence at train time), never stored. Deferred as
  the escalation if B-full leaves signal on the table.
- **Reuse the 224 policy featurizer only (status quo / Tier A as the product).** Cheapest, but
  no information edge over the policy and blind to cross-Trick declines — the bet would rest
  entirely on supervision. Kept only as the ablation floor.
- **56-card play-time vector as the whole history (the first brainstorm).** Player-anonymous
  and structurally silent on Passes (a Pass touches no card slot), so it cannot carry the
  negative-information signal it was meant to. Retained only as the H4 channel.
