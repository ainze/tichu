# ADR-0038: Featurizer v6 — per-player `played_by`, self-only `schupfen_received`, cross-Trick negative-info channels, trick leader

- **Status:** Proposed
- **Date:** 2026-06-14
- **Supersedes (in part):** [ADR-0015](0015-featurizer-v3-drop-leaky-and-redundant-sections.md) — **Finding 3** (the rules-violation classification of `schupfen_received`) is **factually wrong** and is reversed here. ADR-0015's deferral of `played_by` (its rationale 5) is taken up here.
- **Related:** [ADR-0017](0017-featurizer-v4-compact-trick-top-combo.md), [ADR-0019](0019-bit-pack-materialised-bundle.md), [ADR-0028](0028-belief-input-compressed-history-projections.md), [ADR-0024](0024-master-tier-conditions-on-top-skill-decile.md), [ADR-0027](0027-docker-serving-decile-mapped-single-bc-checkpoint.md), [ADR-0033](0033-perfect-info-critic-escape.md), [ADR-0034](0034-full-stack-cotraining-ppo.md)

## Goal of record (Framing X)

This work is a **richer-input, decile-9-targeted BC base** — *not* a claim that BC alone produces exploitative / superhuman play. BC caps at imitating top-decile humans; exploitation stays a **downstream** (PPO / co-train / search) problem on top of a better base. Success metric: **decile-9 Move-Prediction Top-k goes up** without regressing the **Tournament Matrix**. The justification for the new content is **imitation fidelity**: the decile-9 human being cloned *saw* who played each card and who passed them which Schupfen card; the v5 featurizer is blind to both, so it is structurally under-informed relative to its teacher. The repo's own capacity verdict ([2026-06-02-bc-is-label-bound-not-capacity-bound.md](../notes/2026-06-02-bc-is-label-bound-not-capacity-bound.md)) names the live lever exactly: *"the 512-dim representation … is bounded by the Featurizer content (e.g. it encodes no opponent-hand belief), not the trunk width."* v6 is that content lever.

## Context

The v5 featurizer (224 dims) carries **`seen_cards[56]`** — a position-less multi-hot over `PublicState.played_cards_this_round` (every card played this Round by *anyone*, including self). It answers "is this card gone?" but **not "who played it?"** For card-counting / void inference — the core belief signal a strong player uses — *who* is void in a suit determines whether a lead is safe; the aggregate can't say. Separately, the v5 featurizer carries no record of which Schupfen card arrived from which seat, a signal human players act on (receiving the Dog from your partner is a near-tell that the partner will call Tichu).

Two engine facts gate the change:

1. **No play attribution exists in state.** `PublicState.played_cards_this_round` is a bare `frozenset[CardOrSpecial]` ([state.py:110](../../src/tichu_engine/state.py)). Per-player provenance is available *during* replay (the engine processes plays one at a time) but is discarded into the set. ADR-0015 rationale 5 flagged that `played_by` *"would require engine changes … adding fields to `PublicState` triggers ADR-level work."*

2. **`schupfen_received` was removed on a false rules premise.** ADR-0015 Finding 3 deleted the v2 `schupfen_received` block (3 directions × 56) as a **rules violation** — *"the rules do not let them know which opponent vs their partner gave them which card."* This is **incorrect for Tichu / BSW**: Schupfen passes to *specific seats* and is collected from *specific seats*; the face-down rule enforces *simultaneity* (no one commits after seeing others' passes), **not anonymity of source**. The receiver legitimately knows the provenance of all three received cards. ADR-0015's secondary objection (a train/deploy mismatch because the signal "won't be there in honest play") therefore also evaporates — the **Wire PrivateState** can carry it honestly at inference. The feature was deleted on a factual error about the rules.

## Decision

**Bump `FEATURIZER_VERSION` to `"v6"`. Two content changes.**

### 1. `played_by` — 4 relative-seat planes, replacing `seen_cards`

Replace `seen_cards[56]` with **`played_by[4][56]` = 224 dims**, ordered in **relative-seat** convention `[self, next, partner, previous]` (= `(seat+0, seat+1, seat+2, seat+3) % 4`), matching `perfect_info.py:35` and `belief.emit`. Each card played this Round is set in exactly one plane (the seat that played it), so the planes are a one-hot-over-players per played card; the old aggregate is the OR of the four planes and is therefore **strictly redundant — dropped**.

- **Self plane is retained, not dropped.** The acting seat sees `own_hand` (current holdings) but never its own *initial* deal, so its *played* cards are only knowable via this plane; without it the model would wrongly treat its own discards as possibly-in-opponents'-hands. (This is why the user's "3 other players" framing becomes 4 planes.)
- **Relative-seat ordering is a correctness requirement,** not a convenience: absolute-seat planes would break under the Tournament's Seat-Swap and the trunk would learn seat-specific junk.
- Net featurizer dim from this change: **−56 + 224 = +168.**

**Engine change (accepted blast radius):** add `played_cards_by_player: tuple[frozenset, frozenset, frozenset, frozenset]` to `PublicState` (public — everyone saw the plays), reset per Round, maintained as each Play resolves. This same `PublicState` change also carries the **decline accumulators** for section 3 below — one schema change, one codec pass, one Replay-Validation run, not three. The field is additive and must not change any reproduced Ergebnis.

### 2. `schupfen_received` — re-added, self-only

Re-add a **self-only `schupfen_received[3][56]` = 168 dims** (3 give-directions × 56 cards) for the acting seat's *own* three received cards. **Self-only** — a player never sees what opponents passed each other, so this is *not* a per-opponent grid. Carried in the **shared** Feature Vector, so it feeds every net that consumes `featurize` (Play, Tichu-Call, Schupfen, Grand-Tichu). It is **informative only post-Schupfen** — live for **Tichu-Call** (featurizes at first non-Pass Play, ADR-0018) and **Play**, and a structurally **all-zero passenger** for the **Schupfen** and **Grand-Tichu** decisions (which featurize *before* the seat has received anything). The dead-but-harmless passenger is accepted in exchange for keeping a single shared Feature Vector.

### 3. Cross-Trick negative-information channels (ADR-0028 B-core), into the *policy*

Add ADR-0028's **B-core = 27 dims** — `declined_top[3 opp × 6 intent-types = 18]` + `lead_summary[3 × 2 = 6]` + `pass_pressure[3 × 1 = 3]`, **relative-seat ordered** (next/partner/previous) — to the **shared policy featurizer**. This is the **richest belief signal in trick-taking** and the *one* history channel ADR-0028 *measured* as earning its bytes (B-core > A). `played_by` is structurally blind to it (a **Pass** touches no card slot), and the v5 featurizer keeps only the *current*-Trick `trick_passes`; the round-long, per-opponent **negative** record is absent.

This is **not** featurizer-only: the decline/pass/lead history is not reconstructable from a single `PublicState` snapshot, so the per-opponent accumulators ride the **same `PublicState` schema change** as `played_by` (public — everyone saw the passes). Net dim: **+27.**

It is also the **input rung** of the belief ladder (see "Decisions not taken"): the same channels are what give a belief aux head its input edge, so this rung and the aux-loss rung are run consecutively to keep attribution clean.

### 4. `trick_leader` — current-Trick leader, featurizer-only

Add **`trick_leader[4]`** (relative-seat `[self, next, partner, previous]`, all-zero on an empty Trick) — *who led the current Trick*. Genuinely missing and tactically central (mid-Trick the features can't say whether **partner** or an **opponent** led — i.e. whether to step aside or contest); not derivable from `{top_combo, passes, current_player}`. **Free:** `Trick.leader: int | None` already exists ([state.py:42](../../src/tichu_engine/state.py)) and `pub.trick` is already read by the featurizer — it is simply never emitted. **No engine change, no codec, no replay impact.** Net dim: **+4.**

(`tricks_won` *count* was considered and dropped: partly derivable from `round_points_by_player`, weak tempo signal, and it *would* need a new `PublicState` counter — `trick_leader` is the genuinely-new, free slice it was groping toward.)

### Resulting layout

`FEATURIZER_OUTPUT_DIM`: **224 (v5) → 591 (v6)** = 224 − 56 (`seen_cards` out) + 224 (`played_by`) + 168 (`schupfen_received`) + 27 (B-core) + 4 (`trick_leader`). `perfect_info.py`'s `PERFECT_INFO_DIM` tracks it (`+367` on the observable block; the `3×56` true-hand grid is unchanged).

## Decisions explicitly *not* taken here (recorded so they aren't re-litigated)

- **Trunk capacity is NOT increased.** A 2.25× trunk moved play-acc +0.08pp ([2026-06-02 note](../notes/2026-06-02-bc-is-label-bound-not-capacity-bound.md)); BC is label-bound. The actor stays **1024×4**. The user's *PPO-headroom* argument (untested by that note) is real but is routed to **the critic**, where the measured under-fit actually lives (co-train critic R²: play 0.45, grand 0.10, schupfen 0.25). Any actor size-up is gated on a **modest-width PPO smoke** ("does a wider actor let PPO move further off BC?"), decided in the PPO/co-train ADR, not baked in here.
- **Decile-9 data is NOT hard-filtered as the default.** Conditioning already delivers decile-9 *behavior* ([2026-06-03 note](../notes/2026-06-03-master-matches-decile9-human-behavior.md)) and reverses ADR-0024/0027 serving. Baseline = **all-data + conditioning, served @ decile-9**; **decile-9-only is a measured arm** (filter on the *acting* seat's decile), decided by a **head-to-head Tournament** (the only instrument that can see a sub-behavioral / negative-transfer win — Top-k is Bayes-ceiling-bound and won't). Requires building a **decile-9 held-out set**.
- **Belief-into-BC is a ladder, not a single arm.** Attribution is kept clean by separating the **input** edge from the **supervision** edge across three consecutive rungs:
  1. **v6 control** — `played_by` + `schupfen_received` + `trick_leader`, play-imitation loss. (Does positive provenance + leader alone help?)
  2. **+ B-core negative-info input** (section 3) — adds the decline/pass/lead channels to the *policy* input. (Does the *negative-information input* help the play head directly?)
  3. **+ auxiliary belief head** — adds a belief-prediction aux loss on the same trunk/input. (Does *supervision on ground-truth opponent hands* shape a better trunk on top?)

  The **no-belief rungs (1–2) are the control** for rung 3 — else a play-acc gain can't be attributed between input and supervision. The aux head has **no frozen-net pipeline dependency** and **no change to the served inference path** (the *"Belief not used in Phase-1 inference"* invariant holds). Frozen-output fusion is the escalation if rung 3 shows signal. Plain reuse of an A-tier belief (policy-features-only input) is provably useless per ADR-0028's no-edge theorem. This mirrors ADR-0028's own A / B-core / B-full discipline.

  **Implementation note (surfaced building v6):** the Belief Model's `TIER_DIMS` now rebase on the v6 policy width (`A=591, B_core=618, B_full=674`, `belief.history`). Because v6's policy featurizer *already carries* the B-core negative-info channels, the belief module's **B_core history tier is now redundant** with the policy block — the rung-3 redesign must drop that double-count (the belief edge that remains is H4 `play_time` and, above all, the **supervision** on ground-truth hands, not the re-supplied declines). Tracked as deferred belief-rung work, not done here.

  **Resolved 2026-07-27 — v6 absorbs ADR-0028 B-core; the History block and the tier machinery are deleted.** The deferred double-count above was live in the emit path: `belief/emit.py` still appended the 83-dim History block to what was now a 591-dim policy vector (674 total), of which H1/H2/H3 (27 dims) duplicated v6's `declined_top` / `lead_summary` / `pass_pressure` sections. The belief input is now **exactly the v6 Feature Vector** (`belief/input_spec.py`, `BELIEF_INPUT_VERSION = "v2"`), with no suffix and no tier prefix:

  - **H1/H2/H3 → dropped as duplicates.** Verified on the sample BSW replays (1,172 emitted decisions): the v6 featurizer's B-core slice `[564:591]` reproduces the old accumulator's H1/H2/H3 block on **1,143 / 1,172 rows bit-for-bit**. The 29 exceptions are all in H3 `pass_pressure`, where the v6 channel is the **more faithful** of the two: `bsw/replay.py::_sync_for` steps synthetic `PASS`es through the engine that it never appends to `replay.decisions`, so the replay-side accumulator under-counted them while the engine's live accumulators (what the featurizer reads) see them. Dropping the block therefore removes a small train/serve skew as well as the duplication — at inference only the engine accumulators exist.
  - **H4 `play_time` → not re-added.** ADR-0028's own ablation already rejected it (*B-full < B-core*), so v6 having no equivalent channel costs nothing. The `played_by` planes carry provenance without ordering; if ordering is ever wanted it is a new, measured featurizer section, not a belief-only suffix.
  - **A / B_core / B_full tiers → deleted** (`TIER_DIMS`, `_select_tier`, the `tier:` config key, `belief_naive_baseline.py --tier`). The ablation they existed to run is settled and at v6 all three tiers would slice the *same* channels. A leftover `tier:` key now raises rather than being silently ignored, and `tests/training/belief/test_input_spec.py` pins the belief input width to `FEATURIZER_OUTPUT_DIM` so the next featurizer bump fails loudly instead of re-opening this class of bug.
- **The Claim-Solver signal is a *label*, not a BC feature.** As a BC *input* it only helps to the extent decile-9 humans already act on the certainty (then the imitation gain is small by construction) and risks adding label-uncorrelated variance otherwise. As a *label correction* (distillation in states where the solver fires) it is unambiguously valuable and label-independent. Routed to the **B3 strength lever** ([2026-06-14-ppo-strength-lever-backlog.md](../notes/2026-06-14-ppo-strength-lever-backlog.md)), not the v6 featurizer. Cheap pre-check: fire the solver on a decile-9 held-out endgame slice — high agreement → redundant-with-label (drop); low agreement → exploitable headroom only a label correction can capture (confirms B3).

## Rationale

1. **Content, not capacity, is the live lever** — the repo's own conclusion. `played_by` is opponent-hand-belief content (void/provenance inference) the featurizer demonstrably lacked.
2. **`played_by` is the *certain* half of belief.** It observes what each opponent *played* (ground truth); the **Belief Model** estimates what they *hold* over the **Unseen Cards** (probabilistic). Complementary, not redundant — `played_by` is the reliable void-inference substrate.
3. **`played_by` and ADR-0028's B-core are orthogonal.** `played_by` is *positive* provenance; B-core's `declined_top` / `pass_pressure` is *negative* (Pass) inference, which `played_by` is structurally blind to (a Pass touches no card slot). They stack.
4. **`schupfen_received` is legitimate observable signal, full stop** — once the rules fact is corrected. Its home is the Tichu-Call and early Play, exactly where humans use it.
5. **One shared Feature Vector** keeps the dead-passenger cost (Schupfen/Grand-Tichu) far below the cost of forking per-net featurizers.

## Consequences

- **A full v6 stack rebuild.** v6 invalidates **every** v5 artifact — BC / Call / Schupfen / Belief checkpoints, materialised bundles, AWR, and the co-train **Resume Bundles** (ADR-0034 warm-starts from v5 BC). Engine `PublicState` + Wire codec change + a full Replay-Validation pass. Consistent with the "from the top once more" intent; not a small diff.
- **Honest payoff flag.** The nearest existing test of *positive* card-play history — ADR-0028's **H4 (`play_time`)** — was the **lowest-value channel and was dropped** (B-full < B-core). `played_by` differs (per-player attribution; policy-imitation objective), and the strongest case for it is **void inference**, not raw provenance — but go in clear-eyed that positive play-history has under-delivered here once already. The decile-9 Top-k delta is the cheap referee.
- **ADR-0015's status updated** with a forward pointer; its Finding 3 stands corrected.
- **Storage.** Per-row feature footprint grows ~2.5× (224→560 dims, still indicator-heavy → bit-packs well per ADR-0019). Re-checks the full-corpus quantisation math but does not change its shape.

## Rejected alternatives

- **3 opponent planes + keep aggregate `seen_cards`.** Informationally identical to 4 planes (each card played by exactly one seat) but leaves the self plane derivable-not-explicit; 4 explicit relative-seat planes are cleaner for the trunk and match `perfect_info` / `belief.emit`.
- **Keep `schupfen_received` deleted** (trust ADR-0015). Rejected: ADR-0015 Finding 3 is factually wrong about Tichu/BSW; the deletion forfeited real Tichu-Call signal on a false premise.
- **Bigger trunk to "match" the wider input.** Rejected: capacity is label-bound; +336 dims into a 1024×4 trunk is inert on accuracy. See "Decisions not taken."
- **Belief frozen-output fusion as the first cut, or A-tier belief reuse.** Deferred / rejected respectively: aux head first (cheaper, captures the supervision edge, no inference-path change); A-tier reuse is provably useless (no input edge → repackages policy-known info, ADR-0028).
