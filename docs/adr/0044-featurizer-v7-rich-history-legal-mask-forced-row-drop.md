# ADR-0044: Featurizer v7 — additive Rich History block, Legal-Mask Trunk Input, forced-row corpus drop

- **Status:** Accepted (design). Not yet built.
- **Date:** 2026-08-01
- **Reclaims the "v7" name from** [ADR-0039](0039-featurizer-v7-trick-point-stakes-and-current-winner.md), which proposed a *different* v7 (trick point-stakes + current-trick winner), was **Rejected**, and was never built. No `FEATURIZER_VERSION` ever shipped as `"v7"`, so the stamp is free. A reader finding two "v7" ADRs should read 0039 as a dead proposal and this one as the live definition.
- **Related:** [ADR-0038](0038-featurizer-v6-played-by-and-schupfen-received.md) (the v6 engine-accumulator pattern this extends), [ADR-0028](0028-belief-input-compressed-history-projections.md) (B-core, which the Rich History Block supersets), [ADR-0041](0041-belief-ev-value-gate.md) (where the block came from), [ADR-0040](0040-frozen-reference-vine-pooled-ratchet.md) (the −2.69 wishfix-vs-cpfix3328 gap this run must close), [PR #80](https://github.com/ainze/tichu/pull/80) (the legal-mask pre-check), [ADR-0014](0014-pre-featurise-bc-corpus.md) / [ADR-0019](0019-bit-pack-materialised-bundle.md) (the bundle this re-materialises).

## Context

Three changes were ready at once, with different natural homes:

1. A **233-dim Rich History Block** (`belief/rich_history.py`, built for ADR-0041) recovering decline *context*, combo length, declined bombs, trick stakes, the proven wish void, and play order — a strict superset of v6's 27-dim B-core.
2. PR #80's **Legal-Mask Trunk Input** — greenlit at **+3.04% relative NLL** on 300 k decile-9 non-forced Play Decisions, with a **negative (−2.24%) capacity placebo** and flat across an 8× rows × 8× width sweep.
3. Dropping **Forced Play Decisions** from the BC corpus, the counterpart to the rollout drop already shipped in `7874c9e`.

PR #80 explicitly warned against bundling these. We are bundling them anyway — one re-materialisation instead of three — and this ADR records why that is affordable here and what was done to keep it recoverable.

The **Rich History Block's pre-check never returned a verdict**: `scripts/precheck_rich_history.py` was killed mid-collection and the run was abandoned. It is carried on an **assumed** positive. This is the single weakest premise in the design and is called out here rather than buried.

## Decision

### 1. v7 is strictly additive — `v7[:591]` is byte-identical to v6

New sections append to the end of `SECTION_DIMS`, so `SECTION_OFFSETS` places them after `pass_pressure`. `FEATURIZER_OUTPUT_DIM` goes **591 → 824**. A one-line regression test pins the prefix.

The 27 B-core dims (`declined_top`, `lead_summary`, `pass_pressure`) are **redundant with the new block and are kept anyway.** Reclaiming them would save 3.3% of the vector and cost the prefix property — a bad trade, because the prefix property is load-bearing in two places:

- **cpfix3328 stays usable at v7 behind a `feats[:, :591]` slice.** `ppo/cotrain.py:367` featurizes **once** per decision and shares the tensor across seats, and `ppo/greedy_gate.py:78` rebuilds pool opponents through a single live `arch_cfg`. Without prefix-compatibility, keeping the served champion in the gate would require a frozen `featurizer_v6.py` (the `featurizer_v5_frozen.py` pattern) **and a second `featurize()` call in the rollout's hottest loop.** With it, it is a slice.
- **A v6-equivalent control arm is provable rather than asserted**, and remains runnable against the *same bundle* at any later date.

### 2. The Rich History Block lives in the engine, not the featurizer

Extend `PublicState` and `_fold_decision` (the ADR-0038 pattern): accumulators stored **raw**, normalised by the featurizer, as **one frozen `RichHistory` dataclass of tuples** (`tichu_engine/rich_history.py`).

*Revised during implementation.* This ADR originally specified a small immutable **numpy array** copied per fold, for speed. That is wrong as an *interface*: `PublicState` is `frozen=True`, and a numpy field breaks the generated `__eq__` (`array == array` returns an array; `bool()` on it raises), while `tests/training/featurizer/test_featurize.py` pins featurizer purity through `repr(s.public)`. A perf guess does not belong in the type — the internals can change behind the field if profiling ever demands it.

Rejected: a **mutable** accumulator hung off `GameState` (zero-copy, but this repo snapshots states throughout `vine.py`, `pmcpa.py`, `blunder_miner.py` and `forced_claim.py` — an aliased mutable accumulator would let a replayed state see the future); and **featurizer-side accumulation by each caller**, which is precisely the shape that produced the shipped `team_scores` and schupfen `current_player` skews.

`_fold_decision` is **not the only fold site** — `engine.py:133` (Dog) and `engine.py:580` (Bomb) fold separately. All three must carry the new channels, with a test each. `BombInterrupt` has already bitten this code twice.

### 3. The Legal-Mask Trunk Input goes to the policy trunk only

Raw 1809-wide mask concatenated **inside** `BCModel`, behind a constructor flag, via a 3-argument `forward(features, skill_decile, legal_mask)`. Trunk input becomes `824 + 64 + 1809 = 2697`; `input_proj` grows 909 K → 2.76 M params.

- **Non-play rows get zeros.** Head masks are 1809 / 14 / 2 wide and the trunk needs fixed width. Play is 96.8% of rows, `phase` already tells the trunk the decision type, and zeros are literally true ("no play-legality information here"). Padding a 14-wide wish mask into play's action-index space would be semantically incoherent.
- **No critic gets the mask.** PR #80's +0.017 R² was measured on a value head over the **observable** 591 features. The cotrain critic is the **Perfect-Info Critic** (`PERFECT_INFO_DIM`, 992 at v7), which sees every hand and can derive legality exactly — the mask adds no fit. The two `ValueBaseline` instantiations in `cli/train_bc.py` are inside AWR code paths, which are off the BC → cotrain route.
- **The concat is inside the model, not at the call sites**, so the exported TorchScript carries a 3-arg signature and an `MLAgent` that fails to supply the mask **fails loudly** instead of silently feeding zeros.

At `gamma=1, lam=1` (live since ~iter 22.46k) the advantage is exactly `R − V(s)`, so the critic is a **pure variance-reducing baseline** and cannot bias the gradient. It is still required: without it `A = R`, the raw round outcome, which is the R−V variance wall that drowned ADR-0034/0035.

### 4. Forced Play Decisions are dropped at materialise time, at Intent level

The predicate is **`legal_mask.sum() == 1`** — one legal **Intent**. This is *not* the rollout's predicate: `rollout._forced_action` tests `len(legal_actions(state)) == 1`, i.e. one **ConcreteAction**. `|concrete| == 1` implies `|Intent| == 1` but not the reverse (one "Single 7" Intent, two suited 7s). Both are correct for their own purpose; the Intent-level one matches the 1809-way softmax the BC gradient flows through, and matches the slice PR #80 was greenlit on.

**Drop the row, not the step.** Replay still advances through forced decisions; only emission is skipped, and only for the play head. `7874c9e` found this trap on the rollout side: ~8% of forced actions are *plays*, and ADR-0018 solicits a seat's Tichu Call at its first non-Pass play, so a naive `continue` silently costs a seat its call.

**Materialise-time, not read-time — forced by disk, not preference.** The v6 bundle measured **550 GiB**. The Rich History Block adds 155 continuous + 78 binary dims = **+630 B per play row**, taking a full-corpus v7 bundle to **~1,263 GiB**. Dropping forced rows (−39%, 1.248 B → 761 M) brings it to **~771 GiB**, which fits only after deleting the v6 bundle (done; regenerable from `parquet_full_v6_wishfix` + `archive.zst`, and `parse_bsw` is featurizer-independent so no BSW re-parse is needed). A read-time filter would have required writing the 1.26 TiB bundle. PR #80's advice to prefer a read-time filter was written without this arithmetic.

The usual cost of a materialise-time drop — losing the with-forced arm — is small here: forced rows are **provable no-ops on the learned BC function** (one legal Intent ⇒ masked softmax puts probability 1 on it ⇒ `nll = 0`, gradient exactly 0), and the with-forced baseline survives as a trained model in `runs/bc_full_corpus_v6_wishfix_memmap`.

**Consequence — the play-head learning rate. RETRACTED (2026-08-03).** This ADR originally said: `masked_cross_entropy` reduces with `.mean()`, so removing 39% zero-loss rows scales the play-head loss and gradient by `1/0.61 ≈ 1.64×`, therefore scale the play-head LR by 0.61.

The gradient scaling is real, but the conclusion does not follow **because BC trains with `torch.optim.Adam`** (`cli/train_bc.py:102`), which is invariant to a constant loss scale — the factor cancels in `m / sqrt(v)`. Measured: 50 steps at 1.0× vs 1.64× loss leave a max parameter difference of **3e-8** (float noise); the same experiment under SGD gives **2e-3**. **Do not touch the learning rate.** The advice would have been correct for SGD, and would silently have slowed the play head by 39% here.

**Consequence — reported top-1 will appear to collapse and will not be a regression.** Forced rows are ~100% correct for free. A v6 figure of ~0.75 corresponds to `(0.75 − 0.39)/0.61 ≈ 0.59` non-forced — which is exactly PR #80's measured 0.5991 baseline. **Every BC comparison must be non-forced-rows-only on both sides.**

### 5. Only `BCModel` and the Schupfen Network are retrained; the Call Networks are widened

All four nets take `FEATURIZER_OUTPUT_DIM`, so a v7 bump invalidates every v6 Checkpoint. Naively that is four corpora and four trainings. Instead:

- **`BCModel`** — full re-materialise + retrain. This is the experiment.
- **Schupfen Network** — retrained, and **rebuilt from engine-real states**. See §6.
- **Tichu / Grand-Tichu Call Networks** — **widened**, not retrained: copy v6 weights into the v7 first layer and zero-init the 233 new columns. Because the pre-`input_proj` concatenation is `[features, skill_emb]`, the v6 skill columns move from `591:655` to `824:888` — a pure permutation, since the feature prefix is byte-identical. The result is **exactly** function-preserving, not approximately, and ships with an assertion test.

Grand Tichu is called on the first 8 cards, before Schupfen; the Rich History Block is **identically zero** there, so widening is provably lossless. Small Tichu can be called mid-round so its block is partially live, but [calls are a closed null](../notes/2026-07-29-calls-are-not-a-lever.md) — a corpus and a training run on a measured-null axis is not worth it. The bet is that the calls-null holds.

### 6. The Schupfen Network is rebuilt from engine-real states

`_schupfen_examples_for_round` hand-builds a synthetic `PublicState` literal rather than using the engine's. `a3438f9` patched one field of it (`current_player=0` → the acting seat, worth **+4.03** on a head-swap A/B); `hand_sizes=(14,14,14,14)`, `tichu_callers=frozenset()` and `trick=Trick.empty()` are the same class of risk, and **at v7 the Rich History accumulators become four more**. Build the example from the engine's actual schupfen state, as `bc/dataset.py` does for play/wish/dragon — measured clean at 0.000% skew over 84 k decisions. This kills the class, not the instance.

`scores=(0,0)` in that literal is **not** a bug: the training convention is round-local everywhere and `round_local_team_scores` derives it from `round_points_by_player`, which is `(0,0)` at Schupfen on both sides.

### 7. Score-awareness is explicitly deferred

The nets are **score-blind** — they never see the game-cumulative score, so they cannot play safe at 900–200 or gamble at 200–900. The data exists (`ParsedRound.ergebnis` over ordered `ParsedGame.rounds`). It is deferred because:

- It would have to be **new dims**. Changing the existing `team_scores` section's semantics from round-local to cumulative would feed cpfix3328 out-of-distribution values in those two columns through the `[:591]` slice, corrupting the exact comparison this run exists to make.
- The BC side plausibly needs a **BSW re-parse** — the parallel materialiser works per-round with no running ergebnis — which is the one expensive stage v7 otherwise skips.
- The cotrain side is the real work: rollouts start from the position pool at 0–0, so without **seeding the pool with varied cumulative scores** the new dims are constant across every rollout and the feature ships dead. **The pool seeding is the deliverable; the featurizer dims are the easy part.**

### 8. Lineage and gate

`warm_start`: play ← the new v7 BC; schupfen ← the new v7 Schupfen Network; tichu / grand ← widened from the **`_cpfix`** Checkpoints. These stay **BC** Checkpoints rather than cpfix3328's cotrained exports, so `bc_models` (a frozen deepcopy of the warm-started nets, and simultaneously the KL anchor and the gate's `"bc"` floor) keeps its meaning as the always-cleared floor.

cpfix3328 starts in `promotion_gate.extra_opponents` as **`observe_only`** — scored and logged every window, never able to block a promotion. That mechanism already exists and needs no change beyond the `[:591]` slice wiring. It starts observe-only because a *required* cpfix3328 hard-blocked every promotion in the v6 run (learner −9.9 at iter 512): a bar you are nowhere near is a deadlock, not a ratchet.

**Flipped to required at iter 1280** (see the Implementation note below), once the run cleared it CI-clean.

**Pre-registered fallback:** if the v7 cotrain stalls below cpfix3328 at matched window count, **widen cpfix3328 itself into v7 shape** and cotrain from there — starting at parity by construction instead of in ADR-0040's −2.69 hole. Do **not** loosen the leash (measured harmful twice). The cost of that route is that the 2,042 new input columns are zero-initialised and must be discovered by PPO alone, which abandons the supervised bet this ADR exists to make — hence fallback, not plan.

## Implementation notes (added after the build; 1,362 tests green)

- **The 155/78 continuous/binary split is confirmed by the code**, not inferred from the prototype: `CONTINUOUS_FEATURE_COLUMNS` reports exactly 155 continuous v7 dims. The bundle writer's bit-packing guard rejects any other split, so the +630 B/row behind the 771 GiB estimate is now enforced rather than assumed.
- **`card_slots` moved to `tichu_engine`.** `card_play_order` is slot-indexed, and `card_slot` lived in `tichu_training`, which itself imports `tichu_engine.cards` — accumulating slot-indexed state engine-side would have inverted the layering. The table is now `tichu_engine/card_slots.py` with `tichu_training.card_slots` a re-export, so every existing importer uses the same table.
- **Combination length is `combo.length`, not card count.** The belief prototype's `_combo_length` returns `len(action_cards(...))`, which makes a 3-long PairStep (6 cards) indistinguishable from a 6-long Straight. The engine fold uses the declared length, matching the featurizer's own `length` subfield.
- **Rebuilding Schupfen from engine state fixed no live bug.** A probe stepping the engine through Schupfen and diffing against the old hand-built literal found **0 differing dims on all four seats** — `current_player` already advanced exactly as the cpfix assumed. The change is structural: `tichu_engine.state.schupfen_start_state` is now the single round-start constructor shared by `deal_for_schupfen` and the BC emitter, so a future `PublicState` field cannot default differently in the two paths. Note the emitter-vs-engine test can only prove the two *agree*; `tests/engine/test_schupfen.py` pins the constructor against engine behaviour independently (mutation-checked).
- **cpfix3328 became a REQUIRED gate opponent at iter 1280 — held out of the rollout.** The gate promoted at 1280 with cpfix3328 at **+1.489 (ci_lo +0.276)**, the first window in which any lineage cleared the served export CI-clean (champion +1.184, bc +8.928; ADR-0040 had the wishfix champion at −2.69). From that point an observe-only bar is the weaker instrument: a promotion that does not clear cpfix3328 advances a champion not worth shipping, and `reanchor_on_promote` moves the KL anchor with it. `require` and rollout membership were **one flag** and are now two (`resolve_gate_opponents`, `rollout` defaulting to `require`, so every prior config resolves unchanged). cpfix3328 runs `require: true, rollout: false` — deliberately **held out**: including it would swing the rollout mix from champion/bc 50/50 to thirds at the same moment the bar changed, making a subsequent stall unattributable, and would train the learner to best-respond to the very opponent the ship decision is read off, inflating the margin that is meant to be the evidence. Requires the greedy gate — a sampled window reads the rollout stream, so a stream outside the rollout cycle can never fill (a silent stall under observe-only, a hard deadlock under require).
- **Serving strictness is a switch, not a default.** `public_state_from_json(..., require_v7=True)` raises on a payload without `rich_history`; the default stays lenient so the current client keeps working until it is updated, per the decision above. Making it strict today would break `/act` immediately.
- **The forced-row drop is `--skip-forced-play` on `materialise_bc`**, defaulting **off**, threaded through both the sequential and parallel datasets. The v7 corpus must be materialised **with the flag on**; leaving it off silently produces a ~1,263 GiB bundle that does not fit.

## Consequences

- **The rich block is carried on an assumed pre-check.** Its defence moves downstream to the BC result. Because v7 is additive and the mask is a flag, the isolating arms (`[:591]` × mask-on/off) remain runnable against the **same bundle** at zero re-materialisation cost. That option is deliberately preserved, not exercised now.
- **A v7 bump breaks every remaining v6 artifact** the same way `test_pikl.py` is already broken at v5→v6. Budget for it.
- **The served agent's inputs are not what training produces, and v7 widens the gap.** `documentation/INFERENCE_API.md` predates ADR-0038 and documents none of `played_cards_by_player` / `declined_top_by_player` / `lead_summary_by_player` / `pass_stats_by_player`; `codec.py` defaults all of them to zeros via `blob.get(..., default)`. If the client was never updated, the served agent runs v6 with **251 of 591 dims (42%) pinned to zero**, and v7 would take that to **484 of 824 (59%)**. The decision is to **ship the serving path so the client *can* supply the v7 fields, and update the client only after beating cpfix3328** — the ship bar is the in-process Tournament Matrix, where accumulators are live, so that comparison stays faithful. **The v7 codec fields must not silently default to zeros**, or a half-updated client will corrupt 59% of inputs undetectably.
- **Out of scope but found while designing this:** `configs/cotrain_v6_pbrs_resid_wish_gated_wishfix.yaml:33` warm-starts schupfen from `schupfen_full_corpus_v6_resid` (6/21 14:25, **pre-cpfix**) rather than `..._resid_cpfix` (6/23 12:31). Since that net is simultaneously the learner init, the schupfen KL anchor (tight: target 0.008 → 0.025) and the gate's `"bc"` floor, the wishfix lineage has been holding a known-skewed schupfen prior — a concrete mechanical candidate for ADR-0040's −2.69. Not proven (schupfen is co-trained and may partly re-learn), but any new cotrain must warm-start from the `_cpfix` Checkpoint.
