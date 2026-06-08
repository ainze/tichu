---
status: proposed
---

# Full-stack co-training PPO — sharpen play + Schupfen + Tichu/Grand calls together, all-in

[ADR-0033](0033-perfect-info-critic-escape.md) refuted its own ESCAPE hypothesis (the
asymmetric Perfect-Info Critic did not unstick PPO — the ~6th flat RL result) and named
the **one** structural lever left untested: **joint co-training of play with the frozen
subsystems** (Schupfen / Call Networks). Its coupling probes then falsified the call
lever (Tichu-rate flat on EV) but found schupfen **highly sensitive** (`schupfen_rule`
−51.8 vs master) — necessary-not-sufficient evidence that BC's *jointly-cloned*
play+schupfen are already co-adapted at decile-9, settleable only by co-training. This
ADR records the **build design** for that co-training run, locked in the grilling session
of 2026-06-07. The owner's directive: **go all-in** — one multi-day run, no cheap gating,
the bar relaxed to *any* seat-swap Tournament 95% CI excluding 0 (positive). New
vocabulary (**Full-Stack Co-Training**, **Resume Bundle**) is glossed in
[CONTEXT.md](../../CONTEXT.md) §"Phase 2 / online-RL terms".

## Decisions

**Scope — all-in, all nets learnable from iter 0 (no staging).** One PPO run sharpens the
play Trunk+head, the **Schupfen Network**, the **Tichu Call Network**, and the
**Grand-Tichu Call Network** simultaneously. The **Belief Model** is excluded (a predictor,
not a policy). Wish / dragon_assignment ride the play Trunk → **anchored-not-trained** (the
ADR-0029 treatment). The handoff recommended *staging* (play+schupfen first, then calls) for
attribution; the owner overrode it. Attribution is instead preserved **without** staging by
logging **per-net dials every iteration** (schupfen KL-from-BC + call-bonus component; Tichu
/ Grand call rates; play entropy/KL) into `ppo_log.csv`, so a flat result is still locatable
to a net without a second run.

**Collector — extend `rollout.py` into a full-stack *batched* collector (not built on
`play_full_round`).** Keep `rollout.py`'s lockstep batched-play hot path (the only hot
decision — dozens/round; calls/schupfen are once/round). Port `play_full_round`'s
solicitation rules (Grand at deal-time, Tichu at the seat's first non-Pass per
[ADR-0018](0018-tichu-call-featurises-at-first-non-pass.md), Schupfen on `SchupfenPending`)
as additional per-tick batched groups, each recording a trajectory for its net and
**replacing the frozen RuleAgents** that currently serve them. Rejected building *on*
`play_full_round`: its batch-1 single-game self-play is the AWR-painful pattern ADR-0029
deliberately rejected, and unattended throughput matters for a multi-day run. The handoff's
divergence worry is killed by a **parity test**: identical deals through the extended
collector vs `play_full_round` must produce identical callers / schupfen / call-bonus.

**Per-net losses + credit assignment.** A generic PPO sub-update — clipped surrogate +
KL-anchor-to-its-own-frozen-BC + entropy — applied per net, with steps grouped by net-type
in the update. Action encodings: play = existing masked categorical; calls = binary (2
logits); **schupfen = three sequential masked categoricals sampled without-replacement**
(mirroring `ml_agent._act_schupfen`'s distinct-card decode, greedy → sampled), with
logprob / KL / entropy **summed over the three direction heads**. Under γ=1.0 + terminal
`round_outcome` (ADR-0029), every decision in a seat's Round shares the same return, so the
advantage reduces to `R − V(decision_state)` for all decision types — the schupfen
"longest-horizon" problem is *not* a separate mechanism, it is entirely the critic's ability
to value the round-start state.

**Critic — single shared Perfect-Info Critic.** One state-value (392-dim perfect-info
features, ADR-0033's `featurize_perfect_info`) valued at *every* decision-state (play,
schupfen, calls all share the featurizer output). Shared, not per-decision-type: the rare
call/schupfen states (~1/Round) piggyback on the data-rich play-value fit rather than
starving a dedicated head (the AWR value-fit failure mode). Perfect-Info is **on** —
train-time only; the shipped policy is observable-only, so it is free at inference. It is
the one place the PerfectDou/PTIE lever was never tested (the high-variance round-start
one-shots, where hidden hands are a large share of variance) — the ADR-0033 flat result was
play-only. Warm-started via full-stack BC self-play.

**Exploration vs anchor — per-net, schupfen protected.** Four independent
`AdaptiveKLController`s + per-net entropy, all config-exposed (retuned between resumes from
the logged dials). **Schupfen anchored tightest**, loosened only if KL stays in budget while
reward rises — cratering it is the demonstrated, large, immediate downside (random schupfen
−142.9; rule −51.8 vs master); a better joint optimum is only a hypothesis. Play keeps
ADR-0029's small budget (its escalation showed *more* entropy *hurt*). Calls anchored near
BC's calibrated rate (call-rate was flat on EV). No Dirichlet (a tree-search construct).

**Warm-start.** Every net from its v5 BC training checkpoint (corpus-matched to the play
`warm_start`): play `bc_full_corpus_v5_memmap_unshuffled`, `schupfen_full_corpus_v5`,
`calls_full_corpus_v5` (tichu + grand). Critic warm-started under the frozen BC policies
driving the full stack.

**Stop / resume — idempotent single command, lossless.** One CLI that **auto-detects**: a
**Resume Bundle** present in the run dir → load everything and continue from the saved
iteration; absent → fresh warm-start. `--restart` forces fresh. The Resume Bundle holds the
*entire* training state — all four policy nets, the critic, all optimizer (Adam) state, the
four KL-controller `coef`s, the iteration counter (so the deal-pool seed sequence and budget
accounting continue), the league, and RNG state — written **atomically** (`tmp` → fsync →
rename) **every iteration**, keeping the **last 2** for corruption safety. A Ctrl-C at any
moment loses ≤ 1 iteration and never corrupts. Serving-export Checkpoints continue on the
`snapshot_every` cadence.

**Stopping & ship — no auto-kill; companion eval.** ADR-0029's auto-kill budget is
**dropped** for this run (the owner's explicit "run for days, break the ceiling at least a
fraction"). The loop never self-terminates — it runs until stopped. The train loop stays
**lean** (cheap dials + snapshot export only); the heavy seat-swap **Tournament** lives in a
**separate `check` command** run on demand — keeping the days-long process small-footprint so
the heavy eval can never trigger the OOM that killed the ADR-0033 run at iter 163. **Ship
bar: seat-swap Tournament (full zoo: master + frozen-BC + snapshots) 95% CI excluding 0
(positive).** Move-Prediction-Eval is **reported and flagged, not gating** — a CI>0 candidate
with cratered Move-Pred is surfaced for an owner play-test (possible league-overfit /
non-transitivity), not auto-rejected. Ship candidates are tagged + preserved; nothing
auto-stops. Watched guards (in the dials, not hard gates): non-transitivity (beat frozen-BC
*and* master, not just recent snapshots) and bomb-spam over-correction.

**Memory.** Start **M=128**, `max_snapshots=3`, log peak memory/iter; raise M on a later
resume once headroom is confirmed (heavier than the M=256-OOM play-only run: 392-dim critic
+ four nets + four-net league snapshots). Survival-first, tune-via-resume.

## Consequences

- The extended collector and the Resume Bundle save/load are the first builds (test-first);
  the parity test against `play_full_round` is the collector's gate.
- `save_checkpoint` (model+optimizer+step) is insufficient for resume; the Resume Bundle is a
  superset written alongside it — the per-net serving-export Checkpoints are unchanged.
- A flat multi-day result is *not* a fresh refutation of co-training in general — only of this
  warm-started, KL-anchored, perfect-info-critic configuration; the per-net dials are what
  make a flat result diagnosable.
- Known un-mitigated risk: **rare-event frequency** for calls (a policy that rarely calls
  starves its call net) — consistent with ADR-0029's suspected ceiling; relying on entropy,
  not forced-call exploration, because the call-rate→EV curve was flat (ADR-0033).

## Build — test-first, validated on the real master (2026-06-07)

Built additively, every slice green before the next. **67 touched-area tests green.**

- **Collector** (`ppo/rollout.py`, extended): the `RolloutPolicy` protocol gained optional
  `act_schupfen_batch` + `act_call_batch(kind, decisions)`; new `SchupfenChoice` / `CallChoice`;
  `TrajectoryStep` gained a `decision_type` tag (`play|schupfen|tichu|grand`). Tichu is now
  solicited at first-non-Pass and grand at deal-time (`_solicit_grand`, batched per model),
  both routed to the learnable nets (recording learner seats) with a `should_call` fallback for
  play-only policies. **Parity-tested** against `play_full_round` with a caller present.
- **Co-training core** (`ppo/cotrain.py`, new): `sample_schupfen` / `schupfen_logp`
  (without-replacement 3-card action + its re-evaluator), `build_cotrain_batch` (GAE per whole
  seat-trajectory, then group by `decision_type`; the shared critic pools all steps),
  `cotrain_update` (shared-critic value loss + per-net clipped surrogate + KL-anchor-to-BC +
  entropy, per-net coefficients), `BatchedCoTrainPolicy` (the four nets + Perfect-Info Critic
  behind the collector interface), and `train_cotrain` (the loop with four adaptive-KL
  controllers + `start_iter` for resume).
- **Resume Bundle** (`ppo/cotrain_resume.py`, new): atomic `save_resume_bundle` /
  `load_resume_bundle` (all nets + critic + optimizer + KL coefs + iteration + league + RNG),
  `tmp -> fsync -> rename`, keep-2 (`.prev`).
- **Command** (`cli/train_cotrain.py`, new): idempotent start/resume (`--restart` forces
  fresh), per-net dials logged to `ppo_log.csv` every iter, Resume Bundle written atomically
  every iter, serving snapshots every `snapshot_every`. Configs `cotrain_v5.yaml` (the multi-day
  run, M=128, schupfen anchored tightest, perfect-info on, critic warm-up 20) and
  `cotrain_v5_smoke.yaml`. **Smoke ran clean on the real master** (warm-start of all four v5
  nets, full-stack rollout, per-net update with the 392-dim critic, bundle round-trip, and a
  re-run resumed losslessly).

- **Strength read** (`cli/check_cotrain.py`, new): exports the latest (or `--iter`) snapshot to
  TorchScript and runs the seat-swap **Tournament** vs `master` via `run_full_tournament`,
  printing the bootstrap CI; **ship bar = CI lower bound > 0** (process exit 0 iff ship).
  Separate from training (OOM safety). Configs carry an `eval:` block (master `.pt` paths + pool).
  **Validated end-to-end on the real master** (export → MLAgent load → tournament → CI verdict).

**Opponent league (built, opt-in, default off — 2026-06-07).** Pure self-play optimizes "beat my
current self," which can plateau or RPS-cycle; a league pins a frozen-BC (master-level) opponent +
rotating learner snapshots in the training distribution (direct pressure to beat master; ADR-0029's
non-transitivity guard). Built file-based (`ppo/cotrain_league.py` `CoTrainLeague`) because
co-training opponents run in spawned rollout workers: a never-evicted frozen-BC `base` weight file
plus up to `max_snapshots` learner snapshot files (oldest dropped), sampled one-per-iteration; the
worker loads the sampled file into a second net set (`ParallelRollout` opponent path). Round-trips
through the Resume Bundle (`league` field). Config `league: {enabled, snapshot_every, max_snapshots}`
— **parallel-only** (needs `rollout_workers > 1`); off = pure self-play. The plan: run pure self-play
first, and if `check_cotrain` reads flat, flip `enabled: true` and resume into the league.

**Still open (not blocking the run):** serving snapshots embed the shared optimizer (disk-heavy over
days) — trim to weights-only. Critic warm-up isn't parallelized (one-time). Move-Prediction-Eval
reporting in `check_cotrain` is stubbed but not wired (the CI>0 tournament is the decider per Q9).

### Performance — profiled, then GPU-update + process-parallel rollout (2026-06-07)

Profiling one iteration (`scripts/profile_cotrain.py`, M=512) showed it is **CPU-engine-bound**:
the rollout is ~70% of the iter and its self-time is dominated by the rules engine
enumerating legal actions (straights/pairs/bombs, card hashing) — pure Python, GIL-held;
the batched forward+backward update is ~30%. Two additive, opt-in levers were built test-first:

- **GPU the update only** (`cotrain_update(device=...)`, config `update_device: cuda`): moves the
  nets/anchors/critic/optimizer-state/batch to the GPU for the update and restores them to CPU in
  a `finally` — rollout, snapshots, and the Resume Bundle stay CPU. Measured **16× on the update**
  (22.5s→1.4s incl. transfer), ~25% off the iteration. The rollout's small per-tick forwards were
  left on CPU (not worth the transfer).
- **Process-parallel rollout** (`ppo/rollout_parallel.py`, config `rollout_workers: N`): the M games
  are independent and the bottleneck is GIL-held, so threads can't help — a persistent spawn pool
  fans chunks across worker processes (skeletons built once per worker, weights reloaded from a
  small file each iter, single-threaded per worker to avoid BLAS oversubscribe), each running the
  unchanged `collect_rollout` and returning whole trajectories (coarse IPC). `workers=1` keeps the
  validated single-process path. Self-play opponents share the learner's weights; the league hook
  (`opp_weights_path`) is reserved.

The two compose: GPU update + CPU-parallel rollout don't contend for the device. Critic warm-up is
not yet parallelized (one-time cost; lower `warmup_iters` or accept it).

## Heavy prior

This is the **~7th** self-play/RL attempt against a long string of flat results (BC at
decile-9, AWR, PPO-play-only at both lever ends, PIMC, search+learning, perfect-info-critic
play-only). It is run anyway as a deliberate, owner-directed all-in on the one structurally
untested lever, with the bar relaxed to *any* CI>0. If it lands flat, ACCEPT
([ADR-0033](0033-perfect-info-critic-escape.md)) becomes unconditional and the strength
program closes.

## Addendum (2026-06-07): Wish-declaration & Dragon-give — learn or leave frozen?

> **RESOLVED & IMPLEMENTED:** wire the WISH head, leave Dragon-give frozen. The WISH head
> is co-training and working (`cotrain_wish_v5`); see *Implementation status* at the end of
> this addendum. The analysis below is the original decision artifact, kept for the record.

The collector routes Play, Schupfen, and Calls to the nets; **Mahjong-wish-declaration**
and **Dragon-give** stay on the frozen seat agent (RuleAgent declines every wish; gives the
Dragon trick to the shorter-handed opponent). The BC `wish` (14-way: None + ranks 2–14) and
`dragon_assignment` (2-way) heads exist and are BC-trained, but are not sharpened in the loop.
A light engine-only probe (RuleAgent self-play, deterministic 1-ply oracle + fixed-heuristic
tournaments, 400-seed blocks × 2) quantified each lever:

**Wish-declaration — HIGH value, currently 100% unused.** Frequency ≈ 1.0 decisions/round.
RuleAgent *always declines*, so the entire lever is dead today. Even a **dumb fixed** wish
heuristic beats decline by **+9 to +13 team-relative pts/round** (WishHighestHeld +12.65 / +9.68,
WishAce +10.88 / +10.43, WishMostHeld +5.58 across two independent seed blocks; decline-vs-decline
control ≈ 0). A perfect 1-ply hindsight oracle beats decline in 72.7% of rounds (upside inflated
by hindsight, but confirms the decision is high-leverage). Crucially, the 14-way head **includes
the decline (None) action**, so a learned wish head **strictly dominates** the frozen
always-decline — worst case it relearns "decline", best case it captures a large untapped gain.
Strong synergy with the joint-sharpening thesis: the wish should be chosen *for* the wisher's
follow-up play, exactly what co-training the wish + play heads together optimizes.

**Dragon-give — LOW value, narrow.** Frequency ≈ 1.0/round (the Dragon wins a trick most rounds),
avg 23 pts at stake. The immediate team credit is **identical for either opponent** (both are on
the opposing team — `_team_of(target)` is the same), so the decision is first-order EV-neutral;
it matters *only* through the second-order last-in transfer rule (`round_points_by_player[last_in]`
moves to first-out's team). RuleAgent's "give to the shorter-handed opponent" is already a sound
proxy for "less likely to be last-in": it is suboptimal in only **9.5%** of decisions, and a
perfect 1-ply oracle gains just **~5.5 pts/round** (optimistic ceiling; a learned head captures a
fraction of that). The decision carries little learnable structure beyond "who will be last-in".

**Cost to wire** (mirror the Calls path, commit 2458759): a routing branch in `rollout.py` `_drive`
for the pending decision → `model.act_wish_batch` / `act_dragon_batch` (one-shot, like
`act_call_batch`), recording a `TrajectoryStep(decision_type="wish"|"dragon")`; `BatchedCoTrainPolicy`
gains the two `act_*_batch` methods; `models`/`bc_models`/`kl_controllers`/`ent_coefs` gain the two
nets; `_net_logits` gains a wish/dragon branch (both are plain masked categoricals — the
`_policy_kl_entropy` single-categorical path already handles them); the BC nets must be exported
into / loaded from the Resume Bundle as frozen anchors. `build_cotrain_batch` and the GAE/critic
integration already handle one-shot decision types generically (proven for calls), so no new
learning machinery is needed. Wish is ~the same effort as a Call head. Dragon adds the
winner-relative left/right side encoding (`dragon_intent_index(winner_seat)`).

**Recommendation: wire the WISH head; leave Dragon-give FROZEN.**
- Wish: large (~+10 pts/round naive headroom), strictly-dominant (head subsumes decline),
  high-synergy, Call-sized effort. Add it (suggest a *tight* KL-anchor-to-BC like Schupfen, since
  BC wish data is thin and the lever is sharp — let the policy gradient earn departures from BC).
- Dragon-give: ~5.5 pts/round oracle ceiling, first-order EV-neutral, RuleAgent already 90.5%
  optimal, mostly "predict last-in". Not worth a near-zero-gradient head competing for capacity and
  KL budget. Revisit only if a measured plateau traces to dragon-give specifically.

### Implementation status (updated 2026-06-07) — WISH head IMPLEMENTED and working

The recommendation above was signed off and **shipped**: the WISH head is wired and
co-training in the loop; **Dragon-give remains FROZEN** as recommended.

- **What was built (option (a)):** the Mahjong-wish is a **14-way head on the play net's
  trunk** (`0` = decline, `1..13` = ranks 2–14), sharpened in-loop alongside play/schupfen/calls
  and co-adapting with play through the shared trunk. Enabled via `cotrain_wish: true`.
  Code landed across PR #45 (foundation) and PR #48 (reconciled league vs the wish head);
  PR #49 added the `wi kl …` console line. All merged to `main`.
- **KL anchor:** tight anchor-to-BC as recommended — `kl.wish = {target: 0.008, factor: 2.0}`,
  matching Schupfen (sharp lever, thin BC data; policy gradient earns departures).
- **Live run:** `configs/cotrain_wish_v5.yaml` → `data/runs/cotrain_wish_v5` (fresh, not a
  `cotrain_v5` resume). Healthy: `wish_kl_coef` settled flat at 4.0 (1→2→4 early, then never
  crept toward the clamp — the head is anchored, **not pinned-to-decline**); `wish_entropy`
  steady ~1.56 (14-way max ≈ 2.64), so the head holds a real spread, not a collapse to decline.
- **Strength read:** vs `master` on the seat-swap Tournament, the run climbed then **plateaued
  by ~iter 800 around +7**: iter 200 `+5.60` → 800 `+7.57` → 1000 `+6.82` → 6225 `+7.13` (all
  n=8000, overlapping CIs). This **exceeds the shipped no-wish `cotrain_v5` (+6.29)** — the wish
  lever adds value. Decomposition (iter 1000): ~half the edge is call-bonus (+3.25), ~half
  play-only (+3.57); round win-rate ~50.6% (the edge is magnitude-driven, not frequency). The
  plateau is a stable, KL-constrained equilibrium (every coef flat; `wish_kl` parked at its 0.008
  target). **SHIP NUMBER LOCKED (2026-06-08): iter-6225 at n=40k = `+8.37` CI [+6.22, +10.45]**
  (seed 1). Higher than the 4k reads because those covered only the first 4000 pool deals (a low
  sample); the full 20000-deal estimate is the trustworthy one — comfortably above no-wish
  `cotrain_v5` (+6.29).
- **Wish-leash loosening — TESTED, NEGATIVE (2026-06-08).** Tested whether the +7 plateau was a
  wish-leash artifact: `configs/cotrain_wish_v6_loose.yaml` (= v5 but the wish KL target anneals
  **0.008→0.02 over [750,2750]**; play/schupfen/calls unchanged). The head fully used the slack
  (`wish_kl` 0.008→0.022, `wish_coef` 4→1). **It HURT**, not helped: at matched n=40k, v6-loose
  iter-3000 = `+5.03` CI [+2.71,+6.99] vs **v5-tight iter-6225 = `+8.37` CI [+6.22,+10.45]** — a
  ~3.3-pt drop, CIs barely touching (near-separated), with BOTH channels down (call +3.25→+2.57,
  play +3.57→+2.47) and win-rate → 50.0% (dead even). Cause = the under-fit critic on lever
  decisions (see memory `project_cotrain_learning_dynamics`): unreliable wish-value → looser leash
  lets the head wander into worse wishes, corrupting the shared trunk (so play+calls fall too).
  **The tight 0.008 anchor is correct; the +7 plateau is a value-signal-limited optimum, not a leash
  artifact. Don't re-run leash-loosening expecting gains** — the bottleneck is critic generalization
  on rare/lever states, not the KL targets.
- **Dragon-give:** left frozen on RuleAgent (shorter-handed opponent) per the recommendation;
  revisit only if a measured plateau traces to it specifically.
