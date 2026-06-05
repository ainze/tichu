---
status: proposed
---

# Phase-2 search: on-turn PIMC over frozen nets, as an offline strength experiment

Every offline lever (BC, AWR) and the on-turn online lever (**PPO Refine** v1,
[ADR-0029](0029-ppo-refine-design.md)) plateaued below the **Master-level goal** —
the diagnosis is that the strength gap is structural, not a tuning problem
([docs/notes/2026-06-03-ppo-refine-v1-conservative-result.md]). Search is the
remaining lever that structurally matches the goal, and the **Belief Model** was
built ([ADR-0021](0021-belief-trains-on-a-replay-derived-bundle.md)) to bridge to
it. This ADR records the v1 design locked in the grilling session of 2026-06-04.
The vocabulary (**PIMC**, **Determinized World**, **Search Agent**, **Leaf
Rollout**, and the now-resolved **Observation**) is glossed in
[CONTEXT.md](../../CONTEXT.md) §"Phase 2 / search terms".

The unifying principle across every decision below: **v1 isolates exactly one new
variable — look-ahead — and is the cheapest possible falsification of "can search
structurally beat `master` where learning could not?"** Anything that adds a second
variable, or couples the experiment to latency or to new training, is deferred.

## Decisions

**Deliverable — offline strength experiment, not a served tier.** v1's deliverable
is a **Tournament Matrix** row (+ owner play-test) showing a **Search Agent**
beating `master` with CI excluding 0. Latency, the 500ms serving budget,
`Difficulty`-tier integration, the serving codec, and TorchScript export are
**all descoped to Phase-2b**, contingent on a positive strength signal. This frees
the algorithm to optimize *strength per simulation* rather than *strength per
millisecond*. Symmetric with how PPO was killed on a clean strength bar before any
serving work.

**Search-only over frozen nets — no learning in v1.** The trained `master` Policy
Network (prior), critic / **Value Baseline** (escalation leaf value), Belief Model
(sampler), and rules engine (simulator) are all **frozen**. Nothing trains.
AlphaZero/MuZero-style self-play that *also* improves the nets is deferred to a
later phase, contingent on search-only showing signal — otherwise a win could not
be attributed to *search* vs *the extra training*. If unlimited search on frozen
nets cannot beat `master`, learning-from-search would not have saved it either, and
that is learned cheaply.

**Algorithm — PIMC, root-parallel.** From an **Observation**, sample K
**Determinized Worlds** from the Belief Model, run an independent PUCT-MCTS to
completion in each, **sum root visit counts across worlds**, pick the argmax
**Intent**. Chosen over single-tree ISMCTS: simpler, parallelizes onto the existing
process pool ([ADR-0026](0026-full-strength-tournament-runs-in-parallel-processes.md)),
and consumes the Belief Model exactly as designed. PIMC's strategy-fusion weakness
is the documented ISMCTS escalation if v1 plateaus specifically on
information-hiding/gathering symptoms. Per-simulation determinization (shared tree,
more sample-efficient) is the escalation if the per-world sim budget is the
bottleneck.

**Tree shape — single-rooted, others as policy-driven environment.** Inside each
Determinized World the tree branches **only on the root seat's own Decisions**; all
three other seats — both opponents **and** the cooperative partner — are simulated
by the frozen `master` policy as environment dynamics. Each world becomes a cheap
single-agent MDP, modelling the other seats as *exactly the policy faced in the
Tournament-vs-`master` eval*, and reusing the "others-are-environment" pattern from
the PPO rollout ([src/tichu_training/ppo/rollout.py]). A full multi-agent
(best-response) tree is rejected for v1: it optimizes against a *stronger* opponent
than `master` (paranoid, and not the actual objective) at much higher cost.
Approximation accepted: in a Search-vs-Search Tournament the partner is modelled as
`master` though it is itself a Search Agent — a *conservative* underestimate, since
a search-improved partner plays at least as well as `master`.

**Prior & leaf — `master` prior + Leaf Rollout to terminal.** PUCT prior = the
frozen `master` play-head distribution over the legal Intent mask. Leaf value = a
**Leaf Rollout**: continue the frozen-`master` policy-driven simulation of the
Determinized World to round-terminal (`_finalise_round`) and back up the actual
`round_outcome` (team-relative, **includes the ±100/±200 call bonus**). This trusts
only the engine and the `master` policy — the two validated assets — not the
off-distribution frozen critic, whose `value_loss` floored ~6000 on largely
irreducible `round_outcome` variance (ADR-0029) and was trained on a different
state distribution. Random rollouts are strictly dominated and rejected. The frozen
critic as a **depth-truncated bootstrap** is the documented variance-reduction
escalation if pure rollouts prove too slow/noisy — not the v1 default.

**On-turn only — out-of-turn bombs deferred.** v1 matches PPO's on-turn scope, so it
is a clean test of the one thing PPO lacked: look-ahead (e.g. holding a bomb for a
later high-value trick — a lookahead the soft policy cannot represent). Out-of-turn
**bomb interrupts** — search's *signature* structural advantage over PPO, and the
long-term reason search (not PPO) is the right tool — are the **first Phase-2b
escalation**, gated behind either on-turn search showing signal *or* an
interrupt-aware leaf evaluator. Rationale for deferral: the **Leaf Rollout** runs
everyone on frozen `master`, which never interrupts, so every interrupt valuation
would be backed up by rollouts that ignore all future interrupts — the rollout is at
its *weakest* exactly where interrupts matter most, and including them in v1 would
add a second new variable and a noisy signal. The engine already supports it
(`step` accepts a `BombInterrupt` validated by `legal_bomb_interrupts`); only the
search solicitation is new.

**Play only — calls/Schupfen delegated.** The Search Agent runs PIMC on the **Play**
Decision only; Grand-Tichu Call, Tichu Call, Schupfen, Wish, and Dragon Assignment
delegate to the existing Call Networks / Schupfen Network / BC heads, identically
for the root seat and the in-tree environment. The diagnosed pathology is bomb
undervaluation in Play; calls/Schupfen are "fine" (same scope as ADR-0029). The call
bonus reaches the search for free through `round_outcome`, so the caller's *play* is
optimized to fulfill the call without searching the high-variance ±200 call node.

**Sampler — belief-conditioned, with a belief-off ablation.** The Belief Model emits
per-card *marginals*; a Determinized World needs a *joint* assignment (each unseen
card → one opponent, each opponent → its known `hand_size`). v1 sampler: a
**sequential, capacity-respecting** sampler over the belief marginals, honoring the
sparse hard constraints (hand sizes, seen-card mask), validated against held-out
true worlds. Belief fidelity does **not** gate v1: the belief history-gate note
([docs/notes/2026-06-03-belief-history-gate.md]) found the Belief Model
**near-ceiling, not scale-limited** — only ~3–4 pp above a hand-size predictor,
because residual opponent-hand uncertainty in Tichu is largely irreducible. That
note deferred the *only* decisive test of belief to the extrinsic search setting —
which PIMC now is — so v1 runs **belief-on (primary) vs belief-off (constraint-only)
as a first-class ablation**. If PIMC needs better belief, that is a *finding*, not a
prerequisite.

**Eval — staged, slowest-agent-aware; sim-budget is the kill unit.** A Search Agent
is orders of magnitude slower than an ML Agent, so eval is staged cheapest-and-most-
diagnostic first:
- **Stage 0** — behavioral dials on a tiny pool (`eval_matrix --mode behavioral`):
  `caller_bomb_passivity_rate` (baseline **0.323**) must fall hard (leading
  indicator; flat → stop), `bomb_when_legal_rate` bounded (over-correction guard).
- **Stage 1** — small-pool Tournament vs `master` (+ `bc_neutral` anchor) at a fixed
  documented sim budget; directional go/no-go on CI.
- **Stage 2** — larger-pool full-zoo Tournament + belief-on/off ablation; non-
  transitivity guard (beat `master` *and* frozen BC, not cycle).
- **Stage 3** — owner play-test on the best config (the only exceeds-human signal).

Compute budget unit is the **sim budget (K worlds × N sims/world)**, swept low→high
to map strength-vs-compute and locate the kill point. Pre-committed kill: **if the
Stage-0 dials don't move, or the small-pool Tournament-vs-`master` CI fails to
exclude 0 at the max feasible sim budget, stop and diagnose** (sampler? leaf?
strategy fusion?) rather than scaling pool/sims indefinitely. The absolute sim
number is set after the build's first throughput read (positions/hour), exactly as
ADR-0029 set its iteration budget.

## Consequences

- First build is the **vectorized PIMC harness** over the engine — the throughput
  gate. A throwaway runnable prototype (the `prototype` skill) sanity-checks
  feasibility/latency and supplies the positions/hour figure that sets the sim
  budget, before the full build.
- The **determinization sampler** (marginal → constraint-respecting joint world) is
  a net-new sub-build with its own correctness risk; validate against held-out true
  worlds before trusting it.
- v1 explicitly excludes: serving / `Difficulty`-tier integration & the 500ms
  budget, search+learning (self-play training), out-of-turn bomb interrupts,
  searched calls/Schupfen, and the full multi-agent best-response tree — each a
  named Phase-2b/2c escalation, not a discarded idea.
- The reserved **Observation** term is now resolved to the PIMC search root
  (PrivateState + belief distribution); CONTEXT.md updated.

## Prototype throughput read (2026-06-04)

A throwaway single-process spike ran the full on-turn single-rooted PIMC loop
(real engine + real Determinization Sampler, **random-legal policy stub** — no
torch) to profile the non-batchable residue. Script:
[scripts/spike_pimc_throughput.py](../../scripts/spike_pimc_throughput.py)
(deletable; the value is this verdict). Measured on the in-harness machine:

- **Loop is correct end-to-end** — determinize → PUCT → Leaf Rollout → root-parallel
  visit-sum produces a sensible most-visited move with propagated terminal values.
- **Engine `step` is ~100% of wall time.** At 6 worlds × 30 sims over a full round
  (3,060 sims), ~99k engine steps consumed essentially all 6.2s; the sampler and
  tree bookkeeping were negligible. Throughput ≈ **10–16k steps/sec** (≈60–105 µs/step,
  enumeration-bound — each rollout step pays `legal_actions` *twice*: once to choose,
  once inside `step`'s re-validation), and a rollout from the opening is ~98 steps.
- **Headline cost:** ~363 ms per searched decision at 180 sims; **linear in total
  sims** (each sim ≈ one rollout). At a useful ~2,000 sims/decision that is ~4 s/decision,
  ~17 decisions/round → a seat-swapped Tournament position (~2 rounds × 2 searching
  seats) is on the order of **minutes**; a 200-position pool is hours single-process,
  → ~an hour or two across the ADR-0026 process pool. Workable for the **offline
  experiment**, hopeless for the deferred 500 ms serving tier (confirms Phase-2b
  descope).

**Decisions this read forces:**
1. **The Leaf Rollout is the bottleneck, not the search.** The critic-bootstrap
   leaf eval (frozen Value Baseline, depth-truncated rollout) moves from "documented
   variance-reduction escalation" to **likely necessary for a usable sim budget** —
   one critic forward replaces ~30–100 engine steps, a ~1–2 order-of-magnitude
   throughput win. Build the rollout leaf first (it needs no critic and validates the
   loop), but plan the critic-bootstrap as the immediate throughput follow-up, and
   read the sim-budget kill-criterion against *critic-bootstrap* throughput.
2. **Cheap engine win:** `legal_actions` is computed twice per step (choose +
   `step` re-validation). Threading the already-enumerated legal set into `step`
   roughly halves step cost — a low-risk optimization for the real vectorized harness.
3. **The stub caveat:** these are single-process, stub-policy numbers. The real build
   *batches the `master` forward across the M concurrent worlds/games* (ADR-0029
   rollout shape), which this single-threaded spike does **not** — so the per-game
   Python residue measured here is the correct thing to attack (rollout depth), and
   batching is an orthogonal further win.

## Phase A build — pipeline tracer (2026-06-04)

Built the Tournament-capable Search Agent and validated the integration end-to-end.
Code (all TDD'd): `src/tichu_training/search/{mcts,engine_world,agent}.py` (MCTS core
+ PIMC aggregation; the Tichu `world` wiring; `SearchAgent(Agent)` registered as
`"search"`), a one-line side-effect import in `cli/eval_matrix.py` so spawn workers
resolve the factory, and `configs/eval_tournament_search_tracer.yaml`. The leaf is a
pluggable seam shipping `fast_rollout_leaf` (random playout); the master `MLAgent`
supplies the PUCT prior (`play_action_scores`), the in-tree env moves (`act`), and
the delegated calls/pending decisions.

**Latent engine bug found + fixed (benefits all evals).** The tracer surfaced a
slam (doppelsieg) bug in `engine.step`: it finalised the round, then re-evaluated
`done` on the *finalised* state — but `_finalise_round` resets `out_order`, so a slam
(partner pair out, fewer than 3 hands empty) re-checked as not-done and reported
`done=False`. `play_full_round` then kept stepping the finalised state, double-
finalising and corrupting the score for any round containing a slam — silently, for
*every* past full-strength Tournament (the sampler's consistency guard is what
exposed it; `MLAgent` tolerated the junk state). Fixed across all five `step` return
branches (capture `done` before finalise); regression test added; 257 engine tests
green. Past Tournament numbers involving slams are mildly suspect; quantifying is
separate work.

**Tracer result (pipeline check, NOT a strength verdict):** 4 positions, fast-rollout
leaf, 3 worlds × 12 sims. `master` beats `search` +120, CI [+39.9, +201.3], n=8. The
config is deliberately under-budgeted — 12 sims vs a ~41-action branching factor, so
the root never finishes expanding and move choice is near-random; a random-rollout
leaf adds noise. Conclusion: **the pipeline is correct (Search Agent plugs into
`eval_matrix`, plays clean full rounds vs master, emits a Tournament Matrix row); the
fast-rollout/under-budget config is weaker than master, as expected.** The strength
verdict is Phase B: critic-bootstrap leaf (fit + persist + load a Value Baseline —
no artifact exists yet) plus a sim budget above the branching factor, then scale to
the Stage-1 pool.

## Phase B build — critic-bootstrap leaf (2026-06-04)

Built the critic leaf and the missing Value Baseline. Code (TDD'd): `search/critic.py`
(`CriticValue` leaf + `save_critic`/`load_critic`), `SearchAgent(critic_path=...)`
wiring (pluggable seam — critic XOR fast-rollout), and `scripts/fit_search_critic.py`
(master self-play → `(featurize(view), round_outcome/100)` at every Play Decision →
`fit_value_baseline` → persist). Critic target chosen (user): **master self-play
round_outcome** (best distribution match for the leaves PIMC evaluates).

**Critic fit:** 33,861 samples from 600 master self-play rounds, dim 224, final
MSE 4.25 against target variance ~4.79 → **R² ≈ 0.11**. Weak but non-trivial, exactly
as ADR-0029 predicted (round_outcome is largely irreducible variance) — and still a
far lower-variance leaf estimate than a single random rollout (variance ~4.79/sample),
which is what matters for MCTS.

**Result (still tracer-scale — 8 positions, n=16):** critic leaf, 6 worlds × 60 sims,
`master` vs `search` = **+49.4, CI [−52.0, +151.0]** — CI now spans 0. Trajectory:
random-leaf/12-sims gave `master +120` (CI [+40, +201], search clearly worse) →
critic-leaf/60-sims gives `master +49` (CI spans 0, **statistically tied**). The leaf
evaluator was the lever, as the prototype predicted; search is now competitive with
master. **Not yet a win** — the point estimate still favors master (~−49, of which
the call-bonus delta is ~−37.5, high-variance at n=16) and the pool is tiny.

**Next levers (not yet run), in cheap-to-expensive order:** (1) push the point estimate
positive on the tiny pool via more sims and/or a stronger critic (more self-play data,
belief-on sampler) — a cheap directional check, but the CI stays wide at n=16; (2) once
the point estimate is clearly ≥ 0, scale to the Stage-1 few-hundred pool for a tight CI
and the real go/no-go vs master + `bc_neutral`; (3) behavioral dials
(`caller_bomb_passivity`) on the search agent to confirm it actually deploys the bombs
PPO could not. Critic R²≈0.11 is the suspected ceiling; if search plateaus tied, the
critic (or belief-on) is the thing to improve, not the search.

## Phase B Stage-0 — behavioral dials (2026-06-04)

Ran the leading-indicator dials (`eval_matrix --mode behavioral`, self-play) on the
critic-leaf search vs master: 12 rounds, 48 seat-rounds each, search at 3 worlds × 30
sims. Read on **passivity as a whole**, not just the bomb sliver:
- **`caller_passivity_rate`** (a caller ceding *any* legal beat — the reliable metric,
  n≈36/46): search **0.130** vs master **0.139** — essentially identical.
- `bomb_when_legal_rate` (n≈39/46): search **0.051** vs master 0.087; `bomb_per_round`
  search **0.042** vs master 0.083 — search bombs *less*, not more.
- `caller_bomb_passivity_rate` (bomb subset): search 0.000 vs master 0.200 — **but only
  2 vs 5 opportunities, too sparse to read**; do not anchor on it.

**Controlled three-way (identical 90 sims), broad `caller_passivity_rate`:** master
0.139 (n=36), critic-leaf search **0.130** (n=46), random-rollout-leaf search **0.1875**
(n=64). The random-rollout leaf is the *most* passive of the three; the critic leaf is
the *least* passive (marginally below master).

**Finding (corrected — an earlier draft had it backwards): the critic is the right
leaf, not the aggression-suppressor.** A hypothesis that the master-self-play critic
"inherited master's bomb-blindness and amplifies passivity" was **refuted** by the
control — random rollouts are worse, the critic is best. The honest read at this
budget: **search with any leaf reproduces ≈ master-level passivity** (critic 0.130 ≈
master 0.139, within noise; rollout worse), so the fix has *not yet* reduced passivity
— but *not* because of a critic value-bias. Caveats remain: modest n, and this is a
*reduced* 90-sim budget vs 360 in the strength run, so the live question is whether the
search is simply **too shallow** to find the delayed-payoff aggressive lines (hold a
strong combo for a better trick), not which leaf is used.

**Implication for the design.** The leaf-value choice is settled (critic > rollout);
the open lever was **search budget/depth** vs the value's *ceiling*. The deep-budget
test below resolves it.

### Deep-budget test — the decisive result (2026-06-04)

Ran the behavioral dials at the full **360-sim** strength budget over **40 rounds**
(parallelised as 4 disjoint Pool slices via a new `eval_matrix --n-offset`; counts
pooled), with master profiled on the *identical* 40 positions to kill the
position-slice confound. On the reliable broad metric (n≈158 opportunities):

| positions 0–39, 360 sims | `caller_passivity_rate` | `caller_bomb_passivity` | `bomb_when_legal_rate` |
|---|---|---|---|
| master | 0.183 | 0.222 | 0.118 |
| **deep critic-search** | **0.323** | **0.471** | 0.103 |

**Deeper search is ~1.8× MORE passive than master, not less** — robust (same positions,
solid n). This is the **frozen-critic ceiling, confirmed and worse than a plateau:** the
critic is regressed from master self-play, where the passive policy rarely contests
tricks, so the critic *prefers ceding* in caller situations; **faithful (deep) search
amplifies that preference.** It coheres with the strength point estimate (search −49 vs
master, i.e. slightly behind). You cannot search your way out of a value that encodes
the soft policy's passivity.

**Conclusion: search-only on frozen nets does not fix the diagnosed pathology — at
depth it mildly worsens it.** The PIMC machinery is sound (pipeline correct, critic >
rollout, competitive-ish in raw strength), but the frozen master-derived value is the
wall. The indicated path is the escalation ADR-0030 reserved — **search + learning**
(AlphaZero/MuZero-style): the value is *discovered* by search and the policy/value are
retrained on search-improved targets, the only mechanism that bootstraps past the
frozen soft-policy value. The Phase-2 effort thus pivots from "search-only on frozen
nets" (now empirically falsified as a stand-alone fix) to search-as-target-generator
for learning. The built PIMC stack (sampler, MCTS, EngineWorld, SearchAgent, critic) is
exactly the engine that escalation reuses.

### Exploration-critic pilot — setup (2026-06-04)

Before committing to the multi-week search+learning loop, a cheap one-shot test of its
load-bearing premise: *is the value the fixable lever?* The deep run showed the wall is a
value regressed from master self-play — the passive policy rarely contests tricks, so the
critic never sees aggressive caller-states and learns to prefer ceding. The pilot refits
the critic on a deliberately **aggression-exploring** distribution and asks whether that
alone moves deep search off the ceiling — **no new search code**, just a different
`critic_path` into the existing `SearchAgent`.

- **Mechanism** — `ExplorationAgent` (`src/tichu_training/search/exploration.py`) wraps the
  master and perturbs *only* Play Decisions with **uniform ε-greedy (ε=0.25) over the full
  legal set**; pending wish/dragon/schupfen and the Tichu/Grand calls stay master. Uniform
  (not policy-weighted) is deliberate — it surfaces beating plays *and* bombs at their
  structural frequency, covering **passivity as a whole**, not just the bomb subset. The
  call/leadership structure stays realistic, so explored play-states get near-on-policy
  continuations and thus *true* `round_outcome` labels for aggressive choices the master
  suppresses. Wired as `fit_search_critic.py --explore-epsilon` (ε=0 reproduces the
  baseline critic exactly); critic → `data/export/search_critic_explore_v5/critic.bin`.
- **Test** — re-run the *identical* deep dial (`configs/eval_behavioral_search_explore.yaml`,
  360 sims, same 40 positions, 4 `--n-offset` slices) and compare `caller_passivity_rate`.
- **Falsification criterion** — the master-critic deep run sits at **0.323** vs master
  **0.183**. The pilot **confirms** the value-is-the-lever hypothesis only if the
  exploration critic pulls deep search *below master* (≈0.18 or lower); a value that merely
  lands between 0.18 and 0.32 is a partial signal (the loop may need policy retraining too);
  a value that stays ≥0.32 **refutes** the cheap fix and says the passivity is not curable
  by reweighting the value's training data alone — the full search+learning loop (with
  iterated policy improvement, not just value) is then required.

### Exploration-critic pilot — result (2026-06-04)

Critic fit on ε=0.25 self-play (40,556 samples, MSE 4.43 — higher than the master-critic
baseline, as expected: ε-greedy continuations raise outcome variance). Re-ran the identical
deep dial (360 sims, same 40 positions, 4 pooled slices). Broad metric n≈146 opportunities:

| 40 positions, 360 sims | `caller_passivity_rate` (broad) | bomb subset (n=16) |
|---|---|---|
| master | 0.183 | 0.222 |
| master-critic deep search | 0.323 | 0.471 |
| **exploration-critic deep search** | **0.247** | 0.375 |

**Partial / weak-positive — lands in the pre-registered middle band.** The exploration critic
pulled deep search **down** from the master-critic's 0.323 toward master (0.247), i.e. the
value moved the *right* direction — but it did **not** clear master's 0.183. And both deltas
are modest: 0.247 vs 0.323 is **~1.4σ** (unpaired, n≈146), 0.247 vs 0.183 is **~1.3σ**.
Directional, **not decisive** at n=40 rounds. The bomb subset (0.375) is uninformative —
n=16, only 2 of 4 slices had any bomb opportunity.

**Read:** the load-bearing premise of search+learning is *supported, not proven* — a value
**does** respond to its training distribution, and refitting on aggression-exploring data
demonstrably reduces the passivity that the frozen master-self-play value amplified. But a
**one-shot value swap is not sufficient** to beat master: the prior/policy is still the
passive master, and PUCT exploration is steered by that prior, so the residual passivity
plausibly lives in the policy, not only the value. This is exactly the "loop may need policy
retraining too" branch. **Conclusion:** the cheap value-only fix is partially effective but
falls short of master — which (a) de-risks the full **search + learning** loop (value is a
real lever) while (b) confirming it must co-train the **policy**, not just the value. Two
cheap follow-ups could sharpen the verdict before the multi-week build: more rounds (n=40 is
underpowered for a ~0.07 effect — 80–120 rounds would move the ~1.4σ toward significance),
and/or a stronger exploration distribution (higher ε or forced-beat-when-legal) to test
whether a less-passive value *can* clear master, pinning down the value ceiling.

#### Firming the result — 120 rounds (2026-06-04)

Ran the cheap follow-up (1): extended the exploration-critic dial to **120 rounds** (8 more
`--n-offset` slices, pooled with the original 4 → positions 0–119) and re-profiled **master
on the identical 120 positions** (fast, no search) so the comparison is paired at 3× the n.

| 120 positions, 360 sims | `caller_passivity_rate` (broad) | n (opportunities) |
|---|---|---|
| master | **0.162** | 364 |
| exploration-critic deep search | **0.263** | 441 |

**The weak-positive evaporated — the verdict flips and is now decisive.** At n=40 the
exploration critic looked like 0.247 (a hair toward master); the extra 80 rounds came in
≈0.27 and master tightened *down* to 0.162, so the pooled gap is **+0.101 at ~3.5σ**:
exploration-critic deep search is **decisively MORE passive than master**, not less. The
n=40 "moved the right direction" read was mostly noise — the power check earned its keep.

**Conclusion (firmed): the cheap value-only fix is refuted as a way to beat master.**
Refitting the critic on aggression-exploring data plausibly still beats the *master-self-play*
critic (0.263 vs that run's 0.323 — but that anchor is only n=40, so treat as suggestive, not
established), yet it does **not** get deep search below the master policy itself. Reweighting
the value's training distribution alone does not remove the passivity. The residual sits in
the **policy/prior** — which steers PUCT's expansion and the in-tree environment moves — so a
value swap cannot reach it. This decisively selects the "loop must co-train the policy"
branch: the indicated path is the full **search + learning** loop (iterated policy *and*
value improvement on search targets), not any one-shot value intervention. A higher-ε /
forced-beat critic might shave the gap, but the n=120 result makes it implausible that a
value-only change clears master — that follow-up is now low priority versus building the loop.
