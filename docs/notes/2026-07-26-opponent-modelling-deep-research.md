# 2026-07-26 — Deep research: can opponent modelling / belief break the cpfix3328 plateau?

## TL;DR

**No — not in any of the three forms the literature would normally recommend, because this
repo has already built and falsified all three.** The external evidence independently
explains *why* each came back flat, and adds a structural argument that trick-taking /
climbing games have intrinsically low belief headroom.

The one thing genuinely missing is a **measurement**: every belief result we own is in
*prediction-accuracy* units, and the literature repeatedly shows accuracy does not convert
to strength. The recommended path is therefore a cheap **Perfect-Information Policy Ceiling**
gate that converts the question into EV points, followed — on the likely RED outcome — by a
pivot to the **partner-coordination axis (JPS-family joint policy improvement)**, which the
evidence identifies as the structurally untested failure mode for a 2v2 self-play agent and
which has the single best-matched precedent (bridge bidding, 4-player 2v2).

## Provenance caveat — read this before citing anything below

The research workflow was **truncated by a session token limit** during adversarial
verification. Of 108 agents, 54 died. Concretely:

- **Scope / Search / Fetch phases completed**: 6 angles, 25 primary sources, 124 extracted
  claims with verbatim quotes. This is the evidence base used below.
- **Verification largely died**: of 25 claims sent to 3-vote adversarial verification,
  **4 confirmed (3-0 or 2-0), 4 refuted, 17 unverified (verifiers errored)**.
- **Synthesis never ran.** This document is the synthesis, written by hand from the
  recovered journal.

Claims are tagged below: `[verified]` = survived 3-vote adversarial check, `[refuted]` = do
not rely on, `[unverified]` = extracted with a supporting verbatim quote but not adversarially
checked. Most claims are `[unverified]` — they carry direct quotes from primary sources, but
the quote-to-claim inference has not been independently attacked. Treat effect sizes as
indicative, not settled.

---

## 1. Three branches already closed in this repo

| Branch | Local status | External corroboration |
|---|---|---|
| **Explicit belief model** (hidden-hand predictor) | Built (ADR-0021/0028), trained at scale, **near-ceiling**: 10× data + 2× capacity bought **+0.5pp top-1**; tops out 3–4pp over a hand-size floor the policy already computes ([belief gate note](2026-06-03-belief-history-gate.md)) | Skat: higher inference accuracy scored *worse* in null games; DeepStack reimpl. had *lower* CFVnet validation error than the original and still lost head-to-head |
| **Privileged / oracle critic** (PerfectDou PTIE) | Built + A/B'd (ADR-0033): **−3.7 vs master, indistinguishable from the symmetric critic** | PerfectDou's own ablation: ImperfectDou (imperfect critic) already beats DouZero at equal steps (WP 0.717 vs 0.732) — most of the gain is feature design + PPO infra, not privilege. Benefit framed as **sample efficiency**, not asymptotic strength |
| **Belief-driven search** (PIMC, vine, pMCPA, piKL) | Killed four times (ADR-0035/0036/0037/0040) | ReBeL: guarantees only 2p zero-sum, input grows linearly in infostates per public state `[verified]`. Skat: a **cheating** sampler placing all mass on the true world played **worse** than ordinary inference (−3.25 / −8.49 TP/G) |

### The unifying external insight

**Every privileged-information result in the literature is a sample-efficiency result, and
you are not sample-limited.** PerfectDou reports ~10× fewer samples for equal strength.
Suphx's oracle guiding is an annealing schedule that speeds convergence. `[unverified]`
OracleDou (IJCAI 2024) is the cleanest statement: oracle guiding alone **did not reach
DouZero parity** after 30 days — it only beat the authors' *compute-matched* replication, and
the authors explicitly claim a *learning-efficiency* gain `[verified]`.

That is exactly the shape of ADR-0033's flat result. A perfect-info critic accelerates a run
that is still climbing; it does nothing for a run that has plateaued at its asymptote. The
ADR-0033 outcome was not an anomaly — it is the predicted outcome.

> Note: two related OADMCDou claims (that the λ=1 perfect-info ablation also failed to beat
> DouZero, and that the variance-control component rather than the oracle component was the
> stronger contributor) were **`[refuted]` 0-3** by verifiers. Not relied on above.

---

## 2. The structural argument: trick-taking games have low belief headroom

This is the deepest finding in the corpus and it was not previously represented in the ADRs.

Long, Sturtevant, Buro & Bowling's PIMC analysis measures three tree properties — *leaf
correlation*, *bias*, and *disambiguation factor* — and shows they predict how much a
perfect-information sampler loses against equilibrium play. `[unverified]`

- Measured on 10,000 human Skat games and 3,000 Hearts games: **leaf correlation 0.8–1.0,
  disambiguation factor ~0.6**. In that parameter region a sampler that *ignores information
  sets entirely* loses only **~0.1 points/game** versus Nash.
- Skat endgames: an exact CFR solver beats PIMC by 0.42 TP/deal on the 15% of deals still
  unresolved with three tricks left = **0.063 TP/deal average**, against an empirical per-deal
  TP standard deviation of **778**.
- Bridge: Ginsberg's explicit imperfect-information enhancement to GIB improved declarer play
  by **+0.1 IMPs/deal**.
- Poker is the structural *opposite* (Kuhn poker: disambiguation factor 0) — "no private
  information is directly revealed until the end." That is precisely why belief machinery
  pays in poker and not in cardplay.

**The mechanism:** in a climbing/shedding game every play reveals a card, so information sets
collapse rapidly. The residual hidden-hand uncertainty that a belief model could resolve is
small, and shrinking exactly as the decisions get sharp.

This independently predicts our own gate result — belief beat the hand-size floor *least*
late-round (0.674 vs 0.646) — and it falsifies the assumption that poker-family belief
methods (ReBeL, DeepStack, SoG, CAPI) transfer their measured value to Tichu.

---

## 3. Where belief *did* pay in a trick-taking game — and why it's unavailable here

The strongest positive result in the corpus is Solinas, Rebstock & Buro (AAAI 2019), Skat:
a card-location inference network trained on **20M human games**, used **only** to reweight
PIMC world sampling, gave **+4.0 to +4.8 tournament points/game**. `[unverified]` The ablation
is sharp: conditioning on *cardplay move history* was load-bearing (BDCI beat history-free BDI
by 2.6–6.3 pts/game), and inference quality peaked mid-game, tricks 3–5.

Three reasons this does not port:

1. **The mechanism is world sampling for search.** Search is dead here, four times over.
2. **The evidence channel is already in our featurizer.** v6 carries `declined_top`,
   `lead_summary`, `pass_pressure` (ADR-0028 B-core) *and* `played_by` (4×56 per-player card
   provenance) — the move-history attribution that made BDCI work is already a policy input.
3. **The same lab's follow-up shows belief+search can regress.** A "cheating" inference variant
   placing all probability mass on the *true* world played **worse** than ordinary inference
   (−3.25 TP/G suit, −8.49 TP/G grand). Perfect belief inside a determinized searcher is a
   strength *regression*.

Related: IRumAI (Rummy, 2026) reports that **linear probing recovers the opponent's hidden hand
from a network with no explicit belief module** — trained by exactly our recipe, BC warm-start
plus PPO. `[unverified]` If that transfers, cpfix3328 probably already carries an implicit
opponent model, and an explicit one is re-packaging information the trunk has.

---

## 4. Auxiliary belief heads: mixed, and negative in the closest analogue

This is the one form never tried here, so it deserves a fair hearing. The evidence does not
support it.

**Against:**
- **Hanabi / SAD** — the closest structural analogue (multi-agent, several hidden-information
  sources): the auxiliary hidden-card head helped 2-player (24.02 vs 23.87) but **degraded
  3-, 4-, and 5-player** play (5p: 22.06 → 21.47). `[unverified]` Tichu has three hidden hands.
- SAD's authors are explicit that the head was "optionally" added as a shaping device, not a
  validated strength lever — and that SAD's actual gain came from a *different* training-time
  trick: letting each agent observe teammates' **greedy** action during centralized training.
- Feeding an explicit belief distribution as *policy input* is called out as architecturally
  self-defeating in multi-agent settings — it induces recursive higher-order beliefs.
- DouZero+ (explicit hidden-card predictor concatenated onto state features) finished **below**
  plain DouZero in OADMCDou's head-to-head (−0.226 ± 0.031 ADP), though confounded by compute.

**For:**
- BEPAL reports ~16% task-metric improvement from auxiliary hidden-state heads — but only on
  predator-prey and Google Research Football, with no card-game benchmark and no ablation
  separating prediction accuracy from strength.
- Belief-I2Q helped on 3 of 4 cooperative benchmarks, but the authors delimit applicability:
  it degrades when inference depends on *other agents' learned policies*, and it **slows
  convergence**.
- DouZero+ warns of genuine **early negative transfer** — belief-augmented nets are worse
  early because the input is larger, and only overtake later. A short A/B would false-negative.

**Verdict:** cheap to build here ([BCModel](../../src/tichu_training/bc/heads.py) is a shared
trunk with pluggable heads; belief labels are already materialised and aligned to play
decisions), but low expected value. Since the featurizer already carries the evidence, an
auxiliary head can only help via *representation shaping* — and our diagnosed bottleneck
(advantage-SNR probe: 74% epistemic critic residual, call-hinge states missed by ±300) is a
value-function problem, not a representation-poverty problem.

---

## 5. We already own both bounds, and the slice between them is thin

- **Upper bound** is unmeasured in EV units — but bounded by "what perfect hidden-hand
  knowledge is worth."
- **Lower bound** is measured: our belief model recovers only **+3–4pp top-1 over a hand-size
  floor the policy computes for free**, and that is *scale-invariant*.

So even if perfect information is worth a lot, the achievable slice is the fraction a lossy
belief estimate can recover — which our own gate says is thin. The missing step is that our
gate measured **accuracy**, and the literature is emphatic that accuracy does not convert:
Skat's PI100 had higher true-state-sampling ratio than PI20 and scored *worse*; DeepStack's
reimplementation had lower CFVnet validation error than the original and still lost 63 ± 40
mbb/g to a non-searching lookup table. `[unverified]`

---

## 6. Recommended path

### Step 1 (gate, decisive) — Perfect-Information **Policy** Ceiling in EV units

ADR-0033 measured a perfect-info **critic**. Nobody has measured a perfect-info **policy**.
That is the number that caps the entire opponent-modelling program: belief can only ever
approach a cheating agent, never exceed it.

**Two design constraints, both found the hard way while costing this gate out:**

1. **It must be RL / co-training, not BC.** An oracle-*BC* arm is structurally invalid and
   would come back falsely flat: human players never saw the hidden hands, so their actions
   are not conditioned on the perfect-info features, and a BC model given the extra 168 dims
   learns to *ignore* them. A flat result would measure "humans didn't cheat," not "perfect
   information is worthless." This is how the literature ran it — DouZero+'s "maphack"
   pre-experiment and OADMCDou's λ=1 arm both append true hands to the state features of an
   **RL** agent.
2. **The belief bundle does not exist on disk.** `data/parquet_full_v6{,_wishfix}/` carry
   play / wish / dragon / schupfen / calls — **no belief shard** — and there is no
   `materialised_*` directory. ADR-0028 said as much at the time; the 3k/20k-game gate bundles
   were temporary. Labels are gated behind `--bundle-tasks belief` and are cheap to emit
   *during* a parse pass, but obtaining them now costs a full corpus re-parse. The gate below
   does not need them.

- **Build:** cotrain warm-started from `cpfix3328` with the **actor** consuming
  `featurize_perfect_info` (392) — the 224 observable features plus the (3,56) ground-truth
  hand grid. [perfect_info.py](../../src/tichu_training/perfect_info.py) already provides it,
  and `BatchedPolicy(perfect_info=True)` already builds the 392-dim vector during rollout and
  threads `GameState` into `act_play_batch` — it just routes it to the *critic*. The new work
  is feeding it to the actor and re-forwarding the policy at 392 dims (keep the field additive,
  as ADR-0033 did for `critic_features`).
- **Control arm is not optional:** an identical cotrain on observable 224, same seeds/steps,
  to separate "perfect info helps" from "more cotrain from cpfix3328 helps at all."
- **Measure:** seat-swap tournament, both arms vs `cpfix3328`, n ≥ 4000.
  Δ_oracle = the oracle arm's advantage over the control arm — the EV value of perfect
  hidden-hand knowledge to the *policy*. The oracle arm is a cheating agent and is never
  shippable; it is a measurement instrument only.
- **Free warm-start:** `data/tmp/perfect_info_ceiling.npz` holds 1,133,887 rows of
  `obs(224)` / `pi(392)` / `z` / `rounds` / `call_live` from ADR-0033's gate. No actions, so it
  cannot train a policy — but it warm-starts the critic at no cost.
- **Pre-commit bands** (calibrated to our tournament scale — CI widths ~±5–7):
  - **Δ_oracle < 10 pts/round → KILL the whole program.** Perfect information itself is
    below what we can even measure; a lossy belief estimate cannot produce a CI-excluding-0
    ship. This is the outcome the disambiguation-factor argument predicts.
  - **Δ_oracle > 30 → real headroom exists**, and the question becomes extraction, not
    existence. Only then is a belief build justified.
  - **10–30 → amber:** break out by round-phase and call-live states; if the value is
    concentrated in call-hinge states (where the advantage-SNR probe already localised the
    critic's epistemic miss), a *targeted* belief signal on that stratum is the narrow bet.

This mirrors the project's proven MO — falsify the premise cheaply before paying for the
build — and it is the one instrument that converts belief into the units that matter.

DouZero+ ran precisely this pre-experiment ("maphack": append the next player's true hand to
the state features) and reports it *did* boost strength, most strongly for the cooperating
Peasant side. `[unverified]` That role-asymmetry is suggestive for a 2v2 game and is worth
breaking out in the gate, alongside the round-phase and `is_call_live` splits.

**Pre-commit the horizon before starting.** DouZero+ documents genuine early negative transfer
— a belief/oracle-augmented net is *worse* early because the input is larger and learning is
slower, and only overtakes later. A short A/B window will false-negative this gate. Do not
reuse ADR-0033's offline-R² instrument either: it was found structurally confounded
(determinism inflates perfect-info; round-unique hidden-hand dims are memorization-prone under
a round-level split). The verdict comes from the tournament.

### Step 2 (the likely-RED branch) — pivot to **partner coordination**, not opponent modelling

The corpus's strongest "different axis" signal, and the best structural match to Tichu:

- **Self-play is guaranteed in 2p zero-sum but "systematically converges to sub-optimal Nash
  equilibria in multi-agent cooperative settings."** `[unverified]` Tichu embeds a *cooperative*
  sub-problem (partner coordination) inside a competitive game. Every RL result here has been
  measured as if the failure were competitive.
- **JPS (Joint Policy Search, NeurIPS 2020)**: no belief state, no inference-time search,
  training-time only, monotonic never-worsening guarantee in tabular collaborative games, and
  an **online form that plugs directly into gradient updates**. Its headline result is in the
  closest structural analogue that exists — **4-player 2v2 imperfect-information bridge
  *bidding***: +0.63 IMPs/board vs WBridge5, against prior SoTA +0.41. `[unverified]`
- **JPS outperformed BAD** — the explicitly belief-based, public-belief-state algorithm
  purpose-built for collaborative policy learning. Direct comparative evidence that explicit
  belief machinery is *not* the strongest lever for cooperative imperfect-information play.
- Mechanically it decomposes global game-value change into localized per-infoset policy
  changes, improving *joint* policies **without re-evaluating the entire game** — i.e. it
  sidesteps the full-rollout continuation assumption whose self-inconsistency killed
  piKL/pMCPA/vine.

This maps onto our own open thread: ADR-0033's coupling probes found calls flat but
**schupfen highly sensitive** (−51.8 for rule-based, −142.9 for random) — the one non-flat
signal, and unresolvable by any unilateral probe. Schupfen and the call sub-game are exactly
the bidding/communication-adjacent phases where JPS's bridge gain landed. Unilateral PPO
cannot make a *joint* schupfen↔call↔play improvement; JPS is designed to.

### Step 3 (only if the goal includes human partners) — Off-Belief Learning

OBL is the one belief method whose *purpose* is the self-consistency defect that killed our
search line: it computes beliefs under a **fixed reference policy π₀** for interpreting past
actions while optimizing future play under π₁, converging to a "grounded" policy that draws no
inference from opponents' behaviour. The belief is a supervised net over own
observation-action history, used to sample fictitious states during training rollouts — **no
inference-time search**. `[unverified]`

**But note the measurement trap:** OBL's Hanabi gains appear in *cross-play and human-proxy
play, not self-play score* — OBL-4 scored 24.10 vs Other-Play's 24.14 in self-play (a tie),
but 23.76 vs 21.77 in cross-play and 16.76 vs 8.55 with a human-cloned bot. `[unverified]`

Our seat-swap tournament pairs each team with its own policy, so it is closer to self-play
than cross-play — **it may be structurally blind to exactly this class of gain.** Given
cpfix3328 is a *served* agent playing alongside humans, that is a real measurement gap worth
naming, independent of whether OBL is built.

---

## 7. Do not build

- **ReBeL / DeepStack / Student of Games / CAPI.** Guarantees restricted to two-player
  zero-sum `[verified]`; SoG's value input requires enumerating information states per public
  state, flagged by its own authors as prohibitive `[unverified]`; ReBeL's input grows linearly
  in infostates per public state `[verified]`; CAPI's authors concede it cannot reach
  two-player Hanabi and assumes common payoff. Tichu (4-player, 2v2 team, private schupfen,
  14-card hands) violates every precondition.
- **Belief-weighted PIMC world sampling.** The mechanism that worked in Skat, but search is
  dead here and cheating inference *regressed* in the same game.
- **Naive oracle→student distillation.** Failed outright in Mahjong ("difficult for a normal
  agent … to mimic the behavior of a well-trained oracle"). The A2D result explains why: a
  fixed privileged expert "does not know what the trainee cannot see, and so may encourage
  actions that are sub-optimal, even unsafe, under partial information." Any workable scheme
  must *co-adapt* teacher and student, i.e. it is a training-loop change, not offline
  distillation. `[unverified]`
- **Re-running the perfect-info critic in a new costume** (Suphx dropout annealing, VLOG
  variational oracle). Same family as ADR-0033, which is already flat, and the family's payoff
  is sample efficiency we do not need. (VLOG is noted only because it beat Suphx-style oracle
  guiding *and* has the correct control — a "VLOG-no-oracle" arm indistinguishable from
  baseline — but it is specified for value-based DRL, not PPO actors.)

---

## 8. Sources

Primary sources fetched and claim-extracted (25 total; the load-bearing ones):

- PerfectDou — https://arxiv.org/abs/2203.16406
- OADMCDou (IJCAI 2024) — https://www.ijcai.org/proceedings/2024/0660.pdf
- DouZero+ / opponent modelling + coach network — https://arxiv.org/abs/2204.02558
- ReBeL — https://arxiv.org/pdf/2007.13544
- Student of Games — https://arxiv.org/pdf/2112.03178
- DeepStack reimplementation / Supremus — https://arxiv.org/pdf/2007.10442
- CAPI / public-belief-state planning — https://arxiv.org/pdf/2101.04237
- DeepRole (Avalon) — https://arxiv.org/abs/1906.02330
- SAD (Hanabi, auxiliary head ablation) — https://arxiv.org/abs/1912.02288
- Off-Belief Learning — https://arxiv.org/abs/2103.04000
- JPS / Joint Policy Search (bridge bidding) — https://arxiv.org/abs/2008.06495
- Understanding PIMC success (disambiguation factor) — https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf
- Skat card-location inference for PIMC — https://arxiv.org/abs/1903.09604
- Skat policy inference / cheating-inference regression — https://arxiv.org/abs/1905.10911
- Suphx — https://arxiv.org/abs/2003.13590
- VLOG (variational latent oracle guiding) — https://openreview.net/forum?id=pjqqxepwoMy
- A2D (asymmetric imitation is flawed) — https://arxiv.org/abs/2012.15566
- Informed asymmetric actor-critic — https://arxiv.org/abs/2509.26000
- Belief-I2Q — https://arxiv.org/html/2504.08417v1
- BEPAL — https://arxiv.org/abs/2511.01078
- Consistent opponent modelling — https://arxiv.org/abs/2508.17671
- IRumAI (Rummy, implicit belief probing) — https://arxiv.org/abs/2606.21975

Full extracted claim set with verbatim quotes:
`%TEMP%\claude\...\scratchpad\claims.md` (regenerable from the workflow journal at
`.claude/projects/.../subagents/workflows/wf_ab4af865-ab8/journal.jsonl`).
