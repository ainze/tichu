---
status: proposed
---

# Owner-free EV-probing via a deterministic Claim Solver

[ADR-0031](0031-search-and-learning-loop.md) "Final synthesis" closed the entire
Phase-2 line: every self-play / search lever — search-only on frozen nets,
value-only (exploration-critic), search+learning (starved *and* with a good
R²≈0.46 value), and PPO Refine ([ADR-0029](0029-ppo-refine-design.md)) — was
falsified, and the `forced_press` (−1.4, neutral) / `forced_bomb` (−17.4, harmful)
probes showed the **master plays the diagnosed spots correctly**. The standing
conclusion is that the BC master is a strong baseline at/near the ceiling those
methods reach, and the caller-passivity / bomb-undervaluation diagnosis that drove
the whole effort **was not real**. The only surviving evidence of *any* strength
gap is one anecdotal owner play-test.

This ADR records the direction chosen in the grilling session of 2026-06-05 to
move past that impasse. It does **not** lock numbers; it records the reframe, the
decision, the rejected alternatives, and the falsification criteria so the build
(and the eventual results) proceed from a shared picture. The vocabulary
(**Unbeatable Lead**, **Unseen Cards**, **Guaranteed Out**, **Claim Solver**,
**Slam**) is glossed in [CONTEXT.md](../../CONTEXT.md) §"Endgame / claim terms".

## The reframe — why owner-free, decision-level

"Master-level (goal)" is defined as *superhuman*, while "master" is the decile-9 BC
config — so "the master is sub-master" was partly **definitional**, and is in any
case **unmeasurable without a human in the loop**. Direct owner-vs-master play is
also **statistically hopeless**: the automated EV harness needs n=4000 seat-swapped
rounds for a CI half-width of ~±7.7 pts/round; a human can realistically play
~100–200 rounds, giving a CI half-width of ~±48 pts/round (and worse — the owner
cannot be seat-swapped into both teams), so direct play cannot resolve the sub-20
pts/round effects in question by one to two orders of magnitude.

Therefore the goal is reframed from "is the master superhuman?" (unmeasurable) to
**locate EV-costing weaknesses owner-free, at decision granularity, with EV-probe
attribution**: the scarce signal supplies *states and candidate moves*; cheap
automated compute (force-the-candidate, tournament-vs-master at n=4000 — the
`forced_press`/`forced_bomb` pattern) supplies the statistical power.

## Decision

Build a **Claim Solver** — a deterministic, card-counted detector of a
**Guaranteed Out** — and use it as the candidate generator for the first owner-free
EV-probe. It is chosen as the first probe because it is **structurally unlike
everything already falsified**:

1. **A new behavior class.** `forced_press` / `forced_bomb` both probed *contesting
   an opponent's trick*. Executing your *own* guaranteed run-out from the lead is
   untested — the refutations do not cover it, and it has strong face-validity for
   the owner's "passive at times" perception (a dithered guaranteed line *looks*
   weak).
2. **Immune to the wall that killed the ML line.** It is a **worst-case proof over
   the Unseen Cards**, not Monte-Carlo averaging over **Determinized Worlds** — so
   it has no strategy-fusion problem (the exact defect that made PIMC visit-counts a
   bad policy-improvement target, [ADR-0031](0031-search-and-learning-loop.md)).
3. **Exact and sound.** Card-counting (`fresh_deck − hand −
   public.played_cards_this_round`) is ground-truth common knowledge; the solver
   never certifies a non-guaranteed out, so any EV measurement built on it is clean.

### Scope — two triggers, measured separately

The probe fires only where a Guaranteed Out is also clearly +EV, learning the
`forced_bomb` lesson (forcing a behavior in *all* applicable states was −17.4 even
though it is right in some):

- **Call-fulfillment** — the acting seat called **Tichu / Grand-Tichu** and a
  Guaranteed Out provably exists. Highest stakes (±100/±200), cleanest +EV.
- **Slam** — `out_order == (partner,)` (partner is the *sole* player out), so a
  Guaranteed Out secures the **Slam** (+200, ends the round). The cleanest trigger
  of all: a *locked* +200 a call can still fail to reach. The gate is exactly
  `out_order == (partner,)` — the moment an opponent is already out the Slam is
  dead and it reverts to the deliberately-excluded generic out.

A generic "run my hand whenever I can" out is **out of scope** — going out first
can strand the partner or waste cards (the partnership confound). The two triggers
are recorded as **separate sub-classes** so EV is attributed to each (the master may
nail one and fumble the other).

### Guaranteed Out — definition

Empty the Hand against **worst-case adversarial play by all three other seats,
partner included** (a guarantee that needs the partner to step aside is not a
guarantee — adversarial-partner is sound; cooperative-partner is a deferred recall
lever), quantified over every assignment of the Unseen Cards consistent with
`hand_sizes`, the seen-card mask, and any Mahjong wish. Two cases:

- **Uninterrupted chain** (depth-0): a sequence of **Unbeatable Leads** from the
  lead — no opponent ever plays. The *final* combo need not be unbeatable (the last
  card ends your round regardless).
- **Regain-the-lead** (recursive): shed a beatable card, then *prove* a re-entry
  (e.g. the Dragon) forces the lead back against worst-case opponent leads. The
  Dragon is a guaranteed re-entry only when opponents are *forced* to lead into it,
  never by assumption — this is the worst-case minimax, and the real work.

**Soundness is non-negotiable; recall is the negotiable axis.** A false positive
corrupts the probe; a false negative only shrinks n.

### Probe — two-staged

- **Stage 1 — observational fumble screen (cheap; no forcing, no tournament).** Run
  the Claim Solver as an observer over master self-play; at each trigger state with
  a provable Guaranteed Out, record whether the round actually **converted** (Slam
  hit / call fulfilled) at **round granularity** (a different-but-also-winning line
  counts as success; sidesteps the `forced_bomb` timing confound). Output: a
  **fumble rate** per trigger. **If it is ≈ 0, the master already executes its
  guaranteed outs and the hypothesis is dead** — paid cheaply, before any
  production solver or tournament. The metric slightly *under*-counts (a thrown-away
  guarantee that gets lucky vs the non-adversarial master opponent counts as
  success) — acceptable for a screen: conservative, so surfaced fumbles are real.
- **Stage 2 — EV-probe (only if Stage 1 shows fumbles).** Force the guaranteed line
  in the fumble states, tournament vs master at n=4000 seat-swapped, split by
  trigger. The Slam delta can never legitimately be < 0 (slam dominates), so a
  non-positive Slam delta is a **red flag on solver soundness** — the probe doubles
  as a correctness check.

### Build order

Unbeatable-Lead primitive (near-free off [enumeration.py](../../src/tichu_engine/enumeration.py)
over the Unseen Cards) → chain out-solver → regain-the-lead recursion. Stage-1
screen before any tournament. TDD, as the project follows.

## Reuse map

| Component | Source | Verdict |
|---|---|---|
| Unseen-card counting | `public.played_cards_this_round` + `fresh_deck` | reuse as-is (ground truth) |
| Opponent-response enumeration | `tichu_engine.enumeration.enumerate_*` | reuse as-is (the response oracle) |
| Self-play round driver + observer hook | `tichu_eval.play_full.play_full_round(..., collect_telemetry=True)` | reuse skeleton; add a Claim-Solver observer |
| Force-candidate-then-tournament-vs-master | `search/forced_press.py`, `forced_bomb.py` + `configs/eval_tournament_forced_*.yaml` | reuse pattern for Stage 2 |
| Claim Solver (primitive + chain + regain-lead) | — | **new** |

## Rejected alternatives

- **Direct owner-vs-master win-rate** — statistically hopeless at human throughput
  (above). Rejected as the yardstick; the owner is removed from the measurement loop
  entirely.
- **Corpus move-prediction as the candidate source** (force "what decile-9 humans
  do" where the master diverges) — viable and owner-free, but caps at the decile-9
  ceiling and only adjudicates imitation divergence. Held as the **next probe** if
  the Claim Solver comes back neutral, not the first.
- **Deep-search-as-auditor** (force heavy-PIMC's move only at high-disagreement
  states) — owner-free with no human ceiling, but untried and heavier. Also held as
  a later probe.
- **Option 2 — a different imperfect-info method (CFR / DouZero-DMC)** — high cost;
  PPO, its close cousin, was already flat. Not justified before a real located gap.
- **Option 4 — accept BC+PIMC and re-scope** — the honest endpoint *if* this and
  the held probes find no located gap, but premature now: not a single rigorous
  strength measurement has been run, and the Claim Solver tests an untested class.

## Consequences

- Commits to a hand-coded, exact, single-machine solver — no training, no nets, no
  serving budget — in deliberate contrast to the entire falsified ML line. The
  surprise of that contrast is the reason this ADR exists.
- The first deliverable is the **Stage-1 screen**, whose cost is dominated by cheap
  self-play; it can falsify the hypothesis before the regain-the-lead recursion (the
  expensive part) is built.
- The Claim Solver, if a gap is found and confirmed +EV, is a candidate **deployed**
  endgame module (drop-in at a Play Decision), not only a diagnostic — but that is a
  later, separate decision.

## Risks

1. **Partition blow-up.** "For all consistent assignments of the Unseen Cards" is
   combinatorial. **Resolved for v1 (grilling 2026-06-05): regain-the-lead uses a
   capped/pooled-adversary relaxation (B)** — model the opponents as a single
   adversary holding all Unseen Cards, which is *sound for the stranding threat*
   (a pooled adversary strands at least as hard as any real partition) and kills
   the partition combinatorics, at a recall cost. **Exact partition enumeration
   (A)** — enumerate consistent assignments honouring `hand_sizes` and minimax each
   — is the **noted recall-escalation follow-up** if Stage 1 comes back empty, not
   built now. The endgame card-count threshold remains a build-time measurement.
   **Resolved (build, 2026-06-05) — v1 ships chain-only.** The Guaranteed-Out win
   condition is "out *before* the opponents" (Tichu/Grand require first-out; the
   Slam requires second-out before either opponent), and a player is **out the
   instant their hand is empty even if you beat their final combo**. So the moment
   you cede the lead, a worst-case opponent may lead a hand-emptying combo and be
   out before you — meaning *broad* regain-the-lead is almost never a worst-case
   guarantee (the Dragon-re-entry pattern is a strong *probabilistic* play, not a
   proof). Under the non-negotiable soundness requirement, **sound regain-the-lead
   collapses to narrow provable cases**, so v1 is the **chain-only** Guaranteed Out
   (you never cede the lead → no race possible → fully sound). Staged soundness:
   Stage 1 (fumble screen) needs strict guarantees, which chain-only gives; Stage 2
   (EV-probe) tolerates probabilistic lines since the n=4000 tournament adjudicates
   actual EV. Follow-ups, in order if chain-only Stage 1 is empty: narrow-sound
   regain (prove no opponent can empty while you are stranded), then exact-partition
   (A) / probabilistic strong-line detectors for Stage 2 only.
2. **Soundness bugs.** A single false-positive certification poisons the probe.
   Mitigation: a property test that every certified out, replayed against exhaustive
   adversarial opponent lines, *always* completes; plus the Slam-delta red-flag.
3. **False-negative from under-scoped recall.** A chain-only solver would miss the
   regain-the-lead lines where the master most plausibly fumbles — hence
   regain-the-lead is in v1, not deferred.
4. **The decile-9 / owner blind spot.** If the owner is genuinely stronger than top
   BSW play in ways this probe cannot see, a clean result does not prove the master
   is strong — only that this located hypothesis is not the gap. Accepted: the held
   probes and, ultimately, option 4 cover that tail.

## Success criteria

Stage 1 surfaces a non-trivial **fumble rate** in at least one trigger, and Stage 2
confirms forcing the guaranteed line is **+EV vs master at n=4000** (CI excluding 0),
split by trigger. That is the first rigorously *located* strength gap in the project.
A Stage-1 fumble rate ≈ 0 **falsifies the hypothesis cheaply** and routes to the next
owner-free probe (corpus, then deep-search-as-auditor) or, if those also come up
empty, to option 4.

## Result — chain-only EV-probe (2026-06-05)

Built test-first: the Claim Solver (`tichu_engine/claim.py` — `unseen_cards`,
`is_unbeatable_lead`, `guaranteed_out_chain`) and `ForcedClaimAgent`
(`tichu_training/search/forced_claim.py`, registered `forced_claim`), which forces the
first combo of a chain Guaranteed Out when, **leading a fresh trick**, the seat is a
Tichu/Grand caller or has its partner as the sole player out (a live Slam).

- **Opportunity is not the limiter.** A self-play screen (`scripts/screen_forced_claim.py`,
  60 rounds) forced in **~80 spots across 45/60 rounds** — chain Guaranteed Outs at trigger
  states are common.
- **Perf bug found + fixed.** The first `is_unbeatable_lead` pooled the whole Unseen set
  into `legal_actions`, enumerating *every* combination of a ~full deck
  (straights/full-houses/pair-steps blow up) → **~28 GB**. Fixed to enumerate only the
  lead's **own type + Bombs** (the complete legal-beater set; `_beats` still adjudicates;
  Phoenix-following kept explicit) → 3.5 GB across 6 workers, sound (all primitive tests
  unchanged, incl. the Phoenix and bomb cases).
- **Tournament (full pool, n=4000 seat-swapped, `configs/eval_tournament_forced_claim.yaml`):**

  > **forced_claim vs master: mean = −1.4, 95% CI [−7.97, +5.95], call-bonus +0.35.**

  **EV-neutral — indistinguishable from master, and near-identical to `forced_press` (−1.4).**
  The master already converts its uninterrupted-chain Guaranteed Outs; the most blatant form
  of the "fumbles guaranteed outs / passive" intuition is **refuted** at n=4000. This is the
  third sound EV-probe to land neutral-or-negative (forced_press −1.4, forced_bomb −17.4,
  forced_claim −1.4) — a consistent signal that the master plays the diagnosed spots correctly.

**Scope caveat (load-bearing).** Chain-only is *sound but low-recall*: it does **not** test
**regain-the-lead** (the Dragon-re-entry pattern), which is not worst-case certifiable and was
deferred. So the owner's specific intuition is **not yet** falsified — only its blatant,
uninterrupted-chain form is. The one remaining probe that targets it is a **probabilistic
regain-the-lead detector** (Stage-2-only: the n=4000 tournament adjudicates EV, so soundness is
not required), reusing the entire `forced_claim` + tournament harness with a different detector.
If that too lands neutral, the indicated move is option 4 (accept BC+PIMC, re-scope).

## Result — probabilistic regain-the-lead probe (2026-06-05)

Built `reclaim_out` (`tichu_engine/claim.py`) — the chain with one relaxation: a non-final combo
may be a **beatable shed** provided the remaining Hand still holds an Unbeatable Lead (a **Re-entry**
— bomb, card-counted-unbeatable high single/pair — covering all of the owner's regain examples
uniformly). `ForcedClaimAgent(reclaim=True)` swaps it in; same harness.

- **Opportunity:** the reclaim screen forced in ~94 spots / 46 of 60 rounds (more than chain's 80).
- **Tournament (n=4000, `configs/eval_tournament_forced_claim_reclaim.yaml`):**

  > **forced_claim_reclaim vs master: mean = −13.2, 95% CI [−19.6, −6.0], call-bonus −2.35.**

  **Decisively negative (~4σ, CI excludes 0).** Forcing the regain-the-lead line is *harmful* — the
  master is **correct to decline** shed-and-reclaim. This is the `forced_bomb` shape (−17.4): a
  broader, non-guaranteed behavior forced indiscriminately loses EV because the worst-case (stranded,
  or out-raced) bites often enough. (Caveat, as with `forced_bomb`: this forces the reclaim line
  *whenever the detector fires*, not selectively; a perfectly selective reclaim could still be +EV —
  but that is plausibly what the master already approximates, and there is no evidence it under-uses it.)

## Final synthesis (2026-06-05)

**Four EV-probes now target the "master is too passive / misses outs" intuition, and all land
neutral-or-harmful:** `forced_press` −1.4 (neutral), `forced_claim` chain −1.4 (neutral),
`forced_claim_reclaim` −13.2 (harmful), `forced_bomb` −17.4 (harmful). The sound/guaranteed behaviors
the master **already performs**; the broader/probabilistic ones it **correctly declines**. The owner's
"feels too weak / passive" perception does **not** correspond to a located EV gap in anything these
owner-free probes can measure. The honest conclusion is **option 4**: the BC master is a strong baseline
at/near the practical ceiling these methods reach; accept BC (+PIMC) as the product and re-scope away
from "superhuman", **or** pursue a fundamentally different instrument that can see a gap these cannot
(e.g. a corpus decile-9 move-prediction audit — cheap, never yet run — or, ultimately, logged
owner-vs-master games despite the throughput cost). The Claim Solver, EV-probe harness, and four
decisive probes are built, correct, and reusable. Status of this ADR's hypothesis: **refuted** for the
behaviors probed; the solver remains a candidate *deployed* endgame module independent of the gap question.

## Result — decile-9 move-prediction audit (2026-06-05)

The held-fallback corpus instrument, finally run on v5 (`scripts/move_pred_audit.py`: 1500 archive
games, decisions filtered to the acting seat's `skill_decile == 9` via `ratings_full.parquet`, master
conditioned on decile-9):

| decision | n | top-1 | top-5 |
|---|---|---|---|
| **play** | 169,038 | **0.834** | **0.990** |
| dragon_assignment | 2,900 | 0.806 | (no ranking) |
| wish | 2,332 | 0.351 | (no ranking) |
| schupfen | 12,252 | 0.100 | (no ranking) |

**No play-imitation gap.** Play top-1 0.834 / **top-5 0.990** — the master already ranks the decile-9
human's actual move in its top 5 ~99% of the time; the top-1 residual is selection among near-equivalent
moves (the Bayes ceiling of imitation, matching the 2026-06-02 capacity note's ~84%), not unlearned
structure. The disagreement map cuts *against* passivity: the master plays *more* than decile-9 humans on
the pass/play margin (Pass→Single 4455 vs Single→Pass 3365) and bombs slightly *less* (correct per
`forced_bomb`). Only **wish (0.35)** and **schupfen (0.10)** diverge materially — both huge-action-space,
high-entropy, separate-network decisions, secondary to play strength.

**Bearing on the BC-scaling levers (grilled 2026-06-05).** "A stronger player via a deeper/wider trunk or
a shuffled full-corpus BC" both rested on the master *underfitting* decile-9 play. The audit (top-5 0.990)
says it does not — and BC caps at decile-9 regardless. So neither lever can yield a stronger *player* on
the play decisions that matter; the prior capacity sweep (2026-06-02, +0.08 pp at 2.25×) already disfavoured
trunk width. Shuffle is the only genuinely untested one (the master is `*_unshuffled`, an optimization risk),
but the audit predicts a flat play-acc bake-off. **Fifth convergent instrument** (4 EV-probes + this audit):
the master is at/near the decile-9 ceiling and plays the measured decisions correctly → **option 4**, or a
fundamentally different (non-BC, non-search) method, remains the honest path.
