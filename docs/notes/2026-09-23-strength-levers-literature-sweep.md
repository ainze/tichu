# 2026-09-23 — Deep research: do 2024–2026 results offer a new strength lever?

## TL;DR

**No new high-confidence lever.** The sweep's two strongest recommendations are already built
here: the PerfectDou oracle critic (ADR-0033, flat) and Monte-Carlo returns / λ=1 (live since
~iter 22.46k, ADR-0044). A KL anchor that moves as the agent improves also exists already
(`reanchor_on_promote`). No principled imperfect-information search method published in the
window transfers to a 2v2 partnership game with chance deals, and none fixes the τ-continuation
bias that killed piKL (ADR-0037).

Four ideas are untested here. All are low-evidence, and none of the source papers ablates them:
Ataraxos-style advantage-magnitude filtering, power-law annealing of the anchor-KL strength,
soft top-k candidate PPO (DanZero+), and deal-level curriculum by reweighting (DouZero+ coach).
If one is run, phase-stratified advantage filtering is the cheapest.

This agrees with the [2026-07-26 sweep](2026-07-26-opponent-modelling-deep-research.md): in
this family of games, privileged-information results are sample-efficiency results, and this
project is not sample-limited.

## Provenance

`deep-research` workflow run `wf_d9af6e37-256`: 5 search angles, 19 primary sources fetched,
94 claims extracted, 25 sent to 3-vote adversarial verification (**15 confirmed, 10
refuted, 0 unverified**), synthesised to 7 findings; 101 agents, none errored. The prompt
listed the directions this repo had already killed, so that the sweep would not simply rediscover
them. The mapping to local status below was added by hand after the run.

Caveats: several of the most transferable sources are older than the requested window
(PerfectDou 2022, DouZero+ 2022, DanZero+ Dec 2023). The in-window results (Ataraxos, LAMIR,
Obscuro, EPIMC, EMAgnet, SePoT) are nearly all 2-player zero-sum. The Guandan papers evaluate
mostly against rule bots, with no confidence intervals. Every transfer to Tichu is an
extrapolation.

---

## 1. Findings already built in this repo

| Finding | Evidence (vote) | Local status |
|---|---|---|
| **Oracle / asymmetric critic** (PerfectDou PTIE): the critic sees all hands, the actor stays imperfect-info | PerfectDou ablation vs DouZero-1e10: perfect-info critic WP 0.524 / ADP +0.014 vs imperfect-info critic WP 0.486 / ADP −0.057, ≈ +3.8 WP (2-1). The critic must take history **and** hidden state; Baisero & Amato (AAMAS 2022) show that a pure state critic biases the gradient | **ADR-0033: −3.7 vs master, indistinguishable from the symmetric critic.** The July sweep explains why: a sample-efficiency gain, not an asymptotic one |
| **Monte-Carlo returns** (GuanZero, Guandan, Feb 2024: DMC, 81–82% vs the CGAIAC rule champion, 75–77% vs DouZero-based agents) | 3-0 | **λ=1 live since ~iter 22.46k** (ADR-0044); at γ=λ=1 the critic is a pure baseline and cannot bias the gradient |
| **Moving regularisation anchor** | Ataraxos's schedule (see §2) | **`reanchor_on_promote`**: the KL anchor advances with each promotion |

Note on λ: Ataraxos (superhuman Stratego, Nov 2025) uses λ=0.5 for advantages and λ=0.8 for
value targets. That points the **opposite** way from this repo's diagnosed fix
([advantage-SNR probe](2026-07-24-advantage-snr-probe-critic-bias.md)): a lower λ leans harder on a
critic whose residual is epistemic bias. It is not evidence for lowering λ, and the paper reports
no λ ablation.

## 2. Untested leftovers (all low evidence)

1. **Advantage-magnitude filtering** (Ataraxos): train the policy only on transitions whose
   |A| is in the top quartile and at least 0.01. The authors credit it with a ~2.5× wall-clock speedup and
   better asymptote, but report **no ablation**. Local caveat: at λ=1, A = R − V(s), so a large |A|
   largely marks lucky or unlucky continuations (aleatoric noise), not informative decisions. A
   usable version would have to stratify the quantile by phase, or normalise by the probe's
   per-phase aleatoric sd. The forced rows are already dropped (ADR-0044), which removes
   the most obvious zero-signal mass. **Cheapest of the four; the one to run if any.**
2. **Annealed anchor strength** (Ataraxos dynamic damping): a constant reverse-KL 0.1 to the
   data-collection policy (a trust region separate from the anchor), plus a magnet KL decayed as
   `0.05/iter^0.3`, entropy `0.1/iter^0.3`, and lr `clip(0.5/iter^1.1, 5e-6, 1e-4)`. What
   transfers is the *shape* (strong regularisation early, weak late), applied to the BC/champion
   anchor. The uniform magnet does not transfer. Tabula rasa, 2p zero-sum, no ablation. The
   related EMAgnet (June 2026; EMA of own params as the magnet) was tested only on toy games,
   and the claim that it could replace the BC anchor in Tichu was **refuted 1-2**.
3. **Soft top-k candidate PPO** (DanZero+, Guandan): a pretrained Q picks the top-k actions and
   PPO chooses among them conditioned on state (k=2: 92.70% vs baseline1; k=3 72.17%; k=5
   70.28%). The mechanism is verified 3-0, but the **gain over DMC alone was refuted 0-3**, and all
   baselines are rule bots. A hard top-k caps the policy at the prior's candidates, which is
   the opposite of escaping the BC plateau. Only a soft or large-k variant is worth considering.
4. **Deal-level curriculum** (DouZero+ coach network, 2022): predict P(win) from the initial hands,
   reject lopsided deals, and anneal the threshold β from 0 to 0.3. The evidence is single-run
   curves with no CI, and a speed-up confounded with strength. Local caveats: deal luck is only ~14k of
   ~55k round-start return variance (advantage-SNR probe), and dropping lopsided deals
   removes exactly the strong hands where Grand Tichu and Tichu calls are made. Reweight, don't drop.

## 3. Search: nothing transfers (high confidence)

| Method | Why it doesn't fix the piKL failure |
|---|---|
| **EPIMC** (CoG 2024): delays the perfect-info leaf to depth d | Big gains in Dark Chess (≈80% vs 45%), but "increasing the depth does not lead to any improvement" in its trick-taking card game and Battleship, where most observations are public. Tichu's play phase is like this: only the schupfen pass is private |
| **LAMIR** (Oct 2025): learned abstract model + CFR+ continual resolving | 2p zero-sum, no chance nodes, Leduc only with workarounds; unreviewed preprint |
| **Obscuro** (2025/26) | 2p zero-sum, no learned net (Stockfish d11 leaves), full infosets in memory |
| **SePoT**: V-trace critic over opponent-policy transformations | 2p zero-sum, small subgames, +1.1–3.5/100 in Goofspiel. Trades rollout variance for critic bias, the thing already diagnosed as this repo's problem |
| **Ataraxos test-time search**: one damped MMD update at the root, ~1,000/\|legal\| belief-sampled worlds, 39-ply rollouts | The closest analogue to piKL (2-1, low). The ~100× budget plus damping addresses the **variance** half of piKL's −58.66 (per-decision Q signal ≈ per-world SE at N=10), not the **τ-continuation bias**, which gets worse in a partnership game where the partner's continuation also changes. The belief net adds little here (+0.024 top-1, ADR-0041) |

**Refuted 0-3, do not cite:** that team max-margin subgame solving (Farina et al., NeurIPS 2022)
gives Tichu a safe nested refinement; that SePoT or Student of Games avoid the
blueprint-continuation assumption; that Obscuro's KLUSS is a drop-in alternative to
determinized rollouts.

## 4. Other refuted claims

- "Most of PerfectDou's edge is features, not the oracle critic" (0-3): the ablation numbers
  exist, but this reading of them was rejected. It does not rescue the oracle critic locally
  either (ADR-0033 is our own measurement).
- DanZero+ PPO fine-tuning is +2.6–3.6 over DMC alone (0-3).
- GuanZero's hand-designed partnership-behaviour indicators (Cooperating / Dwarfing /
  Assisting one-hots in the action encoding) as the paper's main contribution (1-2).
- AlphaDou's win-probability + expected-score factorised Q as a critic-head fix (0-3).

## 5. Areas with no surviving claims

R-NaD/DeepNash specifics, NFSP successors, PSRO/league training, DiL-piKL/Cicero-style
human-regularised RL, Suphx follow-ups, Bridge/Hanabi/Skat/Hearts/Big2, distributional critics,
offline-to-online RL, and LLM/transformer card agents. This means **not verified**, not that the
literature is empty. A future sweep should target these specifically. The most useful would be
human-level Guandan/Big2 evaluations that report a controlled PPO-over-BC gain.

## Sources

- PerfectDou — https://arxiv.org/abs/2203.16406
- DanZero+ — https://arxiv.org/abs/2312.02561
- GuanZero — https://arxiv.org/abs/2402.13582
- AlphaDou — https://arxiv.org/abs/2407.10279
- EPIMC — https://arxiv.org/abs/2408.02380
- DouZero+ — https://arxiv.org/abs/2204.02558
- Ataraxos (Stratego) — https://arxiv.org/abs/2511.07312
- EMAgnet — https://arxiv.org/abs/2606.23995
- Obscuro — https://arxiv.org/abs/2506.01242
- Team subgame solving — https://www.mit.edu/~gfarina/2022/subgame_solving_teams_neurips22/subgame_solving_teams_neurips22.pdf
- LAMIR — https://arxiv.org/abs/2510.05048
- SePoT — https://arxiv.org/pdf/2312.15220
- Student of Games — https://www.science.org/doi/10.1126/sciadv.adg3256
