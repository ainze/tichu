# PRD — Tichu AI: Python Implementation (Phase 1)

## Problem Statement

We are building a chess.com-style web platform for Tichu where players can practice against AI opponents. Tichu is a four-player partnership game; without competent AI, players who can't find three other humans cannot play, and players who want to practice cannot learn against a consistent opponent. Existing online Tichu platforms either lack bots entirely or use weak rule-based agents that experienced players quickly outgrow, leaving casual users with no opponents and serious users with no realistic practice partner.

The AI opponent therefore is not a feature — it is the precondition for the product existing at all. It must be competitive enough that experienced players treat it as a worthwhile practice partner, fast enough that move latency does not degrade the game experience, and tunable so that newcomers face an opponent at their level rather than being crushed.

We have access to a corpus of 2.4 million Tichu games (entire Brettspielwelt history from 2007 onwards), with consistent player handles across games and full hidden information available post-game. The corpus does not include explicit skill ratings.

## Solution

A Python codebase that produces, trains, and serves a Tichu-playing agent at "decent human" level or better, deployable behind the game server as an inference service.

The system has four cleanly separated layers. A pure-Python rules engine implements full Tichu in a form usable by both the training pipeline and the game server's rule validation. A parser converts the BSW corpus into structured training records, validating each game by replaying it through the engine and confirming the final score matches BSW's recorded result. A multi-head behavioral cloning model — sharing a trunk across the structurally distinct decisions of grand-tichu call, regular-tichu call, card passing, in-trick play, mahjong wish, and dragon-give-direction — learns to imitate strong human play, conditioned on a player-skill signal derived from a TrueSkill rating computed offline from game outcomes. An inference service exposes the trained model behind a single HTTP endpoint, with selectable difficulty levels mapping to distinct checkpoints.

A separate evaluation harness runs all-vs-all tournaments between named agents over a fixed pool of deals with seat-swap variance reduction, producing a matrix of paired score deltas that is the project's north-star quality metric. Every checkpoint is evaluated against the same baselines and previous checkpoints; no model ships without an eval row.

A separate belief model — predicting each opponent's remaining cards from public game history — is trained on the same parsed corpus using post-game-visible hidden hands as labels. It is not used in the phase-1 inference path but is built now because the data scale supports it cheaply and it is the bridge piece that enables phase-2 search-based extensions.

Phase 1 deliberately omits online self-play training (PPO and similar). At the data scale available, offline behavioral cloning with skill-stratified imitation plus optional offline RL refinement is expected to produce a stronger agent than self-play would, with far less infrastructure complexity.

## User Stories

1. As a player, I want to play Tichu against AI opponents at multiple difficulty levels, so that I can practice without coordinating three other humans.
2. As a player, I want AI moves to complete within a few seconds at most, so that the game maintains a natural pace.
3. As a player, I want the AI to correctly use all special cards (Phoenix, Dragon, Mahjong, Dog), so that the game feels authentic rather than buggy.
4. As a player, I want the AI to make sensible Tichu and Grand Tichu calls — neither never calling them nor calling them recklessly — so that the bot faces me with realistic stakes.
5. As a player, I want my AI partner to cooperate intelligibly during a hand, so that partnership play is meaningful rather than random.
6. As a player, I want the AI to make tactical decisions about the three-card pass at the start of each hand, so that opening play reflects real Tichu strategy.
7. As a player, I want a difficulty selector that meaningfully changes how strong the AI plays, so that I can grow from beginner to advanced against the same product.
8. As a player, I want the AI to never make illegal moves or crash mid-game, so that I am never stuck waiting on a broken opponent.
9. As a developer, I want a pure-Python rules engine without ML dependencies, so that the game server can validate moves without loading PyTorch.
10. As a developer, I want the rules engine to be deterministic and importable as a library, so that it can be reused identically across training, eval, and serving.
11. As a developer, I want every game state to expose a legal-actions mask, so that all agents (learned, rule-based, human) share a single legality contract.
12. As a developer, I want the engine to enforce all Tichu rules including Phoenix substitution, Mahjong wish fulfillment, out-of-turn bombs, and Dog-passes-to-partner, so that the simulator matches Brettspielwelt's behavior.
13. As a developer, I want to step the engine forward by applying one action at a time, so that I can replay logged games and run rollouts symmetrically.
14. As a developer, I want the engine to clearly distinguish public state from private (per-player) state, so that the right information is exposed to each agent.
15. As a developer, I want all card and combination types defined as immutable typed objects, so that the engine is easier to reason about and statically check.
16. As a developer, I want property-based tests for the combination enumerator, so that edge cases like Phoenix-in-straight or Phoenix-in-pair-step are caught automatically.
17. As a developer, I want a parser that converts raw BSW log files into structured training records, so that the 2.4M-game corpus is usable as training data.
18. As a developer, I want the parser to validate every parsed game by replaying it through the rules engine and asserting that the final score matches Brettspielwelt's recorded result, so that parsing bugs and rules-engine bugs surface immediately.
19. As a developer, I want the parser to skip incomplete or abandoned games gracefully and log them, so that bad data does not corrupt training.
20. As a developer, I want parsed training records sharded by decision type and stored as Parquet, so that each model head loads only relevant data.
21. As a developer, I want player handles preserved across all games in the parsed output, so that career-level skill ratings can be computed.
22. As a developer, I want the parser to record game IDs and timestamps, so that train/test splits and recency weighting are reproducible.
23. As a developer, I want a TrueSkill rating computed for every player from game outcomes, so that training data can be stratified by skill despite BSW providing no explicit ratings.
24. As a developer, I want to filter training data by player skill or condition the model on skill, so that the policy imitates strong play rather than the population median.
25. As a developer, I want auxiliary skill signals such as per-player Tichu success rate to be computed, so that I can cross-validate the TrueSkill ranking.
26. As a developer, I want minimum-game thresholds enforced on skill ratings, so that cold-start ratings from players with few games do not pollute the skill signal.
27. As a developer, I want a versioned, deterministic, side-effect-free featurization function mapping a private state to a fixed-shape tensor, so that trained models stay compatible across code revisions and re-runs are byte-identical.
28. As a developer, I want the featurizer's version pinned to every saved checkpoint, so that loading a checkpoint with a mismatched featurizer fails loudly rather than corrupting outputs silently.
29. As a developer, I want the full action space encoded as one canonical ordering, persisted with each checkpoint, so that the meaning of policy outputs cannot drift.
30. As a developer, I want a shared-trunk multi-head policy network, so that representation is shared across decision types while each head specializes for its own action subspace.
31. As a developer, I want behavioral cloning training over decisions weighted or filtered by skill, so that the resulting policy targets strong play.
32. As a developer, I want masked cross-entropy losses on every head, so that the model never assigns probability mass to illegal actions during training.
33. As a developer, I want separate small networks for Grand Tichu and regular Tichu call decisions, so that calling aggressiveness can be tuned independently of in-trick play.
34. As a developer, I want a belief model that predicts each opponent's remaining cards from public history, so that phase-2 search methods have a calibrated estimate of hidden state to plan over.
35. As a developer, I want training metrics logged to disk and to a tracking dashboard, so that experiments are auditable and comparable.
36. As a developer, I want training runs checkpointable and resumable, so that multi-day jobs survive interruptions.
37. As a developer, I want training scripts to be parametrized by config files, so that experiments are reproducible from a single artifact.
38. As a developer, I want an optional offline RL refinement stage that improves a BC checkpoint using game outcomes as reward signal, so that the policy can exceed the median quality of the dataset without online self-play.
39. As a developer, I want offline RL to reuse the same featurizer and action space as BC, so that resulting checkpoints are hot-swappable at inference.
40. As a developer, I want a tournament harness running all-vs-all matches between named agents over a fixed pool of deals with seat-swap variance reduction, so that any two checkpoints can be compared with statistical confidence.
41. As a developer, I want baseline agents (Random, Rule-based) implementing the same agent interface, so that they serve as fixed reference points for the whole project lifetime.
42. As a developer, I want every tournament run to produce a matrix of paired average score deltas with bootstrap confidence intervals, so that small per-round differences between checkpoints are detectable.
43. As a developer, I want eval mode to force all agents into deterministic action selection (argmax or fixed RNG seed), so that within-agent randomness does not re-introduce variance.
44. As a developer, I want a held-out set of BSW games used as a separate evaluation — predicting the held-out human moves — so that imitation quality is measurable independently of self-play results.
45. As a developer, I want every eval run reproducible from a config plus a seed, so that any reported result can be re-verified later.
46. As a developer, I want trained models exportable to TorchScript or ONNX, so that production inference does not require the full training environment.
47. As an operator, I want the inference service to be stateless and horizontally scalable, so that many concurrent games can be served by adding replicas.
48. As an operator, I want per-move inference latency under 500 milliseconds at the 99th percentile, so that game pace feels natural.
49. As a developer, I want the inference service to expose a single endpoint that accepts a serialized private state and returns an action, so that the game server's integration surface is minimal.
50. As a developer, I want the inference service to support difficulty modes mapping to distinct checkpoints (Easy: rule-based, Medium: small BC, Hard: full BC, Master: BC plus offline-RL refinement), so that users can pick their level.
51. As an operator, I want the inference service to expose health and metrics endpoints, so that production deployment is observable.
52. As a developer, I want the inference service to fall back to a uniformly random legal action with a loud log if the model produces malformed output, so that no game ever stalls due to inference failure.
53. As a developer, I want the entire Python codebase managed by one dependency manifest and one lockfile, so that local development and CI build identical environments.
54. As a developer, I want CI to run rules-engine tests, parser replay-validation on a fixed subset of games, and a training smoke test on every pull request, so that regressions are caught before merge.
55. As a developer, I want each training, eval, and inference task runnable from a single CLI command, so that experiments and operations are reproducible and scriptable.
56. As a future maintainer, I want the architecture to leave room for a phase-2 rewrite of the policy model to a DouZero-style deep Monte-Carlo network, so that the upgrade path does not require replacing the engine, parser, eval harness, or serving stack.

## Implementation Decisions

**Four-package layout.** The Python codebase is organized as four importable packages with one-way dependencies. The rules engine is dependency-light pure Python. The ML library (featurizer, action space, models, agent interface) depends on the engine and on PyTorch. The training package depends on both and adds data pipeline, BC, offline RL, and eval. The inference service depends on the ML library and adds a web framework. The game server (not part of this PRD) imports the rules engine directly and the inference service over HTTP.

**Engine implements full Tichu, not a simplification.** The Müller paper's simplifications (no straight-flush bombs, no asynchronous bombs, single round equals single episode) are choices about the training environment, not about the engine. The engine implements the full game including straight-flush bombs and out-of-turn play. Training environments wrap the engine and can optionally restrict actions, but the engine itself never lies about Tichu's rules.

**State separation.** `PublicState` and `PrivateState` are distinct types. The engine returns the correct view for the acting player at each step. The serialized form of `PrivateState` is what the inference service accepts and the featurizer consumes.

**Canonical action space, pinned to a version.** A single ordered list defines every action — play combinations, pass, win-trick, Tichu calls, schupfen choices, wish-rank, dragon-give-direction. The version is saved alongside every checkpoint and asserted at load time. Adding or reordering actions creates a new version and invalidates prior checkpoints.

**Featurizer is a pure function with a version.** `featurize(private_state) -> ndarray` has no I/O, no globals, no time-dependence. Its version is also pinned to checkpoints. Changing features creates a new version.

**Multi-head architecture, shared trunk.** A single trunk produces an embedding from `(featurize(private_state), skill_embedding)`. Heads specialize: a play head over the in-trick action subspace, a pass head over schupfen choices, a wish head over rank choices, a dragon-give head. The Tichu and Grand-Tichu call decisions use separate small networks rather than heads on the shared trunk, because their input states differ structurally from in-trick states. Final trunk type (transformer over history sequence vs large MLP over engineered features) is decided by benchmarking on a held-out parsed subset before scaling up training compute.

**Training record schema, as emitted by the parser.** One record per player decision. Fields are kept abstract here; concrete column types live in the data pipeline package's schema definition. The minimal shape decided up front (this is the schema decision the rest of the pipeline depends on):

```
state               : serialized PrivateState
legal_actions_mask  : bitmask over the canonical action space
action_taken        : canonical action index
decision_type       : one of {play, pass_card, call_tichu, call_grand_tichu,
                              wish_rank, dragon_give}
player_handle       : string, stable across the BSW corpus
round_outcome       : final point delta for this player's team
round_won           : bool
game_id, round_id, timestamp
featurizer_version  : pinned to the writer
action_space_version: pinned to the writer
```

**TrueSkill ratings as a precomputed table.** Player ratings are computed offline in one chronological sweep over the corpus and cached as `(player_handle -> mu, sigma, n_games)`. Bucketed into deciles for use as a model-input embedding. At inference, the top decile is always selected. Training jobs receive the table as an input artifact rather than recomputing on demand.

**Skill conditioning beats hard filtering.** Decision: train on the full corpus with skill-bucket embeddings concatenated to the input, rather than dropping low-skill players. The model gets to see weak play labeled as weak, which is more informative than only seeing strong play. Hard filtering by skill is available as a baseline configuration but not the default.

**Behavioral cloning is the primary training method.** Masked cross-entropy on `(state, action)` pairs. Skill-weighted via the conditioning embedding. Training runs on a single high-memory GPU; multi-GPU is an optimization, not a requirement.

**Offline RL (AWR) as the optional refinement stage.** Advantage-Weighted Regression is chosen as the first offline RL method because it is the simplest to integrate (a weighted variant of BC) and has the most robust empirical track record. CQL and Decision Transformer are listed as alternative methods to try if AWR underperforms; both are research items, not phase-1 requirements.

**Belief model is a separate first-class component.** It is not on the inference path in phase 1 but is trained alongside BC because the data permits it cheaply (post-game hidden hands are in the logs). Its outputs feed into phase-2 search methods.

**Replay validation is the highest-leverage test.** The parser must reproduce Brettspielwelt's `Ergebnis` to within rounding on at least 99.9 percent of valid games. This single test catches the vast majority of bugs that would otherwise corrupt training.

**Eval harness uses fixed deal pool + seat-swap.** A deal pool of 10,000 pre-generated deals is the canonical evaluation set. Every checkpoint plays every other checkpoint and every baseline against this same pool with the one-position seat rotation that swaps team hand assignments. Output is a paired-difference matrix with bootstrap CIs. Every checkpoint joins the matrix at training-completion time; no checkpoint ships without an eval row.

**Inference service.** A web framework with native async support serves a single `POST /act` endpoint. The service loads TorchScript or ONNX-exported policy artifacts at startup and never reloads them within a process lifetime. Stateless. Difficulty mode is a request parameter that selects among loaded checkpoints. Fallback to a uniformly-random legal action on any model error.

**Configuration via YAML.** All training and eval jobs read a single YAML config and copy it into the run artifact directory. The CLI surface is one entry point per task: `train_bc`, `train_calls`, `train_belief`, `eval_matrix`, `parse_bsw`, `compute_trueskill`, `serve_inference`.

## Testing Decisions

Good tests for this project verify external observable behavior, not implementation details. The right level of abstraction is "does the engine produce the same score as Brettspielwelt on this game?" not "does this internal helper return a dict with these keys?" Implementation can churn — engine internals will be rewritten as performance demands — and tests pinned to implementation detail become carrying cost rather than safety net.

**Rules engine.** Property-based tests using Hypothesis over the combination enumerator (any hand of 14 cards from the deck should enumerate combinations satisfying invariants: no duplicates, all legal, all rank-comparable). Canonical-example tests for every case explicitly called out in the official Tichu rulebook (Phoenix-in-straight, Mahjong wish forcing, Dog passing, Dragon-give, out-of-turn bombs interrupting pass cycles). Determinism test (same inputs produce identical outputs).

**Parser.** The single most important test is the replay-validation suite. A held-out fixed subset of BSW games (1,000 games in CI, 10,000 nightly, the full corpus weekly) is parsed and replayed through the engine; the engine's final `Ergebnis` must match Brettspielwelt's for at least 99.9 percent. Discrepancies are logged with game ID and surfaced. Game IDs with persistent failures are added to a known-bad list rather than blocking CI.

**Featurizer.** Purity test (same input produces byte-identical output across invocations and processes). Version pinning test (loading a checkpoint with a mismatched featurizer version raises).

**Action space.** Round-trip test (every action's canonical index decodes back to itself). Version pinning test analogous to the featurizer's.

**Agent interface.** A contract tester that runs every concrete `Agent` subclass through a full random game and asserts that every action it produces is legal in the engine's state at that step.

**Tournament harness.** A smoke test that runs `Random` vs `Rule` on a small deal pool and asserts the `Rule` agent wins by a positive margin (this would catch a broken tournament orchestrator, broken legality enforcement, or broken score accounting in one test). A determinism test that running the same config twice produces identical matrix entries.

**Inference service.** Integration tests using a tiny dummy model verifying request/response cycle, action legality, fallback path on malformed model output, and that featurizer-version mismatches surface as errors rather than silently corrupting predictions.

**Training pipeline.** A smoke test that runs one epoch of BC on a tiny synthetic dataset and asserts the training loss decreases. Not testing model quality — that is the eval harness's job — testing only that the pipeline does not crash.

**Prior art worth referencing.** OpenSpiel's test suite is a useful model for the shape of rules-engine tests in multi-agent games. The DouZero repository has examples of testing imitation-learning data pipelines at scale. RLcard has examples of tournament eval harnesses for card games.

## Out of Scope

The frontend (likely React) is a separate codebase and is not part of this PRD.

The game server — persistent game state, WebSocket fanout to clients, matchmaking, user accounts, authentication — is a separate codebase. It may be written in a different language; its only Python touchpoints are the engine (which it can either import directly or call over a thin RPC) and the inference service (which it calls over HTTP).

Online self-play training (PPO, IMPALA, A2C, etc.) is out of scope. At the data scale available, offline behavioral cloning plus offline RL is expected to outperform online self-play in both quality and infrastructure simplicity. Self-play remains a phase-2 option if the data-only approach plateaus.

The full DouZero-style rewrite of the policy network is phase 2. The architecture leaves clean room for it (the engine, parser, eval harness, and inference service all stay; only the model itself changes), but no DouZero-style training code is written in phase 1.

Tree-search agents (MCTS, ISMCTS, perfect-information Monte Carlo) are out of scope. The belief model is built in phase 1 because the data permits it; the search agent that consumes it is phase 2.

Mobile or on-device inference is out of scope. Phase 1 is server-side only.

Hyperparameter sweeps at massive scale, multi-GPU distributed training optimization, and custom CUDA kernels are all out of scope. Initial training uses sensible defaults on a single GPU.

Real-time training or online learning from live production games is out of scope. Phase 1 is fully offline.

Legal review of the Brettspielwelt corpus for commercial use is flagged but not handled here. It must be resolved before any paid product ships against models trained on this data, but it is a business question not a Python engineering question.

Production observability stack (log aggregation, alerting, tracing) beyond simple metric endpoints is out of scope. The inference service exposes Prometheus-style metrics; turning that into a production-grade observability deployment is platform-team work.

## Further Notes

**Suggested build phasing.** The dependency graph implies a natural sequence: rules engine and baseline agents first (validates the rules understanding); parser plus replay validation second (proves engine and parser agree with each other and with Brettspielwelt); TrueSkill computation third (cheap; produces an artifact reused everywhere); featurizer and action space fourth (the contract for everything ML-side); BC training fifth, starting small to validate the pipeline before scaling compute; eval harness sixth (must precede any model comparison claims); Tichu call heads seventh; offline RL eighth; belief model ninth (independent track, can be parallelized); inference service tenth. Rough estimate is eight to twelve engineer-weeks for one experienced developer, dominated by parser edge-case work and engine correctness work.

**Recency of the BSW corpus matters.** The dataset spans nineteen years. Meta-game has drifted (more aggressive Tichu calling in modern play, refined opening passes, etc.). The default training configuration weights recent games more heavily and caps the dataset at games post-2015; this should be validated empirically against alternatives (full corpus, hard cutoff at 2020, exponential decay) before being locked in.

**The belief-model opportunity is unusual.** Most card-game AI projects cannot train a belief model because they lack paired (public history, true hidden hands) data. BSW logs include all four hands post-game, which makes belief-model training a straightforward supervised problem at the corpus scale available. This is a strategic advantage worth investing in even if it does not improve phase-1 play directly.

**Phase 2 reuse.** A DouZero-style rewrite replaces the policy model (and changes the training loop to deep Monte Carlo with massively parallel actors), but everything else in this PRD — the engine, parser, TrueSkill table, eval harness, inference service, baseline agents — is preserved. Designing for that reuse now is mostly free: the contract between engine and learner is the featurizer plus action space, and both are already versioned.

**Compute reality.** Initial BC training is expected to fit on a single rented H100 for three to seven days. Total cloud cost for a complete phase-1 training run including ablations is in the low-thousands-of-dollars range, not the tens-of-thousands. This should not be a constraint on architecture choices.

**Inference cost.** A 30-million-parameter policy at the latency targets specified is comfortably servable on modest CPU. GPU inference is unnecessary at chess.com-scale traffic and would add operational complexity without payoff.

**Tichu calling behavior is a known failure mode.** Müller's PPO agent learned to never call Tichu because the expected value of random calling is negative against equally weak opponents. Behavioral cloning on the BSW corpus inherits human calling rates as a starting point, which avoids this collapse. The separate call networks plus skill-conditioning embedding are the defenses against the same failure recurring under offline RL refinement.

**Action space size.** Müller used about 1,415 actions in his simplification. The full game has a larger action space because of straight-flush bombs (suit-specific) and a richer set of straight lengths. Exact size is determined during enumeration but is expected to be in the range of 5,000 to 10,000 actions. This does not change the architecture — softmax heads scale linearly in size, and most actions are illegal in most states so the masked cross-entropy training cost is similar.

**Determinism is a load-bearing property.** Both the engine and the featurizer must be exactly deterministic. Non-determinism makes the replay-validation test flaky and corrupts the eval matrix. Every random source in the training pipeline (deal generation, data shuffling, model initialization) must be seeded and the seed must be in the run config.
