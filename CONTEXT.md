# Tichu AI

Training and inference stack for a Tichu-playing AI. Learns from a corpus of human games (behavioural cloning), refines with offline RL, and serves predictions over HTTP. This document is the canonical glossary — every term here has exactly one meaning across the engine, training, eval, export, and inference layers.

## Language

### Game-layer terms

**Round**:
One full play-out from initial deal to scoring. Begins with `deal_initial_state`, ends when `_finalise_round` runs. Composed of an optional Grand-Tichu-call phase, a Schupfen, an optional Tichu-call phase, and a sequence of Tricks.
_Avoid_: deal, hand (as a noun for this unit).

**Game**:
A sequence of Rounds played to a target score of 1000 points. Modelled as `ParsedGame` (one per BSW `.tch` log). **Complete Game**: at least one team's cumulative `ergebnis` reaches ≥ 1000 — `game_won` and `game_outcome_margin` are defined for these. **Incomplete Session**: a `ParsedGame` whose summed scores never cross 1000 (abandoned mid-game) — game-level outcome columns are NULL and excluded from the AWR game-outcome target. Per-round records still flow through the BC pipeline regardless.
_Avoid_: match, session (except as the qualified term "Incomplete Session").

**Trick**:
The unit inside a Round, from one player leading a combination to everyone else passing (or to a winning bomb). Ends with cards going to the trick-winner's score pile.

**Play**:
A single action within a Trick — one Combination, by one player. Represented by `state.Play`.
_Avoid_: move, action (those mean other things — see Action terms).

**Hand**:
The cards a player currently holds (`frozenset[Card]` on `GameState.hands`). Always a noun for the current card-set; never a unit of time.
_Avoid_: using "hand" for the Round-unit.

**Deal**:
**Verb only** — "to deal the cards" at the start of a Round (`deal_initial_state`, `deal_for_schupfen`). Not a noun for the Round-unit.
_Avoid_: "a deal", "per deal", "deal-level".

### Decision terms

A player makes exactly **6 kinds of Decision** over a Round. Three are served by heads on the shared-trunk BC model; three are served by standalone networks.

**Decision**:
A player's choice-point — a moment at which the engine awaits an Action. Six types exist; see below.
_Avoid_: move, choice, action (Action means the chosen value, not the choice-point).

**Head**:
One output layer of the shared-trunk BC model, mapping the trunk's 512-dim representation to a Decision-specific action distribution. Four exist.
_Avoid_: branch, output (when referring to the BC model).

**Call Network**:
A standalone small network for a single Tichu / Grand-Tichu call Decision, not sharing the BC trunk. Two exist (Tichu, Grand-Tichu). See [ADR-0007](docs/adr/0007-calls-are-standalone-networks.md).
_Avoid_: call head (calls are not heads — they are separate networks).

**Schupfen Network**:
A standalone network for the Schupfen Decision, not sharing the BC trunk. Same precedent as Call Networks — carved out because the Decision shape (pick 3 of 14 cards, assign 3 labeled destinations) does not fit the single-discrete-output multi-head pattern. See [ADR-0012](docs/adr/0012-schupfen-is-a-standalone-network.md).
_Avoid_: schupfen head, pass head.

**The 6 Decisions** (canonical names, with network mapping):

| Decision | Served by | Action space |
|---|---|---|
| **Grand-Tichu Call** | Call Network | binary (call / skip) |
| **Schupfen** | Schupfen Network | 3 cards × 3 directions (to-next, to-partner, to-previous) |
| **Tichu Call** | Call Network | binary (call / skip) |
| **Play** | BC head `play` | 1809 intents, includes in-trick **Pass** as one slot |
| **Wish** | BC head `wish` | 14 ranks |
| **Dragon Assignment** | BC head `dragon_assignment` | 2 opponents (left / right) |

**Schupfen**:
The card-pass phase at the start of a Round. Each player gives one card to each of the other three players. Engine type is `SchupfenPass`. The Decision is which card goes in which direction.
_Avoid_: pass_card, card-pass, passing.

**Pass**:
The "I decline to play on this Trick" action. One slot inside the Play action space. Distinct from Schupfen.
_Avoid_: skip, fold.

**Wish**:
The rank a player wishes to be played next, declared when they play the Mahjong (rank 1). Holders of that rank must play it at their next legal opportunity.
_Avoid_: mahjong_wish, wish_rank.

**Dragon Assignment**:
When a player wins a Trick that contains the Dragon, they must give the entire Trick to an opponent (left or right). The Decision is which opponent.
_Avoid_: dragon_give, dragon_pass.

### State terms

**Rule:** "State" used bare is banned. Always qualify which of the three engine types, or use Feature Vector / Wire PrivateState.

**GameState**:
Engine ground truth — all four players' hands plus the PublicState. Held by the engine internally and by the eval `play_round` runner. Never serialised to the wire and never given to an Agent.
_Avoid_: world-state, full-state.

**PublicState**:
Common-knowledge slice — current player, hand sizes, team scores, current Trick, Tichu/Grand-Tichu callers, Mahjong Wish, and the PendingDecision slot. What every seat can observe.
_Avoid_: shared-state, world-view.

**PrivateState**:
One player's information set — their Hand plus a reference to the PublicState. The unit that the featurizer consumes and that the Agent interface receives.
_Avoid_: observation, view, perspective, info-set.

**PendingDecision**:
A field on PublicState that marks the engine as mid-resolving a special Decision (`DragonGivePending` | `MahjongWishPending` | `SchupfenPending`). Not a separate state type — a flag inside PublicState.

**Feature Vector**:
The fixed-shape float32 tensor produced by `featurize(PrivateState)`. Width is set by `FEATURIZER_OUTPUT_DIM` (currently 1,983 at v3; targeting 224 at v4 — see [ADR-0017](docs/adr/0017-featurizer-v4-compact-trick-top-combo.md)). Version-pinned via the featurizer version stamped on every Checkpoint.
_Avoid_: encoded state, observation, input, x, featurized state.

**Wire PrivateState**:
The JSON form of a PrivateState as transmitted to the inference HTTP endpoint, encoded by `private_state_to_json` and decoded by `private_state_from_json` in [tichu_inference/codec.py](src/tichu_inference/codec.py).
_Avoid_: request payload, blob, state JSON.

**Observation**:
The Phase-2 search root: a **PublicState** + the acting Player's **Hand** + the **Belief Model**'s opponent-card distribution over the unseen cards — the composite from which **Determinized Worlds** are sampled. Distinct from a **PrivateState**, which is the same Hand + PublicState but carries *no* belief distribution; the Observation is a PrivateState enriched with belief. See [ADR-0030](docs/adr/0030-phase-2-search-design.md).
_Avoid_: view, belief state, info-set, perspective.

### Action terms

The action vocabulary spans two layers — the engine's concrete (suit-aware) representation and the training/inference intent-level abstraction. The bridge between them is a heuristic Resolver. See [ADR-0002](docs/adr/0002-intent-level-action-space.md).

**Concrete Action**:
The engine's representation of a player's action, referencing specific card objects (e.g., the red 7 and the black 7). Type is `tichu_engine.legality.ConcreteAction`.
_Avoid_: engine action, low-level action, raw action.

**Intent**:
The training-layer's abstract representation of an action — "a pair of 7s with no phoenix" rather than naming suits. 1,809 of them total. Type is `tichu_training.action_space.Intent`.
_Avoid_: abstract action, decision, choice, head action.

**Action Space**:
The fixed, ordered catalogue of all 1,809 Intents. Version-pinned via Action Space Version on every Checkpoint.
_Avoid_: intent space, vocabulary, action set.

**Action Space Version**:
The string (currently `"v1"`) stamped onto every Checkpoint identifying the Action Space ordering used to train it. A mismatch on Checkpoint load is a hard error.

**Action Index**:
One slot in the Action Space — an integer in `[0, 1809)`. The output of an argmax over a Head's logits, before resolution.
_Avoid_: action id, slot.

**Combination**:
The engine's concrete representation of a legal multi-card play (Single, Pair, Triple, Full House, Pair-Step, Straight, Four-Bomb, Straight-Flush-Bomb). Always concrete — names specific cards. Engine-only term.
_Avoid_: combo, hand, play.

**Combination Intent**:
The intent-level peer of a Combination — `PlayPair`, `PlayStraight`, etc. Lives in the Action Space. Training/inference talks in Combination Intents, never Combinations.
_Avoid_: combo intent, abstract combo.

**Resolver**:
The function in `tichu_inference/ml_agent.py` that maps an Intent + a Hand → a Concrete Action by picking specific cards (which two 7s; which suit holds the Phoenix). The picks are heuristic — that heuristic is itself a design surface. The **inverse direction** (Concrete Action → Intent index, plus the legal Intent mask at a decision point) is mechanical, not heuristic, and lives in `tichu_training/action_space.py` next to `encode` / `decode`. The two directions are deliberately not co-located — inference needs the forward, training needs the inverse, and forcing a shared module would couple inference to engine internals it does not otherwise touch. See [ADR-0011](docs/adr/0011-bc-training-replay-on-the-fly.md).
_Avoid_: wrapper, mapper, decoder.

### Agent terms

**Agent**:
The interface every strategy implements (`Agent` ABC in `tichu_ml/agent.py`). One method: `act(PrivateState) -> Action`. The single integration point between the engine and any strategy.
_Avoid_: strategy, bot, player (Player is reserved for a seat 0–3).

**Player**:
A seat at the table, identified by index `0..3`. Belongs to a Team (0 or 1).
_Avoid_: agent, user (those mean other things).

**Baseline Agent**:
An Agent with no learned component — `RandomAgent`, `RuleAgent`. Hand-coded, deterministic given seed, no Checkpoint.
_Avoid_: rule bot, hand-coded agent.

**ML Agent**:
An Agent that loads a Checkpoint and runs `Policy Network → Action Index → Resolver → Concrete Action` per Decision (plus Call Networks for Tichu/Grand-Tichu calls). Defined in `tichu_inference/ml_agent.py`.
_Avoid_: learned agent, neural agent, model agent.

**Policy Network**:
The shared-trunk + three BC Heads (play, wish, dragon_assignment) that an ML Agent runs for those in-game Decisions. Specifically excludes the Call Networks and the Schupfen Network.
_Avoid_: bare "policy", model, net.

**Call Network**:
A standalone small network for a single call Decision (Tichu Call, Grand-Tichu Call). Two of them. Loaded by an ML Agent alongside the Policy Network.
_Avoid_: call head.

**Difficulty**:
The wire-level enum exposed by the inference service: `"easy" | "medium" | "hard" | "master"`. Each value maps to a specific Agent via the Difficulty Spec. See [ADR-0003](docs/adr/0003-easy-difficulty-is-a-baseline.md).
_Avoid_: level, mode, tier.

**Difficulty Spec**:
The inference-service config dict that maps each Difficulty to an Agent factory (Baseline or ML Agent + Checkpoint path). Loaded at startup; not editable at request time.
_Avoid_: agent map, levels config.

### Training pipeline terms

**Rules:** "Refinement" is banned bare — always **AWR Refine** (the stage) or **Refined Checkpoint** (the output). **Model** never means **Checkpoint** — a Model is in-memory, a Checkpoint is on-disk.

**Checkpoint**:
The versioned on-disk container ([tichu_training/checkpoint.py](src/tichu_training/checkpoint.py)). Stores `featurizer_version + action_space_version + payload bytes`. One file format; multiple payload kinds (BC, Refined, Call Network, Belief) share it. See [ADR-0004](docs/adr/0004-payload-agnostic-checkpoint.md).
_Avoid_: model file, weights, .bin.

**Model**:
The in-memory `torch.nn.Module` reconstructed from a Checkpoint's payload. Created by Load step: `Checkpoint → Model`.
_Avoid_: using "Model" for the file on disk.

**Featurizer**:
The pure function `featurize(PrivateState) -> Feature Vector`. Version-pinned (currently `"v3"`; v4 in design — see [ADR-0017](docs/adr/0017-featurizer-v4-compact-trick-top-combo.md)); the version is stamped on every Checkpoint.
_Avoid_: encoder, feature extractor.

**Trunk**:
The shared 4-residual-block MLP body of the Policy Network (see [ADR-0001](docs/adr/0001-trunk-architecture.md)). Consumed by every BC Head.
_Avoid_: backbone, body, base.

**BC** (Behavioural Cloning):
The supervised training stage on the BSW corpus. Output is a BC Checkpoint.
_Avoid_: imitation, supervised, pretraining.

**BC Training**:
The act of running BC. CLI: `train_bc`. **Replay-on-the-fly, archive-driven**: the trainer iterates the BSW archive in offset order, replays each round once, featurizes at every decision boundary, and yields training examples. The parquet shards are a per-decision manifest (validated `(game_id, round_id)` set + label / metadata columns), not the training input — Feature Vectors and legal Intent masks are not stored. See [ADR-0011](docs/adr/0011-bc-training-replay-on-the-fly.md).

**BC Checkpoint**:
A Checkpoint produced by BC Training. Payload = Trunk weights + four BC Head weights.

**AWR** (Advantage-Weighted Regression):
The offline-RL stage that re-weights BC samples by per-sample advantage and continues training from a BC Checkpoint.
_Avoid_: offline RL, fine-tune, RL refine.

**AWR Refine**:
The act of running AWR on a BC Checkpoint. CLI: `train_bc --refine-from <bc.bin>`.

**Refined Checkpoint**:
A Checkpoint produced by AWR Refine. **Byte-compatible with a BC Checkpoint** — same versions, same payload shape. ML Agents cannot tell a BC from a Refined Checkpoint; this is what makes Difficulty `hard`→`master` swaps drop-in.
_Avoid_: tuned model, refined model.

**Value Baseline**:
The `V(state)` estimator ([tichu_training/awr/value_baseline.py](src/tichu_training/awr/value_baseline.py)) used by AWR to compute per-sample advantage. Trained separately before AWR Refine against a **Value Target**.
_Avoid_: critic, V-net.

**Value Target**:
The per-row label the Value Baseline regresses against during AWR Refine. Two options live in the v3 parquet schema, selected by the `awr.value_target` config knob:
- `"round"` — fits V on `round_outcome` (per-round signed score delta, team-relative). The original v1/v2 target; high variance from slams + grand tichus.
- `"game"` — fits V on `game_won` (team-relative boolean: did the row's acting player's team win the **Complete Game**). Sparser per-game signal but lower variance and aligned with the actual objective. Rows from an **Incomplete Session** carry NULL `game_won` and are filtered out of the fit. See [ADR-0013](docs/adr/0013-parquet-schema-versioned-by-directory.md) for the schema-version mechanics.

Targets are stamped at parse time; AWR-time config picks which column to read. The Value Baseline architecture (hidden width, loss) may be tuned per target — `game_won` is a 0/1 target so BCE is more principled than MSE.

**Sample Weight**:
The per-row positive scalar multiplied into the loss for each training example. In BC Training, derived from Skill Decile. In AWR Refine, multiplied by the AWR Weight.
_Avoid_: weight (bare), loss weight.

**AWR Weight**:
The advantage-derived multiplier `exp(β · A(s,a))` computed per BSW decision during AWR Refine. Combined with the BC Sample Weight to produce the final per-row weight.
_Avoid_: advantage weight, exp-adv.

**Belief Model**:
A standalone network predicting each opponent's remaining cards from the acting Player's information set. Input is the same 224-dim **Feature Vector** the Policy Network consumes (`featurize(PrivateState)`); output is a `(3 opponents, 56 cards)` occupancy logit grid. Opponents are indexed in **relative-seat order** — next / partner / previous — the same convention as **Schupfen**, so the model is seat-invariant. Trained on the BSW corpus using the opponents' actual Hands at each sampled **Play** Decision as labels (these are visible at every decision during **Replay Validation**, not only post-game); the per-position **Belief Mask** excludes cards in the acting Player's own Hand or already played. **Not used in Phase 1 inference** — built now because the data scale supports it cheaply, and it bridges to Phase 2 search methods. The shipped `SyntheticBeliefDataset` is a smoke placeholder; the real replay-derived training source is materialised alongside the BC bundle (one parse pass, many outputs). **Input is being extended beyond the policy Feature Vector** ([ADR-0028](docs/adr/0028-belief-input-compressed-history-projections.md), proposed): the 224 policy features **+ a compact History block** of per-opponent projections — `declined_top` (max rank each opponent has Passed on, per Intent-type — the cross-Trick negative-information signal the compact policy featurizer discards), `lead_summary`, `pass_pressure`, optional `play_time`. Carried under its own `belief_input_version` so the policy **Featurizer** is untouched. Rationale: an explicit Belief Model only beats the policy's *implicit* belief if it sees signal the policy's input lacks; raw history is terabytes, the sufficient projection is ~tens of bytes. Shipped as an A / B-core / B-full ablation.

**BSW Corpus**:
The full set of `.tch` game logs scraped from brettspielwelt.de. The raw input to the training pipeline. Physically packaged as a zstd archive (see [tools/README.md](tools/README.md)) but logically a flat collection of ~2.4M Games keyed by `game_id`. The Held-out Game Set is a slice of this; everything else feeds BC Training and AWR Refine. Ingested into per-decision parquet shards by `parse_bsw` — see [ADR-0009](docs/adr/0009-bsw-ingest-streaming-pipeline.md).
_Avoid_: BSW data, BSW logs, the dataset.

**Replay Validation**:
The corpus-scale check that every parsed BSW round can be re-played action-by-action through the rules engine and reaches BSW's reported final scores. Granularity is the Round: only rounds whose engine-computed Ergebnis matches BSW's contribute training records. A game with at least one failing round is logged to `known_bad_games.txt` for monitoring, but its matching rounds still appear in the parquet shards. See [ADR-0008](docs/adr/0008-bsw-replay-validation.md).
_Avoid_: parse check, sanity check, score check.

**Call Network Checkpoint**:
A Checkpoint whose payload is one Call Network (Tichu or Grand-Tichu). Smaller payload than a BC Checkpoint. An ML Agent loads three Checkpoints total: one BC-or-Refined + one Tichu Call + one Grand-Tichu Call.

### Skill terms

**TrueSkill**:
Microsoft's rating algorithm. Run across the BSW corpus to produce per-handle `(mu, sigma, n_games)`. Not derived from explicit ratings — BSW does not publish them — but from game outcomes.
_Avoid_: rating system, MMR.

**Player Rating**:
The per-handle record: `(handle, mu, sigma, n_games)`. Output of the TrueSkill sweep. Games containing an Anonymous Seat, or where any seat's handle changes between rounds (mid-game player substitution), are excluded from the sweep — preserves per-identified-stable-game rating semantics. See [ADR-0010](docs/adr/0010-per-round-handles-with-asymmetric-tolerance.md).
_Avoid_: rating row, skill record.

**Anonymous Seat**:
A seat in a BSW round whose handle is missing in the `.tch` log — recorded as the empty string `""` in `ParsedRound.handles[seat]`. Arises from guest / freshly-joined accounts at the deal-moment of a player substitution, and from occasional BSW serialiser quirks that drop the handle on a single line. Excluded from the TrueSkill sweep and the Tichu Success Rate counter; included in BC training rows with `skill_decile = None` → **Neutral Skill Decile**. See [ADR-0010](docs/adr/0010-per-round-handles-with-asymmetric-tolerance.md).
_Avoid_: anonymous player, guest, unnamed seat.

**Min-games Filter**:
The eligibility cut applied to Player Ratings before quantile-slicing — drops handles with `n_games < threshold`. Prevents cold-start ratings from polluting the Skill Decile distribution.
_Avoid_: cold-start cut, eligibility filter.

**Skill Decile**:
Integer `0..9`, assigned by quantile-slicing `mu` across players who survived the Min-games Filter. 0 = bottom 10%, 9 = top 10%.
_Avoid_: skill bucket, rank, percentile, tier.

**Neutral Skill Decile**:
The sentinel 10th bucket (index `10`) for players who did not survive the Min-games Filter, OR for training-time records from an **Anonymous Seat**, OR for inference-time requests where the player's identity is unknown. Has its own learned Skill Embedding row.

**Skill Embedding**:
The learned vector indexed by Skill Decile (`0..10` — eleven rows total). Concatenated with the Feature Vector before the Trunk.
_Avoid_: skill vector, decile embedding.

**Skill Conditioning**:
The act of feeding Skill Decile into the model as input. **The model is conditioned on skill, not re-weighted by skill** — there is no Sample Weight derived from skill in BC Training. `Sample Weight` exists but is `1.0` until AWR Refine populates it with the AWR Weight.
_Avoid_: skill input, skill feature.

**Tichu Success Rate**:
Per-handle ratio of "Tichu called and won / Tichu called" across the corpus. Used to cross-validate TrueSkill (e.g., Spearman correlation between Skill Decile and Tichu Success Rate). Not fed to any Model.
_Avoid_: tichu rate, success ratio.

**Inference-time skill default**:
The **Neutral Skill Decile** is the *default* Skill Conditioning a tier receives when its **Difficulty Spec** sets no `skill_decile` ([ADR-0005](docs/adr/0005-inference-time-skill-conditioning.md)) — it is **not** forced on every tier. Serve-time Skill Decile is a per-tier lever, threaded through `MLAgent(skill_decile=...)` by the Difficulty Spec builder: `master` pins Decile 9 ([ADR-0024](docs/adr/0024-master-tier-conditions-on-top-skill-decile.md)), and the Docker serving spec maps `medium / hard / master` onto ascending Deciles of a single **Checkpoint** ([ADR-0027](docs/adr/0027-docker-serving-decile-mapped-single-bc-checkpoint.md)).

### Eval terms

**Starting Position**:
A single **pre-Schupfen, deal-time** `GameState` used as a Tournament starting point — all 56 cards dealt 14-per-player in **deal order**, `current_player = 0`, a `SchupfenPending` pending decision, no points or trick state. Produced by `deal_for_schupfen(seed + i)`. The deal order is preserved so the **Grand-Tichu Prefix** (each seat's first 8 cards) is a deterministic, genuine prefix of the same 14 cards that go on to Schupfen and Play in that round. Earlier the Tournament started post-Schupfen (`deal_initial_state`, Schupfen skipped) to isolate play strength; that variant is retired now that every Tournament is Full-strength.
_Avoid_: deal (noun), starting deal, hand (noun), post-schupfen position.

**Grand-Tichu Prefix**:
The first 8 cards (in deal order) of a seat's 14-card Starting-Position hand — the synthetic `(8,8,8,8)` deal-time state on which the Grand-Tichu Call is decided, matching the only hand size the Grand-Tichu Call Network ever saw in training. Not a detached hand: it is a real prefix of the 14 cards that seat then Schupfens and Plays in the same round. The 8/6 split point is fixed by deck-deal order (a shuffle has no other canonical split).
_Avoid_: pre-deal hand (that is the BSW-corpus term `pre_deal_hands`), first-eight.

**Starting-Position Pool** (or "Pool"):
The fixed-seeded list of Starting Positions used by all Tournaments. Identity is exactly `(seed, n)` — reproducible across runs and machines. Stores each seat's 14-card hand in deal order so the Grand-Tichu Prefix is reconstructible.
_Avoid_: deal pool, deal set.

**Tournament**:
All-vs-all match orchestration over a Pool: every unordered pair of Agents plays every Starting Position twice (Seat-Swap), yielding per-pair score deltas with bootstrap 95% CI. One variant only — **Full-strength**: the complete product stack (Grand-Tichu Call → Schupfen → Tichu Call → Play → Wish → Dragon), each Decision served by the Agent's own networks. (Historically there were two variants, Play-strength and Full-strength, per ADR-0006; the Play-strength variant — Schupfen skipped, calls declined — was retired in favour of measuring true end-to-end product strength. Calls, ±200 for Grand-Tichu, are the highest-variance Decisions and silently declining them is a systematic bias, not a neutral isolation.) See [ADR-0025](docs/adr/0025-full-strength-tournament-is-the-only-variant-and-includes-calls.md) (supersedes [ADR-0006](docs/adr/0006-tournament-play-strength-vs-full-strength.md)).
_Avoid_: matrix run, eval matrix, "play-strength tournament" (retired), "full-stack tournament" (the variant is **Full-strength**).

**Seat-Swap** (or "Seat-Swap Variance Reduction"):
The technique of playing each Starting Position twice between two Agents A and B — once with A in team-0 seats, once with B in team-0 seats — to cancel the team-assignment advantage of the Mahjong holder sitting at a fixed seat.
_Avoid_: seat-rotation, mirroring.

**Tournament Matrix**:
The Tournament's output table: per-pair mean score delta + bootstrap CI + observation count. The project's north-star quality metric.
_Avoid_: result matrix, score matrix.

**Move Prediction Eval**:
Held-out-BSW-games evaluation. For each decision the human made, score whether the Agent's ranked Actions agree with the human's actual move. Measures imitation fidelity (orthogonal to Tournament play strength).
_Avoid_: held-out eval, imitation eval, BSW eval.

**Top-k Accuracy**:
Fraction of held-out decisions where the human's actual move appears in the Agent's top-`k` ranked Actions.
_Avoid_: top-k score, agreement rate.

**Held-out Game Set**:
The slice of BSW games reserved from BC Training for Move Prediction Eval. Real games, not synthetic. Distinct from Starting-Position Pool.
_Avoid_: test set, eval set, held-out pool.

**Eval double-bind**: Every Checkpoint that ships passes **both** a Tournament row (synthetic, measures self-play strength) **and** a Move Prediction Eval row (real BSW games, measures imitation fidelity). Pool and Held-out Game Set serve different purposes and are not interchangeable.

## Relationships

### Game

- A **Game** consists of many **Rounds** played to a target score (typically 1000).
- A **Round** consists of optional Grand-Tichu Calls → **Schupfen** → optional Tichu Calls → a sequence of **Tricks** → scoring.
- A **Trick** consists of one or more **Plays** and **Passes** until everyone but the leader has Passed (or a Bomb wins out of turn).
- A **Player** occupies a seat `0..3` and belongs to a **Team** (0 or 1, partners sit across from each other).
- A **Hand** belongs to one Player and changes only via Schupfen (start of Round) and Plays (during Tricks).

### Decision → Network

- A **Play / Wish / Dragon-Assignment** Decision is served by a BC **Head** on the shared **Trunk**.
- A **Tichu Call / Grand-Tichu Call** Decision is served by a standalone **Call Network**.
- A **Schupfen** Decision is served by a standalone **Schupfen Network** (see [ADR-0012](docs/adr/0012-schupfen-is-a-standalone-network.md)).
- An **ML Agent** loads **one** Policy Network Checkpoint + **two** Call Network Checkpoints + **one** Schupfen Network Checkpoint.

### Action

- A **Concrete Action** (engine) is what the engine consumes via `step(state, action)`.
- An **Intent** (training/inference) is what a Head emits as an Action Index.
- The **Resolver** maps `Intent + Hand → Concrete Action`.
- The 1,809-entry **Action Space** is the canonical Intent catalogue; its **Action Space Version** is stamped on every Checkpoint.

### State

- **GameState** = ground truth (all four Hands + PublicState). Engine-only.
- **PrivateState** = one Player's Hand + the PublicState. The Agent sees this.
- **Feature Vector** = `featurize(PrivateState)`. The Policy Network sees this (plus the Skill Embedding from Skill Conditioning).
- **Wire PrivateState** = JSON encoding of PrivateState. Inference service sees this on the wire.

### Training

- **BC Training** consumes parquet shards (one per Decision type) → produces a **BC Checkpoint**.
- **AWR Refine** consumes a BC Checkpoint + parquet shards + a Value Baseline → produces a **Refined Checkpoint** (byte-compatible with the BC Checkpoint).
- **Skill Decile** is computed once per player from **TrueSkill** Player Ratings; it feeds into BC examples and AWR examples as Skill Conditioning input (never as Sample Weight in BC).
- **Sample Weight** is `1.0` in BC; **AWR Refine** populates it with **AWR Weight** = `exp(β · advantage)`.

### Eval

- A **Tournament** is run over a **Starting-Position Pool** producing a **Tournament Matrix** via **Seat-Swap**.
- A **Move Prediction Eval** is run over the **Held-out Game Set** producing **Top-k Accuracy** rows.
- Every shipped Checkpoint has rows in **both** Tables.

### Difficulty

- **Difficulty** is a wire-level enum mapped via the **Difficulty Spec** to concrete Agents.
- `easy` → **RuleAgent** Baseline (not a weak ML Agent — see [ADR-0003](docs/adr/0003-easy-difficulty-is-a-baseline.md)).
- `medium / hard / master` → **ML Agents** that may differ by **Checkpoint and/or Skill Decile conditioning**. Two valid shapes exist: differ-by-Checkpoint (e.g. `hard`=BC, `master`=Refined) and differ-by-Decile (one **Checkpoint**, tiers pinned to ascending Skill Deciles — the Docker serving image, [ADR-0027](docs/adr/0027-docker-serving-decile-mapped-single-bc-checkpoint.md)).
- Serve-time **Skill Decile** is a per-tier lever set in the **Difficulty Spec**; the **Neutral Skill Decile** is only the default when a tier leaves it unset (see [ADR-0005](docs/adr/0005-inference-time-skill-conditioning.md), [ADR-0024](docs/adr/0024-master-tier-conditions-on-top-skill-decile.md)).

### Phase 2 / online-RL terms

**Master-level (goal)**:
The operational success bar for Phase 2 play strength: an Agent that (a) strongly dominates the entire BC/AWR agent zoo on the **Tournament Matrix** and (b) is no longer beatable by a strong human (the project owner) in direct play. Genuinely superhuman, DouZero-style — by construction unreachable by any offline method (BC caps at imitating top-decile humans; **AWR Refine** caps at re-weighting within **BSW Corpus** state-support). The bar that motivates leaving the offline regime for online self-play. Measurement instruments: the **Tournament Matrix** (high-throughput, self-relative) and owner human play-test (low-throughput, the only "exceeds-human" signal available).
_Avoid_: "master tier" (that is the **Difficulty** enum value, a serve-time Decile-9 BC config — not the strength goal).

**PPO Refine** (proposed):
The online-RL stage that **sharpens** a BC Checkpoint by on-policy self-play through the rules engine. PPO actor-critic, **warm-started from the BC Checkpoint**, with a **KL-anchor** to the frozen BC policy + an entropy bonus, so the policy stays human-plausible (does not crater the **Move-Prediction Eval**) while sharpening toward winning play. Chosen over DouZero-style **DMC** — the precedent ADR-0001 cites — because the goal is to sharpen an already-strong policy on single-machine compute, not to learn from scratch at cluster scale; and over AlphaZero-style search, which is Phase 2. Output is a **Sharpened Checkpoint**. Distinct from **AWR Refine** (offline, corpus-bounded, came back flat — see [docs/notes/2026-05-29-awr-game-target-flat.md]). The full locked design (scope, reward, opponent structure, single-process vectorized rollout, separate critic, hyperparam levers, kill-criteria) is [ADR-0029](docs/adr/0029-ppo-refine-design.md).
_Avoid_: RL refine, fine-tune, online refine, "the PPO stage" (use **PPO Refine**).

**Sharpened Checkpoint** (proposed):
A Checkpoint produced by **PPO Refine**. Byte-compatible with a BC / Refined Checkpoint (same Trunk + Heads payload shape), so it is a drop-in **Difficulty**-tier swap — same property that makes BC↔Refined interchangeable.
_Avoid_: PPO checkpoint, RL model.

**Behavioral Telemetry**:
The per-Agent behavioral-rate panel — *how* an Agent plays, orthogonal to the strength the **Tournament Matrix** measures. Metrics: `bomb_per_round`, `bomb_when_legal_rate` (of Play decisions where a Bomb was legal, the fraction that Bombed — the headline soft-policy diagnostic), `tichu_call_rate` / `tichu_success_rate`, `grand_call_rate` / `grand_success_rate`, `trick_win_rate` (neutral 0.25 at a homogeneous 4-seat table), `out_first_rate` (neutral 0.25), `slam_rate`, `caller_passivity_rate` (as a Tichu/GT caller following an opponent with a legal beat, the fraction of tricks ceded by Passing) and its `caller_bomb_passivity_rate` subset (beat was a Bomb — a near-pure error rate). The caller-passivity metrics quantified the play-policy's bomb/strong-combo undervaluation — see [docs/notes/2026-06-03-caller-passivity-bomb-undervaluation.md]. Captured per-Round per-seat by `play_full_round(..., collect_telemetry=True)` (a zero-overhead opt-in side-channel; the Tournament hot path leaves it off), aggregated in self-play by [tichu_eval/behavioral.py](src/tichu_eval/behavioral.py), exposed as `eval_matrix --mode behavioral` ([configs/eval_behavioral_v5.yaml](configs/eval_behavioral_v5.yaml)). The engine surfaces the trick winner via the previously-unused `step` `info` dict (`{"trick_winner", "trick_points"}`) — also the reward-shaping hook for **PPO Refine**. Built first as the diagnostic instrument for the soft-policy / never-bombs hypothesis, reused later as the PPO Refine tuning dashboard (RL is tuned by watching behavior drift, not win-rate alone).
_Avoid_: eval stats, metrics dump.

### Phase 2 / search terms

**Rule:** "search" used bare for the Phase-2 lever is banned — it is **PIMC** (the algorithm) producing a **Search Agent** (the strategy). All terms below are proposed in [ADR-0030](docs/adr/0030-phase-2-search-design.md); v1 is a latency-free offline strength experiment, not a served tier.

**PIMC** (Perfect-Information Monte-Carlo search):
The Phase-2 algorithm. From an **Observation**, sample K **Determinized Worlds** from the **Belief Model**, run an independent **PUCT**-MCTS to completion in each, sum root visit counts across worlds, and pick the argmax **Intent**. Chosen over single-tree ISMCTS for v1 (simpler, parallelizes onto the [ADR-0026] process pool, consumes the Belief Model as designed); its strategy-fusion weakness is the documented ISMCTS escalation. Runs on **frozen** nets — no learning in v1 (AlphaZero-style self-play training is a later phase).
_Avoid_: MCTS (bare), AlphaZero, tree search, lookahead.

**Determinized World**:
A perfect-information `GameState` sampled from an **Observation** — the acting Player's real **Hand** plus a constraint-respecting assignment of every unseen card to the three opponents (honoring their known `hand_sizes`, shown voids, and the Mahjong Wish obligation), drawn to match the **Belief Model**'s per-card marginals. The unit a single MCTS tree searches.
_Avoid_: world, sample, hypothesis, particle, determinization (use the noun "Determinized World").

**Search Agent**:
An **Agent** whose `act` runs **PIMC** over the frozen `master` Policy Network (the **PUCT** prior), Belief Model (the **Determinized World** sampler), and rules engine (the simulator). The opponents and partner are simulated *in-tree* by the frozen `master` policy as environment dynamics — only the root seat's own Decisions branch. Leaf value is a **Leaf Rollout**. The strategy v1 measures against `master`.
_Avoid_: MCTS agent, search bot, PIMC agent (use **Search Agent**).

**Leaf Rollout**:
The v1 leaf evaluator: from a tree leaf, continue the frozen-`master` policy-driven simulation of the **Determinized World** to round-terminal (`_finalise_round`) and back up the actual `round_outcome` (team-relative, includes the ±100/±200 call bonus). Unbiased within the world; trusts only the engine and the `master` policy, not the off-distribution critic. The frozen critic / **Value Baseline** as a depth-truncated bootstrap is the documented variance-reduction escalation, not the v1 default.
_Avoid_: rollout (bare), playout, leaf eval, value bootstrap.

## Example dialogue

> **Game designer:** "When the user picks `hard`, they get a stronger AI than `medium`, right?"
> **Engineer:** "Yes — the gap comes from one of two levers the **Difficulty Spec** controls. Either a different **Checkpoint** (`hard`=**BC Checkpoint**, `master`=**Refined Checkpoint** — byte-compatible, same Trunk/Heads), or the *same* Checkpoint conditioned on a higher **Skill Decile** (e.g. `medium / hard / master` → Decile 3 / 6 / 9). It is *not* pinned to the **Neutral Skill Decile** — that's only the default when a tier sets no Decile."
> **Game designer:** "And `easy` is the worst-trained model?"
> **Engineer:** "No — `easy` is the **RuleAgent Baseline**. No Checkpoint at all. A weak ML model is *unpredictably* bad; a Baseline is *legibly* weak, which is more useful for new players."
>
> ---
>
> **Reviewer:** "Why does the BC Head output 1,809 things? There are only ~14 ranks."
> **Engineer:** "That's the **Action Space** size — the number of distinct **Intents** a player can express. One Intent per (combination-shape, rank, Phoenix-position) tuple. The **Resolver** picks specific suits afterwards. See [ADR-0002](docs/adr/0002-intent-level-action-space.md)."
>
> ---
>
> **New dev:** "The eval module talks about a deal pool — I thought 'deal' was a verb?"
> **Engineer:** "It is. The eval module was mis-named; the noun is **Starting Position**, the collection is the **Starting-Position Pool**. The rename landed in session 2026-05-25 — the module is now `tichu_eval/starting_position_pool.py` and the runner is `play_round()`."

### Flagged ambiguities

- "deal" used as both verb and noun (`play_deal()` named the Round-unit) — resolved: Deal is verb-only; the Round-unit is **Round**. `play_deal()` renamed to `play_round()` in session 2026-05-25.
- "hand" used for both cards-held and Round-unit — resolved: Hand is cards-held only.
- "round" vs "deal" vs "hand" — resolved: **Round**.
- "pass" used for both in-trick decline and pre-Round card exchange — resolved: **Pass** is in-trick only; pre-Round card exchange is **Schupfen**.
- `pass_card` / `wish_rank` / `dragon_give` (parquet shard names + BC head labels) — resolved: canonical names are **schupfen** / **wish** / **dragon_assignment**. Renames done in session 2026-05-25.
- "state" used bare across engine, training, and inference — resolved: always qualify (GameState / PublicState / PrivateState / Feature Vector / Wire PrivateState).
- "observation" — resolved: banned in Phase 1; **introduced in Phase 2** as the **PIMC** search root (a **PrivateState** enriched with the **Belief Model**'s opponent-card distribution, from which **Determinized Worlds** are sampled). See [ADR-0030](docs/adr/0030-phase-2-search-design.md).
- Type name `Action` exported by both `tichu_engine.legality` and `tichu_training.action_space` — resolved: canonical names are **ConcreteAction** and **Intent**. Renames done in session 2026-05-25.
- "combination" vs "intent" — resolved: **Combination** is engine-only (concrete cards); **Combination Intent** (or the specific intent class like `PlayPair`) is training/inference.
- "policy" used to mean both "the Agent as a function" and "the neural network inside an ML Agent" — resolved: bare **policy** is banned; the network is the **Policy Network** (in-game heads) or a **Call Network** (Tichu/Grand-Tichu).
- "player" used to mean both "an Agent" and "a seat at the table" — resolved: **Player** is a seat 0–3 only.
- "refinement" / "offline RL" / "AWR" used interchangeably for the same stage — resolved: stage is **AWR Refine**, output is a **Refined Checkpoint**, bare "refinement" is banned.
- "model" used for both the in-memory `nn.Module` and the on-disk file — resolved: **Model** is in-memory only; the file is the **Checkpoint**.
- "checkpoint" used to imply payload kind (e.g., "this is a BC checkpoint") even though the file format does not encode it — resolved: Checkpoint kind is determined by the loader's expectation, not the file itself. See [ADR-0004](docs/adr/0004-payload-agnostic-checkpoint.md).
- "skill weight" vs "skill conditioning" — resolved: the model is **conditioned** on Skill Decile via the Skill Embedding; **Sample Weight is unrelated to skill** in BC Training (always `1.0`).
- "deal" used as a noun in `tichu_eval` for "the synthetic starting position" — resolved: **Starting Position** is the noun; "deal" stays verb-only per §Game-layer terms. Module renamed to `tichu_eval/starting_position_pool.py` and `play_deal()` → `play_round()` in session 2026-05-25.
- "player handle" treated as game-stable (one tuple per `ParsedGame`) — resolved: handles are **per-round** (`ParsedRound.handles`), not per-game. Captures BSW mid-game player substitutions and **Anonymous Seats** correctly. `ParsedGame.handles` removed in session 2026-05-26. BC ingestion tolerates anonymous / substituted seats (Neutral Skill Decile); TrueSkill ingestion rejects them (per-identified-stable-game invariant). See [ADR-0010](docs/adr/0010-per-round-handles-with-asymmetric-tolerance.md).
- "parquet shards are the BC training input" — resolved: **the BSW archive is the training input**; parquet shards are a per-decision manifest carrying labels, `sample_weight`, `round_outcome`, and `player_handle`. Feature Vector and legal Intent mask are computed at train time by replaying the round from the archive. The schema's reserved `state` / `legal_actions_mask` / `skill_decile` columns are dead weight in v1 and slated for removal. Skill Decile is joined live against the TrueSkill ratings table by `player_handle`. See [ADR-0011](docs/adr/0011-bc-training-replay-on-the-fly.md).
- "parquet schema version is the same as featurizer version" — resolved: they are independent. `featurizer_version` versions the **Featurizer** (`featurize` function output) and nothing else. Parquet schema additions (e.g., `game_won` in v3) bump the **directory suffix** (`parquet_<scale>_v<N>`); featurizer_version is left untouched. Readers detect schema features by column-presence, not by string comparison. See [ADR-0013](docs/adr/0013-parquet-schema-versioned-by-directory.md).
- "game" used loosely for either a `ParsedGame` or a Tichu Game-to-1000 — resolved: a **Game** is a sequence of Rounds played to 1000; a `ParsedGame` is a Game iff at least one team's cumulative `ergebnis` reaches ≥ 1000 (a **Complete Game**). The remainder are **Incomplete Sessions** — included in BC labels (round-level data is intact) but excluded from the `"game"` Value Target.
- "schupfen is a BC head with `HEAD_LOGIT_DIMS['schupfen']=3`" — resolved: schupfen is a **standalone Schupfen Network**, same pattern as Call Networks. The 3-output stub in `bc/heads.py` is removed; the BC model now has three heads (play, wish, dragon_assignment). The `schupfen_00000.parquet` shard feeds `train_schupfen`, not `train_bc`. See [ADR-0012](docs/adr/0012-schupfen-is-a-standalone-network.md).
- "version" used loosely for both Featurizer Version and a model's training-run identity — resolved: a Tournament agent may mix networks from **different training runs / scales / epochs / checkpoints** freely, but **only at a single Featurizer Version**. Every export an MLAgent loads is asserted against the harness's global `FEATURIZER_VERSION` (`load_exported`), so featurizer-v3 and featurizer-v5 exports cannot coexist in one process — cross-Featurizer-Version comparison is two separate runs, not one matrix. See [ADR-0025](docs/adr/0025-full-strength-tournament-is-the-only-variant-and-includes-calls.md).
- "Tichu Call sees no Trick or public-play history" (ADR-0007 §rationale 1) — resolved: that's the upper bound at deal-time, not the actual featurise state. **Tichu Call featurises at the seat's first non-Pass Play state** (per-seat, symmetric across positives and negatives). May include up to ~3 prior Plays of public history. Grand-Tichu still featurises at the synthetic deal-time 8-card state. See [ADR-0018](docs/adr/0018-tichu-call-featurises-at-first-non-pass-play.md).
