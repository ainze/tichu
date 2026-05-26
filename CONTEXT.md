# Tichu AI

Training and inference stack for a Tichu-playing AI. Learns from a corpus of human games (behavioural cloning), refines with offline RL, and serves predictions over HTTP. This document is the canonical glossary — every term here has exactly one meaning across the engine, training, eval, export, and inference layers.

## Language

### Game-layer terms

**Round**:
One full play-out from initial deal to scoring. Begins with `deal_initial_state`, ends when `_finalise_round` runs. Composed of an optional Grand-Tichu-call phase, a Schupfen, an optional Tichu-call phase, and a sequence of Tricks.
_Avoid_: deal, hand (as a noun for this unit).

**Game**:
A sequence of Rounds played to a target score (typically 1000 points). Not yet modelled in code, but the word is reserved — do not use "game" to mean a single Round.
_Avoid_: match, session.

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
The 16,568-float tensor produced by `featurize(PrivateState)`. Version-pinned via the featurizer version stamped on every Checkpoint.
_Avoid_: encoded state, observation, input, x, featurized state.

**Wire PrivateState**:
The JSON form of a PrivateState as transmitted to the inference HTTP endpoint, encoded by `private_state_to_json` and decoded by `private_state_from_json` in [tichu_inference/codec.py](src/tichu_inference/codec.py).
_Avoid_: request payload, blob, state JSON.

**Observation**:
Reserved word — not used in Phase 1. Will be introduced in Phase 2 (search / RL) for a state-plus-belief-distribution composite.

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
The pure function `featurize(PrivateState) -> Feature Vector`. Version-pinned (currently `"v1"`); the version is stamped on every Checkpoint.
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
The `V(state)` estimator ([tichu_training/awr/value_baseline.py](src/tichu_training/awr/value_baseline.py)) used by AWR to compute per-sample advantage. Trained separately before AWR Refine.
_Avoid_: critic, V-net.

**Sample Weight**:
The per-row positive scalar multiplied into the loss for each training example. In BC Training, derived from Skill Decile. In AWR Refine, multiplied by the AWR Weight.
_Avoid_: weight (bare), loss weight.

**AWR Weight**:
The advantage-derived multiplier `exp(β · A(s,a))` computed per BSW decision during AWR Refine. Combined with the BC Sample Weight to produce the final per-row weight.
_Avoid_: advantage weight, exp-adv.

**Belief Model**:
A standalone network predicting each opponent's remaining cards from public history. Trained on the BSW corpus using post-game-visible hidden hands as labels. **Not used in Phase 1 inference** — built now because the data scale supports it cheaply, and it bridges to Phase 2 search methods.

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
At serve time the ML Agent feeds the **Neutral Skill Decile** (not Decile 9). See [ADR-0005](docs/adr/0005-inference-time-skill-conditioning.md).

### Eval terms

**Starting Position**:
A single post-Schupfen `GameState` used as a Tournament starting point. Synthetic — produced by `deal_initial_state(seed + i)`. Schupfen is intentionally skipped so the Tournament measures play strength, not Schupfen heuristics.
_Avoid_: deal (noun), starting deal, hand (noun).

**Starting-Position Pool** (or "Pool"):
The fixed-seeded list of Starting Positions used by all Tournaments. Identity is exactly `(seed, n)` — reproducible across runs and machines.
_Avoid_: deal pool, deal set.

**Tournament**:
All-vs-all match orchestration over a Pool: every unordered pair of Agents plays every Starting Position twice (Seat-Swap), yielding per-pair score deltas with bootstrap 95% CI. Two variants: **Play-strength** (Schupfen skipped) and **Full-strength** (Schupfen played). See [ADR-0006](docs/adr/0006-tournament-play-strength-vs-full-strength.md).
_Avoid_: matrix run, eval matrix.

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
- `medium / hard / master` → **ML Agents** differing only by Checkpoint.
- Skill Conditioning is **always Neutral Skill Decile** at inference, independent of Difficulty (see [ADR-0005](docs/adr/0005-inference-time-skill-conditioning.md)).

## Example dialogue

> **Game designer:** "When the user picks `hard`, they get a stronger AI than `medium`, right?"
> **Engineer:** "Yes — but it's a different **Checkpoint**, not a different **Skill Decile** input. The **ML Agent** always feeds the **Neutral Skill Decile** at inference. `hard` and `master` differ in which Checkpoint they load — `master` is a **Refined Checkpoint**, `hard` is a **BC Checkpoint** of the same shape. They are byte-compatible because the **Policy Network**'s Trunk and Heads have the same dimensions."
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
- "observation" — resolved: reserved for Phase 2, banned in Phase 1.
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
- "schupfen is a BC head with `HEAD_LOGIT_DIMS['schupfen']=3`" — resolved: schupfen is a **standalone Schupfen Network**, same pattern as Call Networks. The 3-output stub in `bc/heads.py` is removed; the BC model now has three heads (play, wish, dragon_assignment). The `schupfen_00000.parquet` shard feeds `train_schupfen`, not `train_bc`. See [ADR-0012](docs/adr/0012-schupfen-is-a-standalone-network.md).
