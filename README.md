# Tichu AI

This is a complete training and inference stack for a card-playing AI for the four-player partnership card game **Tichu**. It learns by watching humans play (behavioral cloning), refines its play using game outcomes (offline RL), and serves predictions over a small HTTP service.

This README is written for someone who isn't an expert in machine learning or in the game. It explains what each piece does, in what order to run things, and what to expect at each step.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [How the pieces fit together](#how-the-pieces-fit-together)
3. [Installation](#installation)
4. [The command-line tools](#the-command-line-tools)
5. [End-to-end training pipeline](#end-to-end-training-pipeline)
6. [The inference HTTP service](#the-inference-http-service)
7. [Project layout](#project-layout)
8. [Running the tests](#running-the-tests)
9. [Known limitations and deferred work](#known-limitations-and-deferred-work)

---

## What this project does

**Tichu** is a partnership trick-taking card game played with a 56-card deck (the standard 52 cards plus four special cards: Mahjong, Dog, Phoenix, Dragon). Two players sit across from each other and play as a team against the other two players. A round involves passing cards before play (called **schupfen**), optionally announcing a "Tichu" bid that doubles or halves your score depending on whether you go out first, and then playing combinations of cards until three of the four players have emptied their hands.

The project does three things:

1. **Learns to play** by watching tens of thousands of recorded games from a real online Tichu server (BSW), then trains neural networks that imitate the human players.
2. **Improves on what it learned** by reweighting good outcomes more heavily than bad ones (offline reinforcement learning).
3. **Serves predictions** over a small HTTP endpoint so a game server can ask the AI "what should this player do?" and get back a legal action.

The code is organized as a series of stand-alone command-line tools that each do one job (parse, train, evaluate, export, serve). You run them in order; each tool reads what the previous one wrote.

---

## How the pieces fit together

Here is the entire flow, from raw human game logs on the left to a running inference server on the right:

```
.tch files (BSW logs)
        |
        |  tools/compress.py
        v
archive.zst + archive.zst.idx
   (indexed-blob archive of all .tch payloads —
    see ADR-0009)
        |
        +------------------ compute_trueskill ------------------+
        |                                                       v
        |                                            TrueSkill ratings parquet
        |                                            (player_handle, mu, sigma,
        |                                             n_games, skill_decile)
        |                                                       |
        |  parse_bsw                                            |
        v                                                       |
Parquet manifest                                                |
  (one shard per decision type:                                 |
   play / schupfen / wish /                                     |
   dragon_assignment / call_tichu /                             |
   call_grand_tichu)                                            |
                                                                |
   Manifest only — Feature Vectors and legal-Intent masks       |
   are computed at training time by replaying each round        |
   from the archive (ADR-0011: replay-on-the-fly).              |
        |                                                       |
        +--------+--------------+-------------+-----------------+
                 |              |             |                 |
                 v              v             v                 v
              train_bc     train_calls   train_schupfen     (live join
              (play / wish /  (tichu /    (standalone        of skill_decile
              dragon-assn)    grand-tichu) network —          by player_handle)
                              call nets)  ADR-0012; deferred)
                 |              |
                 v              v
        BC checkpoint     Tichu / Grand Tichu
        (.bin, version-   call-network checkpoints
         pinned)
                 |              |
                 |  train_bc --refine-from <bc.bin>
                 v              |
        AWR checkpoint           |
        (same .bin format,       |
         drop-in for BC)         |
                 |              |
                 |  eval_matrix  |
                 +-----+--------+
                       v
        Tournament matrix (Parquet) +
        held-out move-prediction CSV
                       |
                       |  export_model
                       v
               TorchScript artifacts
               (policy.pt, tichu_call.pt,
                grand_tichu_call.pt)
                       |
                       |  serve_inference
                       v
          HTTP server: POST /act, GET /health,
                      GET /metrics
```

Several things are worth understanding at this point:

- **Checkpoints are version-pinned.** Every checkpoint file stamps the versions of two things: the **featurizer** (the code that turns a game state into a numeric vector) and the **action space** (the canonical list of 1,809 possible actions a player can take). If you change either, old checkpoints won't load — you get a clear error instead of silently broken predictions.
- **The action space is "intent-level".** Instead of a separate index for every concrete play (e.g. "the pair of red 7s"), there is one index for the intent ("a pair of 7s"). The inference layer maps the model's chosen intent onto the specific cards in the player's hand.
- **Parquet is a manifest, not a dataset.** The shards carry the validated `(game_id, round_id)` set plus label / metadata columns (`action_taken`, `round_outcome`, `player_handle`, `sample_weight`). Feature Vectors and legal Intent masks are recomputed at training time by replaying each round from the BSW archive — no on-disk feature cache. See [ADR-0011](docs/adr/0011-bc-training-replay-on-the-fly.md). Skill Decile is joined live from the ratings parquet by `player_handle` — no need to re-emit parquet shards after computing ratings.
- **Decisions → networks split.** Three of the six decision types are served by BC heads on the shared trunk (`play`, `wish`, `dragon_assignment`). The other three are standalone networks: Tichu Call and Grand-Tichu Call ([ADR-0007](docs/adr/0007-calls-are-standalone-networks.md)) plus Schupfen ([ADR-0012](docs/adr/0012-schupfen-is-a-standalone-network.md)). An ML Agent at inference time loads four checkpoints total: BC + two Call Networks + one Schupfen Network.

---

## Installation

The project requires **Python 3.11 or newer**.

```bash
# From the repo root:
python -m pip install -e ".[ml,inference,dev]"
```

The three extras pull in optional dependencies:

- `ml` — PyTorch (needed for all training and any model loading).
- `inference` — FastAPI + uvicorn (needed only if you want to run the HTTP service).
- `dev` — pytest, hypothesis, pip-tools (needed only if you want to run the tests).

If you only want to run the inference service against a pre-trained artifact, install just `[inference]`. If you want to retrain from scratch you need `[ml]` too.

---

## The command-line tools

After installing the package, eight tools become available on your PATH. Each is described below. Every one of them accepts `--help` for the full flag list.

### `parse_bsw`

**What it does:** Reads `.tch` files (the format that the BSW online Tichu server uses to log games), parses them into structured records, replays each round through the engine to validate it (see [ADR-0008](docs/adr/0008-bsw-replay-validation.md)), and writes the validated records out as Parquet shards — one shard per decision type. Each row represents one decision a human made: which card to play, which direction to send a schupfen card, what rank to wish for, etc. The pipeline is fully streaming — bounded memory regardless of corpus size (see [ADR-0009](docs/adr/0009-bsw-ingest-streaming-pipeline.md)).

**When you'd run it:** Once at the start of your pipeline, against either a directory of raw `.tch` files or the indexed-blob archive produced by `tools/compress.py`.

```bash
# Directory of loose .tch files:
parse_bsw \
  --input   data/bsw/                       # directory of .tch files
  --output  artifacts/parquet/              # directory for output shards
  --workers 12                              # parallel replay workers
  --recency-cutoff-game-id 1855844          # the first 2015 game id
  --recency-weight 0.5                      # downweight pre-2015 games by 0.5

# Or the indexed-blob archive (preferred at corpus scale):
parse_bsw \
  --archive data/archive.zst                # zstd archive + .zst.idx sidecar
  --output  artifacts/parquet/
  --subset  100000                          # first N games by sorted game_id
  --workers 12
```

The recency flags exist because Tichu's strategic landscape has shifted over the years; downweighting older games stops the model from over-fitting to obsolete play.

`--trueskill RATINGS.parquet` is also accepted and stamps `skill_decile` into the rows, but it is **no longer required** for BC training — the trainer joins ratings live by `player_handle` (see ADR-0011). The flag is kept for spot-checking and legacy callers.

**Outputs:** Six parquet shards under the output dir — one per decision type (`play`, `schupfen`, `wish`, `dragon_assignment`, `call_tichu`, `call_grand_tichu`). Plus `known_bad_games.txt` (games with at least one replay-failed round), `parse_failures.txt` (games whose `.tch` couldn't tokenize), and `failure_details.tsv` (per-failed-round category + detail).

---

### `compute_trueskill`

**What it does:** Streams through the BSW archive and computes a [TrueSkill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) rating for every player handle. TrueSkill is the rating system Xbox Live uses — it's like Elo, but it understands team games and represents uncertainty about a player's true skill. The pipeline is a single fused streaming pass (parse → TrueSkill update → call-stats update → discard, per game) so memory stays bounded by the unique-handles dicts regardless of corpus size. The output is a Parquet file with `(player_handle, mu, sigma, n_games, skill_decile, tichu_calls, tichu_success_rate, ...)` columns. The skill decile is fed into the training pipeline so the model conditions on "what would a strong player do here?" specifically (see [ADR-0005](docs/adr/0005-inference-time-skill-conditioning.md) and [ADR-0010](docs/adr/0010-per-round-handles-with-asymmetric-tolerance.md)).

**When you'd run it:** Independently of `parse_bsw` — the trainer joins ratings live, so order doesn't matter. Typically before training.

```bash
# Against the indexed-blob archive (preferred at corpus scale):
compute_trueskill \
  --input    data/archive.zst                 # .zst + .zst.idx archive
  --output   artifacts/ratings.parquet
  --subset   100000                           # match the parse_bsw --subset for a smoke run
  --min-games 20                              # exclude players with fewer than 20 games

# Or against a directory of .tch files (also accepts .tar.zst legacy archives):
compute_trueskill \
  --input  data/bsw/
  --output artifacts/ratings.parquet
  --min-games 20
```

Reports a `decile share` distribution and a `Spearman ρ(mu, tichu_success_rate)` cross-validation at the end — positive rho means higher TrueSkill correlates with higher Tichu success rate, the expected signal.

**Outputs:** A Parquet file mapping each handle to a TrueSkill `(mu, sigma)`, an integer skill decile in `[0, 9]`, and per-handle Tichu / Grand-Tichu call success counts.

---

### `train_bc`

**What it does:** Trains the main policy network using **behavioral cloning** — i.e. it learns to predict what a human did, given the same game state. The network has a shared "trunk" (a deep residual MLP) feeding **three heads** — one per BC-served decision type (`play`, `wish`, `dragon_assignment`). Schupfen has its own standalone network per [ADR-0012](docs/adr/0012-schupfen-is-a-standalone-network.md); the Tichu / Grand-Tichu call decisions have theirs per [ADR-0007](docs/adr/0007-calls-are-standalone-networks.md). The BC trunk also conditions on the acting player's skill decile so it can learn "what would a strong player do here?" specifically.

The trainer reads from the BSW archive directly and replays each round on the fly — the parquet shards are just a validity manifest plus label / metadata columns. See [ADR-0011](docs/adr/0011-bc-training-replay-on-the-fly.md).

**When you'd run it:** After `parse_bsw` has produced the parquet manifest and `compute_trueskill` has produced the ratings table.

```bash
# Smoke run against the 100k parquet manifest:
train_bc \
  --config       configs/bc_smoke_parquet.yaml \
  --run-dir      runs/bc/smoke_100k \
  --workers      11                            # parallel data-loader processes
  --device       auto                          # cuda if available, else cpu
  --max-examples 200000                        # cap epoch length for fast smoke
```

```bash
# AWR offline-RL refinement (streaming — works on the full 2.4M corpus):
train_bc --config configs/awr_smoke_100k.yaml --run-dir runs/awr/smoke_100k \
         --refine-from runs/bc/smoke_100k/checkpoints/step_e000_b00020000.bin \
         --device cuda --workers 11
```

Flag highlights:

- `--workers N` spawns N data-loader processes that shard the manifest by `hash(game_id)`. Default 0 = sequential. **The single-threaded Python data loop is the throughput bottleneck on small-model configs**; with N workers, CPU utilisation scales toward 100% × N cores.
- `--device {auto,cpu,cuda}` — where the model runs. `auto` picks CUDA if `torch.cuda.is_available()`, else CPU. Useful for the large-model `bc_full.yaml` config. Note: the default `pip install torch` on Windows pulls the CPU-only wheel; install via `pip install torch --index-url https://download.pytorch.org/whl/cu{NNN}` to enable GPU.
- `--max-examples N` caps each epoch to the first N yielded BCExamples — useful for smoking the full 100k/2.4M manifest without committing to a full sweep. In AWR streaming mode the cap applies per stream pass (i.e. per epoch).

Two config keys worth knowing about for long runs:

- `checkpoint_every_batches: N` — saves a checkpoint every N fired batches inside an epoch. Without this, BC only saves at end-of-epoch, which is hours on the full corpus. With `N=20000` you get ~hourly mid-epoch checkpoints suitable for `--resume` or `--refine-from`.
- `awr.streaming: true` — selects the streaming AWR pipeline. Required for any AWR run beyond ~5M examples; the in-memory path holds every BCExample in RAM at ~66 KB each. See AWR section below.

#### AWR refinement modes

AWR has two implementations behind `--refine-from`, selected by `awr.streaming` in the config.

**In-memory mode (`awr.streaming: false`, the default)** — materialises the whole training set into RAM once, fits a value baseline on `round_outcome`, computes per-example **advantage weights** (good outcomes get a bigger weight, bad outcomes get a smaller weight), and trains the same network for more epochs with those weights applied. Simple and fast for ≤ a few million examples; OOMs on anything larger.

**Streaming mode (`awr.streaming: true`)** — never materialises the dataset:

1. **Held-out scan**: one pass through the stream collecting a capped subset (`max_held_out`, default 20k) via a deterministic hash predicate. The same predicate excludes those examples from training on every subsequent pass.
2. **Baseline fit**: one streaming SGD pass over the train partition with multiple SGD steps per chunk.
3. **Refine epochs**: one fresh stream pass per AWR epoch. Per chunk: baseline forward → advantages → per-chunk AWR weights → distribute to per-head buffers → fire weighted-BC step when each buffer fills.

Two subtleties of the streaming path:

- **Advantage standardisation** (`awr.standardize_advantages: true`, default in the smoke config). Tichu's `round_outcome` ranges ±200 points; the AWR paper's beta is calibrated against unit-variance advantages. Without standardisation the per-chunk max-subtract collapses every non-max weight to ~0 and the model effectively trains on one example per chunk. Standardisation makes `beta: 1.0` behave as the paper intends.
- **Mid-epoch eval** (`awr.eval_every_chunks: N > 0`). Every N chunks the held-out subset is forward-passed for an intermediate `win_rate_proxy`, written as a row in `epoch.csv` with a populated `chunks_done` column. Lets you watch refinement progress within a long full-corpus epoch (otherwise blind for hours) and kill early if the metric craters.

The output is a checkpoint with the same on-disk format as the BC one — your inference service can swap them transparently.

**Outputs in the run dir:**
- `step.csv` — per-batch training metrics (one column per head: `loss_play`, `loss_wish`, `loss_dragon_assignment`). Written per-batch as the trainer fires them — durable for live monitoring and survives mid-epoch crashes.
- `epoch.csv` — per-epoch AWR summary (`epoch`, `chunks_done`, `loss_total`, `avg_weight`, `win_rate_proxy`). In streaming mode with `eval_every_chunks > 0` the file also gets intermediate rows with `chunks_done` populated, one per evaluation snapshot. Flushed per row.
- `checkpoints/step_NNNNNN.bin` — versioned weights. Streaming AWR additionally writes one `awr_epoch_NNN_step_NNNNNN.bin` per completed epoch so multi-hour runs are resumable on crash.
- A copy of the YAML config you passed in (so the run is fully reproducible from the run dir).

To visualise a run, point `tools/plot_step_csv.py` at the run directory — it auto-detects BC vs AWR by the columns present and adds the right diagnostic panels (per-head loss + accuracy, plus AWR weight diagnostics and per-epoch summary when applicable).

---

### `train_calls`

**What it does:** Trains the two binary "should I call Tichu / Grand Tichu?" networks. These are separate from the main policy because the decision is made in a very different game phase (Grand Tichu is called after seeing only 8 cards; regular Tichu is called after the post-schupfen 14-card hand). Same skill-conditioned architecture, just smaller.

**When you'd run it:** Once you have parquet shards. You can run this in parallel with `train_bc`.

```bash
train_calls --config configs/calls_smoke.yaml --run-dir runs/calls/smoke
train_calls --config configs/calls_smoke.yaml --run-dir runs/calls/smoke --only tichu
```

The `--only {grand,tichu}` flag trains just one of the two networks. By default both train.

**Outputs:** Two checkpoint files (`grand_final.bin`, `tichu_final.bin`) plus per-tag step CSVs and a per-epoch "calling rate by skill decile" CSV that lets you sanity-check that the model has learned to call more often when strong players would have called.

---

### `train_belief`

**What it does:** Trains a **belief model** — a network that, given the public game history, predicts which cards are in each opponent's hand. This is not used by the inference service in phase 1; it exists as a phase-2 enabler for search-based agents (e.g. ISMCTS) that need to sample plausible opponent hands.

**When you'd run it:** Optional. Run it once you have parquet shards if you want a belief-model checkpoint for future experimentation.

```bash
train_belief --config configs/belief_smoke.yaml --run-dir runs/belief/smoke
```

**Outputs:** `step.csv`, `epoch.csv`, `calibration.csv` (a 10-bucket reliability table — predicted probability vs. empirical frequency, your calibration sanity check), and `checkpoints/belief_final.bin`. The belief checkpoint is explicitly **not** loaded by the inference service.

---

### `eval_matrix`

**What it does:** Runs the project's headline quality benchmark. Two modes:

**Tournament mode (default):** plays an all-vs-all tournament among any set of named agents over a fixed pool of 10,000 deals. For every pair of agents it plays each deal twice with the seats swapped (this cancels the structural advantage of being the Mahjong holder), then reports the average score delta with a 95% bootstrap confidence interval. The output is a tidy Parquet that's also pretty-printed to stdout.

```bash
eval_matrix --config configs/eval_matrix.yaml
```

A minimal config looks like:

```yaml
deal_pool: artifacts/deal_pool_10k.parquet   # produced by tichu_eval.deal_pool
n_deals: 100                                  # subset of the pool head
agents:
  - {name: random, factory: random, kwargs: {seed: 0}}
  - {name: rule,   factory: rule,   kwargs: {}}
bootstrap_iters: 1000
seed: 0
output: artifacts/eval_matrix.parquet
```

**Held-out move-prediction mode:** loads a directory of held-out `.tch` files (games never seen during training), walks each one through the engine, and asks each agent what it would have done at every decision point. Reports top-1 accuracy always; top-5 only for agents that implement `rank_actions` (learned agents do; the rule-based and random ones don't).

```bash
eval_matrix --config configs/eval_move_pred.yaml \
            --mode move_prediction \
            --held-out data/held_out_tch/
```

**Outputs:** A Parquet (tournament) or CSV (move-prediction) file with one row per (agent, decision_type) — plus the same data pretty-printed to stdout so you can read it without opening the file.

The smoke test for this CLI verifies that `RuleAgent` beats `RandomAgent` over 100 deals with a positive margin — if that ever fails, you know something has broken in the orchestrator, the legality checking, or the score accounting.

---

### `export_model`

**What it does:** Converts a training checkpoint into a **TorchScript** artifact — a self-contained file that includes the network weights, the computation graph, and the versions of the featurizer and action space (stamped as extra files inside the archive). TorchScript artifacts can be loaded with just a torch install — no training code, no data pipeline, no Python source tree.

**When you'd run it:** Once per checkpoint you intend to ship to the inference service. Run separately for the BC policy and each of the two call networks.

```bash
export_model \
  --checkpoint        runs/bc/checkpoints/final.bin \
  --tichu-checkpoint  runs/calls/checkpoints/tichu_final.bin \
  --grand-checkpoint  runs/calls/checkpoints/grand_final.bin \
  --model-config      configs/model_arch.yaml \
  --format            torchscript \
  --output            artifacts/exported/ \
  --benchmark                                  # optional: run latency benchmark
  --p99-budget-ms     500                      # optional: fail if p99 over budget
```

The `--model-config` is a small YAML listing the architecture hyperparameters used at training time (hidden width, depth, etc.), because those aren't stored in the checkpoint itself. You provide it to make sure the exporter rebuilds the network with the exact shape that matches the weights.

If you pass `--benchmark`, the tool runs 1,000 sequential `act()` calls on a dummy input and prints p50 / p95 / p99 / mean latency in milliseconds. It exits with a non-zero return code if p99 is above the budget.

ONNX export is documented as a follow-up; the CLI explicitly rejects `--format onnx` with a clear message for now.

**Outputs:** `policy.pt`, `tichu_call.pt`, `grand_tichu_call.pt` in the output directory.

---

### `serve_inference`

**What it does:** Starts the HTTP inference service. The service loads four agents at startup — one per difficulty level — and serves them through three endpoints. It is **stateless**: no per-game memory between requests, so you scale it horizontally by adding replicas behind a load balancer.

**When you'd run it:** When you want to actually use the trained AI inside a game server.

```bash
serve_inference --config configs/serve.yaml          # binds the port
serve_inference --config configs/serve.yaml --no-serve  # builds the app but exits — for smoke testing
serve_inference --config configs/serve.yaml --port 9000  # override port from CLI
```

A serve config looks like this:

```yaml
agents:
  easy:   {factory: rule}
  medium: {factory: ml, checkpoint: artifacts/exported/policy_small.pt}
  hard:   {factory: ml, checkpoint: artifacts/exported/policy.pt}
  master: {factory: ml, checkpoint: artifacts/exported/policy_awr.pt}
port: 8000
```

The four difficulty slots are required. If any of the listed checkpoints fails its version check at startup, the service refuses to start — you get a clear error rather than silently broken predictions.

The endpoint contract is described in detail in the next section.

---

## End-to-end training pipeline

Here is the order to run things if you're starting from a directory of `.tch` files (or an existing `archive.zst`) and want to end up with a serving HTTP endpoint. Substitute paths to match your layout.

```bash
# 0. (One-time, if you don't already have the archive.) Compress the raw .tch
#    directory into the indexed-blob archive used by the rest of the pipeline.
python tools/compress.py data/bsw/ data/archive.zst

# 1. Compute TrueSkill ratings. Streams the archive in chronological order
#    and writes one row per handle. The trainer joins this live by
#    player_handle — no need to re-emit parquet afterwards.
compute_trueskill \
  --input    data/archive.zst \
  --output   artifacts/ratings.parquet \
  --min-games 20

# 2. Parse the archive into the per-decision parquet manifest. The trainer
#    will replay rounds on the fly at training time — these shards are a
#    validated (game_id, round_id) manifest plus label / metadata columns.
parse_bsw \
  --archive data/archive.zst \
  --output  artifacts/parquet/ \
  --workers 12

# 3. Train the main BC policy network. Joins ratings live by player_handle.
train_bc \
  --config  configs/bc_full.yaml \
  --run-dir runs/bc/v1/ \
  --workers 11 \
  --device  auto

# 4. Train the two call networks (in parallel with step 3 if you like).
train_calls \
  --config  configs/calls_full.yaml \
  --run-dir runs/calls/v1/

# 5. (Optional) Refine the BC policy with offline RL (AWR).
#    awr_smoke_100k.yaml uses the streaming pipeline; copy it and point
#    dataset_kwargs at your full-corpus paths for the production run.
train_bc \
  --config  configs/awr_smoke_100k.yaml \
  --run-dir runs/awr/v1/ \
  --refine-from runs/bc/v1/checkpoints/<latest>.bin \
  --device  cuda --workers 11

# 6. Evaluate every checkpoint against the others.
eval_matrix --config configs/eval_matrix.yaml

# 7. Export the checkpoints you want to ship to TorchScript.
export_model \
  --checkpoint        runs/awr/v1/checkpoints/<final>.bin \
  --tichu-checkpoint  runs/calls/v1/checkpoints/tichu_final.bin \
  --grand-checkpoint  runs/calls/v1/checkpoints/grand_final.bin \
  --model-config      configs/model_arch.yaml \
  --format            torchscript \
  --output            artifacts/exported/ \
  --benchmark

# 8. Start the inference server.
serve_inference --config configs/serve.yaml
```

For early experimentation, replace each `*_full.yaml` with the matching `*_smoke.yaml`. `configs/bc_smoke.yaml` uses a synthetic dataset for engine-independent testing; `configs/bc_smoke_parquet.yaml` runs the real replay-on-the-fly path against the 100k parquet manifest with a smaller model — combine with `--max-examples` for sub-minute end-to-end smokes against real data.

---

## The inference HTTP service

Once `serve_inference` is running, you interact with it through three endpoints. All payloads are JSON.

### `POST /act`

The single endpoint a game server uses to ask "what should this player do?".

**Request body:**

```json
{
  "difficulty": "hard",
  "private_state": {
    "player": 0,
    "hand": [3, 7, 14, 21, 28, 35, 42, 49, 50, 51, 52, 53, 54, 55],
    "public": {
      "current_player": 0,
      "hand_sizes": [14, 14, 14, 14],
      "scores": [0, 0],
      "trick": {"plays": [], "leader": null, "passes": []},
      "mahjong_wish": null,
      "pending_decision": null,
      "round_points_by_player": [0, 0, 0, 0],
      "out_order": [],
      "tichu_callers": [],
      "grand_tichu_callers": []
    }
  }
}
```

- `difficulty` is one of `"easy"`, `"medium"`, `"hard"`, `"master"`. Each maps to a different agent loaded at startup — `easy` is rule-based (no ML), the other three are learned policies of increasing quality.
- `private_state` is a JSON-serialized `PrivateState`: the acting player's hand plus everything they can see about the public game state. **Cards are encoded as integer IDs from 0 to 55**, matching the canonical deck order in `tichu_engine.deck.fresh_deck()`. The codec round-trips: whatever the codec serializes, it can also deserialize back to an exact `PrivateState`.

**Response body:**

```json
{
  "action": {"kind": "Pair", "cards": [3, 7]},
  "action_index": null,
  "fallback_used": false
}
```

- `action` is the engine action the AI chose, serialized the same way the codec serializes combinations. The `kind` tag tells you which variant (e.g. `"Single"`, `"Pair"`, `"Triple"`, `"FullHouse"`, `"Straight"`, `"PairStep"`, `"FourOfAKindBomb"`, `"StraightFlushBomb"`, `"Pass"`, `"DragonGive"`, `"MahjongWish"`, `"SchupfenPass"`).
- `fallback_used` is `true` if the inference path failed (NaN logits, model exception, no legal action with positive probability) and the service fell back to picking a random legal action. **The action returned is always legal**, even in the fallback case — the game never stalls.

**Error responses:**

- `400 Bad Request` if the difficulty is unknown or the `private_state` payload is malformed.
- The service returns `503` from `/health` if any agent failed to load, but `/act` itself won't hit that case — startup would have raised before binding the port.

### `GET /health`

Returns 200 with `{"status": "ok", "agents": ["easy", "hard", "master", "medium"]}` when every configured agent is loaded and ready. Use this as a load-balancer health probe.

### `GET /metrics`

Returns Prometheus-format text. Exposed metrics:

- `tichu_requests_total{difficulty="..."}` — counter, total `/act` requests per difficulty.
- `tichu_fallback_total{difficulty="..."}` — counter, total fallback events per difficulty.
- `tichu_latency_p50_ms`, `tichu_latency_p95_ms`, `tichu_latency_p99_ms` — gauges, rolling-window latency over the last 200 requests.

### What "stateless" means here

The service holds no per-game or per-session memory. Every request carries the full game state. This is on purpose: it means you can run multiple replicas behind a load balancer and route requests to whichever is free, without having to pin a game to a specific replica.

---

## Project layout

```
src/
  tichu_engine/             # rules engine: cards, combinations, legality, scoring
  tichu_ml/                 # the Agent interface + Random/Rule baselines + registry
  tichu_training/           # everything that produces a checkpoint
    bsw/                    #   .tch parser, replay validator, parquet writer
    ratings/                #   TrueSkill rating computation
    bc/                     #   trunk + heads + masked loss + training loop
    awr/                    #   value baseline + AWR weight + refinement loop
    belief/                 #   opponent-hand prediction model + training
    cli/                    #   one module per CLI tool
    action_space.py         #   the canonical 1,809-action space
    featurizer.py           #   PrivateState -> 16,568-dim feature vector
    checkpoint.py           #   versioned on-disk checkpoint wrapper
  tichu_eval/               # tournament harness + held-out move prediction
  tichu_export/             # TorchScript export primitive + latency benchmark
  tichu_inference/          # FastAPI service + ML agent adapter + codec
    cli/serve.py            #   the serve_inference CLI

tests/                      # mirrors the src/ layout — one test dir per package
configs/                    # example YAML configs (each command has a *_smoke.yaml)
documentation/              # the PRD and rulebook
docs/adr/                   # Architecture Decision Records — read these to understand non-obvious design choices
sample/                     # two sample .tch games used by the tests
tools/                      # standalone scripts (uv-script format) — data ingest +
                            # post-training visualisation (plot_step_csv.py)
```

A couple of design notes worth knowing:

- **`tichu_engine` is import-free of any ML side.** The engine knows nothing about PyTorch, the action space, or the featurizer. This means it stays small and easy to test, and it remains usable if you ever want to plug in a different ML stack.
- **The training side never imports the inference side, and vice versa.** Once an artifact is exported, the inference service can load it without any training dependencies. This is enforced by structure: `tichu_export.torchscript.load_exported` lives in a package that depends only on torch.

---

## Running the tests

The test suite is the most thorough piece of documentation. There are 624 tests and they run in about 2 minutes.

```bash
# Run everything except the slow-marked tests:
python -m pytest -q

# Run everything including slow tests (currently only the production-size
# latency gate):
python -m pytest -q -m ""

# Run just the inference-side tests:
python -m pytest tests/inference -q

# Run a single test by name:
python -m pytest tests/eval/test_tournament.py::test_rule_beats_random_on_100_deals
```

Every CLI has a smoke test that does the full thing on a tiny synthetic dataset. Reading those tests is a good way to see what a real invocation looks like.

---

## Known limitations and deferred work

The following items are documented but not yet implemented. They live as comments in the relevant modules.

- **Schupfen Network implementation.** [ADR-0012](docs/adr/0012-schupfen-is-a-standalone-network.md) carves Schupfen out into a standalone network because a Schupfen Decision is structurally "pick 3 of 14 cards, assign 3 labeled destinations" — not a single-discrete-output decision that fits the shared-trunk multi-head pattern. The `tichu_training/bc/schupfen_model.py` module + `train_schupfen` CLI are not yet implemented; for now the ML Agent falls back to a heuristic at Schupfen-time. Output shape (multi-output, autoregressive decoder, or pointer network) is itself an open design choice to be resolved when the implementation lands.
- **Wish / dragon_assignment inference decoding.** The BC heads exist and train end-to-end as of [ADR-0011](docs/adr/0011-bc-training-replay-on-the-fly.md), but the `MLAgent` doesn't yet wire the `wish` and `dragon_assignment` head outputs through to concrete engine actions. At inference the ML Agent currently delegates pending-decision moments to `RuleAgent` heuristics. The plumbing is small — argmax + the small index→engine-value mapping — and is the most immediate follow-up.
- **Negative call examples.** The BSW parser only emits positive Tichu / Grand Tichu calls. Negative examples (a player at a call-decision point who chose not to call) need to be synthesized — deferred to a follow-up.
- **Trunk sharing between BC and value baseline.** The AWR value baseline is its own small MLP rather than sharing the BC trunk. Cleaner from a wiring standpoint at this stage; trunk sharing is a possible optimization later.
- **ONNX export.** TorchScript is the primary target. The `--format onnx` flag is reserved but currently rejected with a clear message.
- **BombInterrupt in tournament play.** The tournament harness exercises only in-turn actions because `legal_actions_for` doesn't enumerate out-of-turn bomb interrupts. Engine support exists; surfacing it through the harness is a follow-up.
- **Tichu calling composed onto play agents.** The play and call models exist as separate networks; a thin wrapper that uses the call models to decide whether to announce Tichu before play begins is a follow-up.
- **Within-epoch shuffle buffer.** `ParquetBCDataset` iterates the archive in offset order (= chronological by `tools/compress.py` construction). SGD tolerates this, but a bounded shuffle buffer (e.g. 16k examples) would improve within-epoch independence. Documented in ADR-0011's "Consequences" section.

The combination of "what's complete + what's deferred" is also captured in the commit messages, in `documentation/PRD-python-ai-implementation.md`, and across the ADRs under `docs/adr/`.
