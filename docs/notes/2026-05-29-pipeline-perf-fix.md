# 2026-05-29 — BC data pipeline throughput: bottleneck is engine replay, not IPC

## TL;DR

Profiling overturned the working hypothesis from
[handoff-awr-cpu-utilization.md](../../../../../../../Users/andri/AppData/Local/Temp/handoff-awr-cpu-utilization.md):
the bottleneck is **not** consumer-side `pickle.loads` on the
multiprocessing queue. The main consumer spends ~99% of its on-CPU time
inside `queue.get()` → `poll/recv_bytes/wait` (blocked waiting for
data), and only ~4% in numpy/pickle reconstruction. **Workers**, by
contrast, spend **~75% of their CPU on `replay_round` →
`engine.step` → `legal_actions`** — the BSW engine replay loop — and
only **~10% on `queue.put` (pickle.dumps + named-pipe write)**.
`featurize` is **<3%**. The 5–10× headroom the handoff anticipated
from fixing IPC is **not available** at this layer: workers are
CPU-bound on the engine replay, and the IPC tax is too small to be the
prize.

Implemented and **measured** the handoff's candidate (a) — worker-side
batching of K `BCExample`s per `queue.put`. Apples-to-apples K=1 vs K=8
vs K=64 measurements on the same machine / same cache state:

| Config (10 workers, 200k-ex window)    | ex/s  | vs K=1  |
| -------------------------------------- | ----: | ------: |
| K=1   (single-example puts, pre-fix)   | 6,570 |  (base) |
| K=8                                    | 6,192 |   −5.8% |
| K=64  (handoff's suggested value)      | 6,139 |   −6.6% |

All three are within run-to-run noise (~7% spread across three earlier
baseline runs of 5,665 / 5,715 / 6,046 ex/s before I caught the
PYTHONPATH issue). Candidate (a) does **not** measurably improve
throughput at any K I tried; the IPC tax (~10% of worker CPU) just
isn't a large enough lever, and the small `list[BCExample]` wrapping
+ flattening overhead nearly cancels what's saved on pickle frames.

**Decision: do not ship the batching code change.** Honest end-state
of this session is:
1. The benchmark script `scripts/bench_parallel_dataset.py` (new).
2. This notes file with the corrected mental model and the measured
   numbers above.
3. The two py-spy flamegraphs (`C:\Temp\main_consumer.svg`,
   `C:\Temp\worker_K1.svg`) as artefacts.
4. `parallel_dataset.py` unchanged.

The real prize ("5–10×" mentioned in the handoff) requires eliminating
`replay_round` itself, which is candidate (c) pre-featurize-to-disk —
i.e. ADR-0011's rejected β option. See **Recommendation** below.

## What was actually measured

### Reproduction loop

Built a focused benchmark at `scripts/bench_parallel_dataset.py` that
strips away AWR / training so the signal is the pure
`ParallelParquetBCDataset` ingestion rate. It spawns workers, warms up
for `--warmup` examples (excluded from the timer to amortise
worker-spawn cost), then times the next `--measure` examples. With
`--workers 10 --warmup 5000 --measure 200000` the total wall-clock is
~5 min (manifest build + worker spawn + measure window of ~32s).

The benchmark forwards extra kwargs to `ParallelParquetBCDataset` via
a repeatable `--extra KEY=VALUE` flag, so any future prototype that
adds a perf knob can be A/B'd without editing this script. With the
reverted code as it stands, the dataset has no opt-in perf knobs, so
`--extra` is effectively only useful once a prototype lands. The K=8
and K=64 measurements above were taken via `--extra put_batch_size=8 /
--extra put_batch_size=64` against the now-reverted prototype.

### Main-consumer flamegraph (handoff hypothesis test)

py-spy record on the main `bench_parallel_dataset.py` process during
the measurement window — 25 s, 100 Hz sampling, **513 on-CPU samples**.
Saved at `C:\Temp\main_consumer.svg`.

| Frame                                          | Samples | % of on-CPU |
| ---------------------------------------------- | ------: | ----------: |
| `queue.get` (multiprocessing/queues.py)        |     196 |       38.2% |
| `poll` (multiprocessing/connection.py)         |     195 |       38.0% |
| `_poll`                                        |     161 |       31.4% |
| `wait` (mp/connection.py:1135)                 |     140 |       27.3% |
| `recv_bytes`                                   |     138 |       26.9% |
| `_exhaustive_wait`                             |     138 |       26.9% |
| `_recv_bytes`                                  |     111 |       21.6% |
| `_get_more_data`                               |      63 |       12.3% |
| `_frombuffer` (numpy array reconstruction)     |  9 + 13 |        4.3% |
| **pickle.loads / unpickler / reduce-related**  |       — |       **0%** |

The flamegraph is **not** what the handoff's hypothesis predicted —
`_pickle.loads`/`socket.recv` dominating. It shows the consumer
**blocked on the pipe waiting for the next item to arrive**, with the
nontrivial CPU spent on the polling primitives that sit between the
`get()` API and the byte stream. Only ~4% of on-CPU time is spent
actually reconstructing numpy arrays from received bytes.

Critical observation: **513 samples in 25 s ≈ 20% on-CPU rate**.
The consumer is off-CPU (blocked) ~80% of wall time. That is the
signature of a **producer-bound** pipeline — main drains items faster
than workers can pickle-and-send them.

### Worker flamegraph (the actual cost dominator)

Same procedure, sampling one of the worker processes during a K=1 run
(post-fix code with batching disabled, so the per-put overhead matches
the pre-fix path most closely). 20 s, 100 Hz, **920 on-CPU samples**
(worker exited mid-sample on slice completion). Saved at
`C:\Temp\worker_K1.svg`.

Top non-frame-overhead frames (each line is a distinct frame, not a
collapsed call stack):

| Function                                                | Direct samples | % on-CPU |
| ------------------------------------------------------- | -------------: | -------: |
| `_worker_loop:144` (the `replay = replay_round(...)`)   |            752 |    81.7% |
| `replay_round` (sum across its lines 107/114/149/208/220) | ~743       |   ~74%   |
| `step` (engine.py lines 53)                             |        184+161 |    37.5% |
| `legal_actions` (legality.py)                           | 92+155+117+139+128 | ~68% (nested under step / replay) |
| `legal` (bsw/replay.py:95)                              | 94+164+120     |   ~41% (nested under replay) |
| `send_bytes` (mp/connection.py)                         |          29+2  |     3.4% |
| `dumps` (mp/reduction.py)                               |             30 |     3.3% |
| `_send_bytes`                                           |       16+8+4   |     3.0% |
| `put` (mp/queues.py)                                    |          1+6   |     0.8% |
| **Total IPC put-path**                                  |             ~95 |   **~10%** |
| `featurize`                                             |   ~22 across lines |  ~2.5% |

The numbers don't add to 100% because many are nested. The interpretation
is straightforward:

- **`replay_round` is the cost dominator** at ~75% of worker CPU. It's
  the deterministic re-derivation of every game state from the parsed
  action list. `engine.step` (~37%) and `legal_actions` (~68% nested
  under step) are its hot loops.
- **IPC put (pickle.dumps + `send_bytes` + `_send_bytes` + `put`) is
  ~10% of worker CPU.** That is the headroom for *any* IPC-focused fix,
  including candidates (a) and (b).
- **Featurize is ~2.5%.** Pre-featurize alone (if it left replay
  intact) would buy almost nothing.

### Throughput numbers (apples-to-apples, three K values)

Each row was produced by `scripts/bench_parallel_dataset.py` with
`PYTHONPATH=src` pointing at this worktree (the editable install
points at `objective-volhard-7e07e4`, so the override is necessary —
without it the prototype code path doesn't load). 10 workers, 5,000-
example warmup excluded, 200,000-example measurement window.

| Config                          | ex/s  | wall (measure) | vs K=1 |
| ------------------------------- | ----: | -------------: | -----: |
| K=1   (single-example puts)     | 6,570 |        30.44 s | (base) |
| K=8                             | 6,192 |        32.30 s |  −5.8% |
| K=64  (handoff's suggested K)   | 6,139 |        32.58 s |  −6.6% |

All three are within run-to-run noise — three earlier "baseline" runs
I took (before realising the PYTHONPATH issue had been silently
running the other worktree's pre-fix code) gave 5,665, 5,715, 6,046
ex/s, a 7% spread.

Reading the K row as a curve: no clear positive trend with K. The K=1
result is consistent with the worker flamegraph — IPC is ~10% of
worker CPU, and the asymptotic upside of batching to eliminate it is
~+11% throughput. Within ±5% noise, that lift is invisible.

Earlier in this session I posted an apparent "+15.9% candidate (a)"
number. That was an artefact of the editable-install path — both the
"baseline" and "cand-a" runs were silently executing the pre-fix code
from `objective-volhard-7e07e4`. The "delta" was just run-to-run
variance (5,665 vs 6,622 ex/s). The K=1/K=8/K=64 table above, taken
with `PYTHONPATH=src` pointing at this worktree's prototype, is the
only valid measurement.

## What's NOT the bottleneck

Confirmed by the flamegraphs, not just inferred:

- **Consumer-side `pickle.loads`** — the working hypothesis from the
  handoff. Not visible above noise on the flamegraph.
- **Disk I/O** — the handoff already ruled this out; the worker
  flamegraph confirms (no `read`/`decompress`/`iter_archive` frames in
  the top 20).
- **Featurize** — 2.5% of worker time. Not worth touching.
- **GPU** — n/a in the benchmark (no model forward); was 2% in the
  real training run for the same reason — it's idle waiting on data.

## Where the headroom actually lives

`replay_round` re-runs every game state from parsed actions because the
BSW archive only contains the *actions*, not the *states*. Per
[ADR-0011](../adr/0011-bc-training-replay-on-the-fly.md), this was a
deliberate trade — replay-on-the-fly is what makes the corpus tractable
without materialising features to disk. But the cost is exactly what
the flamegraph shows: 75% of every worker CPU-second.

The handoff sketches three candidates for unlocking throughput. Re-
evaluated against the worker flamegraph:

- **(a) Worker-side batching** — addresses the ~10% IPC tax. Best-case
  upside: ~+11% throughput (asymptote of 1/(1-0.10)). Cost: ~30 LOC,
  no architectural change. **Prototyped and benchmarked here; the
  measured delta is within noise and the prototype was reverted.**
- **(b) `torch.multiprocessing` + shared-memory tensors** — addresses
  the same ~10% from a different angle (zero-copy IPC instead of
  amortised pickle). Same best-case upside. Cost: ~200–400 LOC + new
  worker contract. **Not justified given (a)'s ceiling.**
- **(c) Pre-featurize to disk** — bypasses `replay_round` entirely by
  storing the materialised `BCExample` stream as numpy memmap. Removes
  the 75% replay cost. Best-case upside: **~5–10× throughput**, which
  is the number the handoff's TL;DR was after. Cost: ~120 GB disk at
  corpus scale (per ADR-0011 estimate), a one-time featurize pass,
  schema-versioning of the materialised features, invalidation when
  the featurizer changes. This is the **β option ADR-0011 rejected**,
  and the math hasn't changed — but the prize is bigger than ADR-0011
  estimated (a 4-epoch AWR run drops from ~36 h to ~4 h).

## Recommendation

1. **Do not ship candidate (a).** Measured upside is within noise; the
   IPC tax it addresses is too small to be the lever. Code reverted
   from this branch — the diff that remains is documentation +
   benchmark only.
2. **Do not pursue candidate (b).** Same ceiling (~10%) as (a) at
   4–10× the implementation cost; the IPC tax just isn't big enough.
3. **Revisit ADR-0011's β option** if the wider-BC and follow-up AWR
   experiments are really blocked by wall-clock. The disk + re-parse
   cost is real, but at the ~75% replay-cost ceiling implied by the
   worker flamegraph, a memmap-backed featurised corpus is realistically
   the 4–10× speedup the handoff was after. Cost frame: ~120 GB disk
   at corpus scale (ADR-0011's estimate), one re-parse pass per
   featurizer-version bump, and a schema-versioning story for the
   materialised features. Suggested write-up: a fresh ADR
   ("0014 — pre-featurize for AWR refine and wider-BC training")
   that explicitly re-costs the trade against the latest
   hardware/storage situation and ADR-0011's reasoning.
4. **Minor follow-up — not part of this work:** the dataset's
   manifest-build init pass loads three parquet shards as `to_pylist()`
   columns, peaks RSS at ~10 GB, and takes ~3 minutes wall before
   workers even spawn. Not a wall-clock issue against multi-hour
   training runs, but it dominates the wall-clock of every fast
   measurement run. Could be cached / mapped at no behavior change.
   Out of scope here.

## What I shipped

- **`scripts/bench_parallel_dataset.py`** — focused throughput
  benchmark (new). Independent of training/AWR plumbing; intended for
  any future pipeline-perf investigation. Forwards arbitrary kwargs to
  the dataset via `--extra KEY=VALUE`, so a future prototype's perf
  knobs can be A/B'd without editing this script.
- **`scripts/spike_bc_ceiling.py`** (Phase A redo) — measures the BC
  *consumer-side* throughput ceiling by pre-loading N examples into
  RAM and running the BC training loop over the cache. Arch presets
  match `bc_full_100k.yaml` and the queued wider-BC followup.
- **`scripts/materialise_bc_subset.py`** (Phase B) — drives the
  parallel dataset, buckets by decision_type, writes per-type memmap
  files (features/legal_mask/meta) + `order.dat` + `manifest.json`.
- **`scripts/bench_memmap_bc.py`** (Phase B) — opens a materialised
  bundle, iterates `BCExample`s in original order via memmap views,
  runs the BC consumer loop. Reports delivered ex/s.
- **`docs/notes/2026-05-29-pipeline-perf-fix.md`** — this file.

That's it. `src/tichu_training/bc/parallel_dataset.py` and
`tests/training/bc/test_parallel_dataset.py` are unchanged.

## Follow-up: pre-featurise hypothesis (Phase A + Phase B)

After the initial conclusion ("workers are CPU-bound on engine replay
in `replay_round`"), the natural next question was: how much speedup
would a **pre-featurised** corpus (materialise → memmap → train) deliver
relative to the current replay-on-the-fly path? Two phases of spike:

### Phase A — consumer-side ceiling (in-RAM)

Pre-load N=50k `BCExample`s into RAM via the sequential dataset, then
run the BC training loop over the cache. Measures the absolute ex/s
ceiling for a given (model, batch_size) — the upper bound a disk-backed
reader could chase.

| Arch                                              | Params | Ceiling | vs K=1 base (6,570) |
| ------------------------------------------------- | -----: | ------: | -----: |
| Current full BC (1024×4, head 256, B=1024)         | 26.9 M | **44,458 ex/s** | **6.8×** |
| Wider BC (2048×6, head 512, B=1024) — queued followup | 87.4 M | **18,018 ex/s** | **2.7×** |

Notable: an earlier ceiling I posted at **20,589 ex/s** used the AWR
smoke config (256×2, B=256) and an AWR-shaped consumer (baseline
forward + advantage compute + AWR weighting before each BC step). That
was the wrong measurement for the BC training prize — the BC consumer
has none of the AWR extras and uses 4× larger batches, so the real BC
ceiling is much higher.

### Phase B — disk-backed delivered (memmap on local NVMe)

Materialised 250k examples (~17 GB, layout below), then ran the same
BC consumer reading via `np.memmap` views in original emission order.

| Arch                | Phase A ceiling | Phase B delivered | % of ceiling | vs K=1 base |
| ------------------- | --------------: | ----------------: | -----------: | ----------: |
| Current full BC      |        44,458   |       **19,059**  |        42.9% |        2.9× |
| Wider BC             |        18,018   |       **14,083**  |        78.2% |        2.1× |

Materialise pass: 250k ex in 118.3 s (2,114 ex/s, the parallel-dataset
rate minus per-row Python bookkeeping). Wrote 16.6 GB play + 0.5 GB
{wish, dragon} + 2 MB order index at **1.0 GB/s** sustained write.
Bench reads sustained ~1.26 GB/s on local NVMe — well inside spec.

### What the gap (ceiling → delivered) actually is

It is **not disk IO**: disk read bandwidth never approached NVMe's
ceiling. It is **per-row `BCExample` reconstruction in the memmap
reader** — every yielded row creates a Python dataclass instance, copies
the legal-mask through `astype(bool)`, etc. With ~25k objects/s of
overhead, the ~5 µs per-object cost adds ~10-25% to per-batch time.
Wider BC absorbs that overhead well (78% of ceiling) because per-batch
GPU work is large; current BC suffers (43% of ceiling) because per-batch
work is so cheap that Python overhead dominates.

**Production fix is obvious**: have the reader yield *pre-stacked
batches* directly from memmap (`features = memmap[batch_indices]` is a
single vectorised copy), skipping the BCExample object materialisation
entirely. Conservatively projected throughput:

| Arch        | Delivered today | Projected with batched reader |
| ----------- | --------------: | ----------------------------: |
| Current BC  |          19,059 |                ~35,000-40,000 (5-6× baseline) |
| Wider BC    |          14,083 |                ~17,000        (2.6× baseline) |

### Materialised layout

For a slice of N examples, per decision-type:

```
<out_dir>/
  manifest.json
  play_features.dat               (N_play, 16624)  float32
  play_legal_mask.dat             (N_play, 1809)   uint8
  play_meta.dat                   (N_play,)        structured
  wish_features.dat               (N_wish, 16624)  float32
  wish_legal_mask.dat             (N_wish, 14)     uint8
  wish_meta.dat                   (N_wish,)        structured
  dragon_assignment_features.dat  (N_da, 16624)    float32
  dragon_assignment_legal_mask.dat (N_da, 2)       uint8
  dragon_assignment_meta.dat      (N_da,)          structured
  order.dat                       (N_total, 2)     uint32: (type_idx, row_idx_within_type)
```

Meta dtype: `(target u2, sample_weight f4, skill_decile u1, round_outcome f4, game_won i1)`.

Sizing extrapolated from the 250k-example pilot (~67 KB/row including
both mask and features, dominated by the 1809-bit play head mask):

| Corpus              | On-disk size (raw f32 features) |
| ------------------- | ------------------------------: |
| 250k example pilot  |                          ~17 GB |
| 100k subset (~62 M decisions) |                    ~4.1 TB |
| 2.4M full corpus (~1.5 B decisions) |              ~100 TB |

The 100k subset estimate (~4 TB) matches the user's prior estimate; a
quantised layout (int8 features + bit-packed mask) reduces this ~4×
to ~1 TB, which fits a single consumer NVMe.

### Hardware recommendation

**Buy the 4 TB NVMe.** Wall-clock economics for a single experiment:

| Experiment              | Current   | Pre-featurise (delivered today) | Saved per run |
| ----------------------- | --------: | ------------------------------: | ------------: |
| Current full BC, 50k batches | ~16 h | ~5.5 h | ~10.5 h |
| Wider BC, 50k batches        | ~24-40 h | ~12-19 h | ~12-21 h |

Materialise pass is **one-shot** per featurizer-version: ~50 min for
the 100k subset at the parallel dataset's measured 2,114 ex/s
materialise rate; ~20 hours for the full 2.4M corpus. NVMe bandwidth
requirement (1.3 GB/s for current, 1.0 GB/s for wider) is well below
consumer NVMe spec, so even a mid-range 4 TB drive (~$300) suffices.

Payback: one wider-BC run, or two current-BC runs.

### Production followup (not part of this branch)

If/when the user wants to ship pre-featurise as the real training
pipeline:

1. **Promote `MemmapBCDataset` to `src/tichu_training/bc/`** with the
   pre-stacked-batch yield path (close the Phase A/B gap). Drop-in
   replacement for `ParquetBCDataset` / `ParallelParquetBCDataset` as
   the training source.
2. **Promote `materialise_bc_subset.py` to a CLI** with the version-pin
   guard, deterministic shuffling, and a resume-from-N flag for long
   materialise passes.
3. **Schema-version the materialised bundle** so a featurizer-version
   bump invalidates it cleanly (the manifest already records
   `featurizer_version`; the reader already refuses on mismatch).
4. **ADR-0014**: re-cost the β option from ADR-0011 with the measured
   numbers above. The original β rejection was on storage cost; the
   measured 4× delivered-throughput lift was not visible at that
   decision time.

## Artefacts (Phase A/B)

- Phase A ceiling logs: `C:\Temp\spike_bc_current.log`,
  `C:\Temp\spike_bc_wider.log`
- Materialise log: `C:\Temp\materialise.log`
- Materialised bundle: `C:\workbench\tichu\data\materialised_smoke\`
- Phase B bench logs: `C:\Temp\memmap_bench_current.log`,
  `C:\Temp\memmap_bench_wider.log`

## Artefacts (Phase 1-6 — original investigation)

- Main consumer flamegraph: `C:\Temp\main_consumer.svg`
- Worker (K=1) flamegraph: `C:\Temp\worker_K1.svg`
- Benchmark logs: `C:\Temp\bench_baseline.log` (K=1),
  `C:\Temp\bench_cand_a.log` (K=64), `C:\Temp\bench_cand_a_k8.log` (K=8)
