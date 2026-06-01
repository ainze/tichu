# ADR-0026: The Full-strength Tournament parallelises across processes, chunked over Positions

- **Status:** Accepted
- **Date:** 2026-06-01
- **Related:** [ADR-0025](0025-full-strength-tournament-is-the-only-variant-and-includes-calls.md)

## Context

The Full-strength Tournament ([ADR-0025](0025-full-strength-tournament-is-the-only-variant-and-includes-calls.md))
ran as a single-threaded sequential loop: for each agent pair, for each Starting
Position, `play_full_round` twice (seat-swap). The example config is ~40k full
Rounds, and Full-strength is ~4× heavier than the retired Play-strength run
(4× more Positions, Schupfen now plays, and up to 8 extra `should_call` torch
forwards per Round). It was slow enough to be the bottleneck on iterating.

Every pair and every Position is independent — the work is embarrassingly
parallel.

## Decision

**Parallelise with a `multiprocessing` process Pool, chunked over the Pool of
Positions, behind a new `workers` config key (default `1` = today's exact serial
behavior).**

1. **Processes, not threads.** A Round is mostly pure-Python CPU work (engine
   `step`, legality enumeration over 1809 intents, ranking, `featurize`),
   serialized by the GIL on the standard CPython build the project runs
   (`Py_GIL_DISABLED == 0`). PyTorch releases the GIL inside tensor kernels, but
   the forwards are batch-1 tiny MLPs dominated by Python overhead, so the
   GIL-free window is small. Threads would net ≈1×. (On a free-threaded build,
   PEP 703, threads would become viable — revisit then.)

2. **Builder-callable agent interface.** `run_full_tournament` takes
   `dict[str, Callable[[], Agent]]` — zero-arg builders — not live `Agent`
   instances. Each worker **rebuilds** every agent from its builder (torch models
   do not pickle/fork cleanly; Windows uses spawn regardless). The serial path
   just invokes each builder once. The eval CLI builds the builders as
   `partial(_build_agent, factory, **kwargs)` from each agent's checkpoint paths.

3. **Chunked over Positions, not one task per Round.** The Pool of Positions is
   split into `min(workers, n)` contiguous index ranges; one task plays a whole
   chunk. Dispatching a task per Round would drown in pickling/IPC. Positions are
   sent to each worker **once** via the Pool initializer (identical across all
   pairs) and referenced by index range thereafter.

4. **`torch.set_num_threads(1)` per worker** so `W` processes don't each spin up
   `W` intra-op threads and oversubscribe the cores.

5. **The orchestration is deterministic; bit-identity holds for torch-free
   agents.** Chunk results are stitched back in Position order, and the bootstrap
   is still drawn on the main process in pair order — so for an agent whose
   per-Round behavior is a pure function of state with no torch inference
   (`RuleAgent`), a parallel matrix is **bit-identical** to the serial matrix at
   the same `seed`. Two pre-existing caveats, both far below the bootstrap CI:
   - **ML agents are not bit-reproducible across the process boundary.** Identical
     featurized inputs yield bit-different torch CPU logits in a different process
     (kernel/vectorization differences, *not* threads — it persists single-threaded
     and with zero inference fallbacks), occasionally flipping a near-tie greedy
     argmax, which cascades to a different Round outcome. Measured: ~1 Round in 40
     differs, shifting a pair mean by ~0.1% — orders of magnitude under the ±60-ish
     bootstrap CI. The serial-only run hid this by staying in one process; it was
     never reproducible across separate process invocations either.
   - **`RandomAgent`** threads one RNG through the whole run; rebuilt fresh per
     worker, its stream resets at chunk boundaries, so it is reproducible per
     run-config but **not** stream-equal to serial under `workers > 1`. Accepted —
     it is only the baseline floor.

6. **Fail fast, never deadlock.** A `Pool` whose initializer raises hangs forever
   (multiprocessing endlessly respawns the dead worker). So every agent is built
   once on the main process before the Pool is created: a broken builder (unknown
   factory, missing checkpoint) raises a clear error up front. Those instances are
   discarded; workers rebuild their own. Relatedly, the `ml` factory is registered
   only by importing `tichu_inference.ml_agent`; routing the CLI builders through
   `tichu_training.cli.eval_matrix._build_agent` makes a spawned worker import that
   module (and so register `ml`) when it unpickles the builder.

## Consequences

- `run_full_tournament`'s signature changed from `dict[str, Agent]` to
  `dict[str, Callable[[], Agent]]`. The play-strength `run_tournament` and
  move-prediction paths still take live agents; the CLI builds those once from the
  builders.
- The matrix is only bit-reproducible across `workers` settings for torch-free
  deterministic agents (`rule`). `ml` rows shift by torch float noise (well under
  the CI) and `random` rows shift because its RNG rebuilds per worker — both when
  `workers` changes, by design. Compare `ml`/`random` rows statistically (within
  the bootstrap CI), not bit-for-bit.
- Building agents once on the main process for validation adds a one-off model
  load on the parent for `ml` agents; negligible against a 40k-Round run, and it
  buys a clear error instead of a silent hang.

## Rejected alternatives

- **Threads.** Serialized by the GIL on the standard build (see Decision 1).
- **One multiprocessing task per Round.** Pickle/IPC overhead per tiny task
  dominates; chunking amortizes it.
- **Pass live agents to workers.** Torch models do not pickle/fork cleanly, and
  Windows spawn re-pickles everything — hence builders + worker-side rebuild.
- **Parallelise across pairs instead of Positions.** Coarser, with worse load
  balance (pair costs differ wildly, e.g. random-vs-random vs ml-vs-ml) and far
  less parallelism than the Position count. Chunking Positions load-balances
  cleanly and keeps the bootstrap trivially deterministic.
