# Tichu — 100k pipeline runbook (PowerShell, featurizer v5)

End-to-end commands to take the BSW archive → ratings → parsed shards +
materialised bundles → trained networks → AWR refine → eval → export, at the
**100k** subset on **featurizer v5** (content identical to v4; the stamp bump
is ADR-0022, which retires every v4-stamped checkpoint).

> **Run everything from the worktree** so the new multi-bundle code is used,
> and put `src` on `PYTHONPATH` so it shadows any pip-installed copy from the
> main checkout. Do this once per shell:

```powershell
cd C:\workbench\tichu\.claude\worktrees\festive-zhukovsky-a9feda
$env:PYTHONPATH = "src"
```

All commands below assume that working directory + env var. `--device cuda`
assumes a GPU — change to `cpu` if none. Swap `py -3.14` for whatever launches
your Python 3.14 if different.

## Paths (v5 generation)

| What | Path |
| --- | --- |
| BSW archive | `C:\workbench\tichu\data\archive.zst` |
| TrueSkill ratings | `C:\workbench\tichu\data\ratings_100k.parquet` |
| Parquet shards (manifest) | `C:\workbench\tichu\data\parquet_100k_feat_v5` |
| Bundle root | `C:\workbench\tichu\data\materialised_100k_v5\` |
| → BC bundle | `…\materialised_100k_v5\bc` |
| → Calls bundle | `…\materialised_100k_v5\calls` |
| → Schupfen bundle | `…\materialised_100k_v5\schupfen` |
| Run outputs | `C:\workbench\tichu\data\runs\<name>` |

---

## 0. (Optional) TrueSkill ratings

`ratings_100k.parquet` already exists; regenerate only if the archive/subset
changed. Must use the **same `--subset`** as the parse below so handles line up.

```powershell
py -3.14 -m tichu_training.cli.compute_trueskill `
  --input   C:\workbench\tichu\data\archive.zst `
  --output  C:\workbench\tichu\data\ratings_100k.parquet `
  --subset  100000 `
  --min-games 20 -v
```

---

## 1. Parse + materialise (one replay pass, many bundles)

This is the consolidation: one engine replay emits the parquet manifest **and**
the BC / calls / schupfen bundles into `materialised_100k_v5\{bc,calls,schupfen}`.
~1.5–2 h on a 12-core box at `--workers 10`.

```powershell
py -3.14 -m tichu_training.cli.parse_bsw `
  --archive        C:\workbench\tichu\data\archive.zst `
  --output         C:\workbench\tichu\data\parquet_100k_feat_v5 `
  --bundle-out-dir C:\workbench\tichu\data\materialised_100k_v5 `
  --bundle-tasks   all `
  --subset         100000 `
  --trueskill      C:\workbench\tichu\data\ratings_100k.parquet `
  --workers        10
```

- `--bundle-tasks all` = `bc,calls,schupfen` (belief is **gated out** of `all`).
- Add belief explicitly if you want it (play-scale, several GB more):
  `--bundle-tasks bc,calls,schupfen,belief`.
- Sanity-check after it finishes:
  ```powershell
  py -3.14 -c "from tichu_training.bc.materialised import MemmapBCDataset; print('bc rows', MemmapBCDataset(r'C:\workbench\tichu\data\materialised_100k_v5\bc').total)"
  ```

---

## 2. Train BC (Policy Network)

Reads the packed BC bundle directly (`dataset: memmap` → `…\materialised_100k_v5\bc`).
Produces `checkpoints\step_000001.bin` (epoch-end) + mid-epoch saves.

```powershell
py -3.14 -m tichu_training.cli.train_bc `
  --config  configs\bc_full_100k_v5_memmap.yaml `
  --run-dir C:\workbench\tichu\data\runs\bc_full_100k_v5_memmap `
  --device  cuda -v
```

BC checkpoint produced:
`C:\workbench\tichu\data\runs\bc_full_100k_v5_memmap\checkpoints\step_000001.bin`

---

## 3. AWR refine (optional — produces the `master` checkpoint)

Continues from the BC checkpoint, fitting a Value Baseline and re-weighting by
advantage. Two value targets (run either or both); see ADR-0013 for the
round-vs-game trade-off. The model arch in these configs **must** match the BC
config exactly.

```powershell
# game_won target (the actual objective)
py -3.14 -m tichu_training.cli.train_bc `
  --config      configs\awr_full_100k_v5_memmap_game.yaml `
  --run-dir     C:\workbench\tichu\data\runs\awr_full_100k_v5_memmap_game `
  --refine-from C:\workbench\tichu\data\runs\bc_full_100k_v5_memmap\checkpoints\step_000001.bin `
  --device      cuda -v

# round_outcome target (diagnostic)
py -3.14 -m tichu_training.cli.train_bc `
  --config      configs\awr_full_100k_v5_memmap_round.yaml `
  --run-dir     C:\workbench\tichu\data\runs\awr_full_100k_v5_memmap_round `
  --refine-from C:\workbench\tichu\data\runs\bc_full_100k_v5_memmap\checkpoints\step_000001.bin `
  --device      cuda -v
```

> If `step_000001.bin` doesn't exist, point `--refine-from` at the latest
> `checkpoints\step_e000_b000*.bin` from the BC run.

---

## 4. Train Call Networks (Tichu + Grand-Tichu)

One run trains **both** networks → `checkpoints\tichu_final.bin` and
`checkpoints\grand_final.bin`.

```powershell
py -3.14 -m tichu_training.cli.train_calls `
  --config  configs\calls_full_v5.yaml `
  --run-dir C:\workbench\tichu\data\runs\calls_full_v5 -v
```

> Reads the packed `materialised_100k_v5\calls` bundle directly
> (`dataset: memmap` → `call_tichu` / `call_grand_tichu` slices) — no archive
> re-replay, so the config no longer sets `workers`.

---

## 5. Train Schupfen Network

```powershell
py -3.14 -m tichu_training.cli.train_schupfen `
  --config  configs\schupfen_full_v5.yaml `
  --run-dir C:\workbench\tichu\data\runs\schupfen_full_v5 -v
```

> Reads the packed `materialised_100k_v5\schupfen` bundle directly
> (`dataset: memmap`) — no archive re-parse, `workers` dropped.

---

## 6. Train Belief

`train_belief` now reads the replay-derived belief bundle via `dataset: memmap`
→ `MemmapBeliefDataset` (ADR-0021), in addition to the synthetic smoke path.
The bundle is gated behind an explicit `--bundle-tasks …,belief` on the parse
pass (step 1) and is play-scale (several GB). There is **no committed
`belief_full_v5.yaml` yet** — point a memmap config at
`materialised_100k_v5\belief` to train on real data; the loop stacks the whole
example list per epoch (no `iter_batches` fast path wired for belief yet).

Smoke run (synthetic), for completeness:

```powershell
py -3.14 -m tichu_training.cli.train_belief `
  --config  configs\belief_smoke.yaml `
  --run-dir C:\workbench\tichu\data\runs\belief_smoke -v
```

---

## 7. Evaluate

Both eval modes need an eval **config** (agent roster / checkpoint paths) that
you author — there's no checked-in 100k eval config. Point it at the BC (or AWR
`master`) + call + schupfen checkpoints from above.

```powershell
# Tournament (self-play strength, synthetic Starting-Position Pool)
py -3.14 -m tichu_training.cli.eval_matrix `
  --config configs\eval_100k_v5.yaml `
  --mode   tournament -v

# Move-Prediction Eval (imitation fidelity on held-out real BSW games)
py -3.14 -m tichu_training.cli.eval_matrix `
  --config   configs\eval_100k_v5.yaml `
  --mode     move_prediction `
  --held-out C:\workbench\tichu\data\heldout_tch -v
```

---

## 8. Export (TorchScript)

Bundles the Policy/Refined checkpoint with the two Call Networks for inference.
`--model-config` is the BC arch config.

```powershell
py -3.14 -m tichu_training.cli.export_model `
  --checkpoint       C:\workbench\tichu\data\runs\bc_full_100k_v5_memmap\checkpoints\step_000001.bin `
  --tichu-checkpoint C:\workbench\tichu\data\runs\calls_full_v5\checkpoints\tichu_final.bin `
  --grand-checkpoint C:\workbench\tichu\data\runs\calls_full_v5\checkpoints\grand_final.bin `
  --model-config     configs\bc_full_100k_v5_memmap.yaml `
  --format           torchscript `
  --output           C:\workbench\tichu\data\export\bc_full_100k_v5 `
  --benchmark -v
```

> Swap `--checkpoint` for an AWR `master` checkpoint
> (`…\runs\awr_full_100k_v5_memmap_game\checkpoints\…`) to export the refined
> Policy. Schupfen export isn't a flag on `export_model` yet (ADR-0012's
> `schupfen_path` is still inference-side TODO).

---

## Notes / gotchas

- **v4 is retired (ADR-0022).** Every v4-stamped checkpoint
  (`runs\bc_full_100k_v4_memmap`, `runs\awr_full_100k_v4_memmap_*`, any v4
  call/schupfen) fails to load against this code — retrain from v5. The old
  138 GB `materialised_100k_v4` can be deleted once v5 is validated.
- **All trainers now read the bundles.** BC + AWR (since v5), and now
  calls / schupfen / belief consume `materialised_100k_v5\{calls,schupfen,belief}`
  via `dataset: memmap` (`Memmap{Call,Schupfen,Belief}Dataset`) — no archive
  re-replay. The parquet path remains as a fallback in each trainer.
- **Run order:** ratings (0) → parse (1) → BC (2) → AWR (3, needs BC) →
  calls (4) + schupfen (5) can run any time after parse → eval (7) /
  export (8) need the trained checkpoints.
- **Disk:** the v5 BC bundle is ~17 GB (packed); calls/schupfen are single-digit
  GB; belief (if materialised) is play-scale (~several GB). Ensure headroom.
