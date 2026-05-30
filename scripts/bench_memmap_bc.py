r"""Phase B bench — measure disk-backed BC training throughput.

Opens a directory produced by `materialise_bc_subset.py`, memmaps the
per-type files, and runs the same BC training consumer that
`spike_bc_ceiling.py` measures the in-RAM ceiling for. Reports ex/s.

Comparison: Phase A ceiling (in-RAM cache) vs Phase B delivered
(disk-backed). The gap quantifies what disk delivery costs relative to
the consumer's compute-only ceiling.

The reader yields `BCExample`s in the original emission order recorded
during materialisation (via `order.dat`). Features and legal_mask are
memmap views, so per-example construction is zero-copy; the per-batch
`np.stack` in `_to_tensors` is the only memcpy from disk pages into the
batch tensor.

Usage (PowerShell):

  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\bench_memmap_bc.py `
      --data-dir C:\workbench\tichu\data\materialised_smoke `
      --device cuda --warmup 50000 --measure 200000 --arch current
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from tichu_training.awr.refine import _to_tensors
from tichu_training.bc.dataset import BCExample
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS
from tichu_training.bc.loss import masked_cross_entropy
from tichu_training.featurizer import FEATURIZER_VERSION


# Mirrors spike_bc_ceiling.py — same presets, same defaults.
ARCH_PRESETS = {
    "current": dict(
        trunk_hidden=1024, trunk_depth=4, trunk_out_dim=512,
        skill_dim=64, head_hidden=256, batch_size=1024, lr=3.0e-4,
    ),
    "wider": dict(
        trunk_hidden=2048, trunk_depth=6, trunk_out_dim=512,
        skill_dim=64, head_hidden=512, batch_size=1024, lr=1.5e-4,
    ),
    "smoke": dict(
        trunk_hidden=256, trunk_depth=2, trunk_out_dim=128,
        skill_dim=32, head_hidden=64, batch_size=256, lr=1.0e-4,
    ),
}


class MemmapBCDataset:
    """Disk-backed BC dataset.

    Opens the per-type memmaps written by `materialise_bc_subset.py`
    and yields `BCExample`s in the original emission order recorded in
    `order.dat`. Per-row reads are memmap slice views — no copies until
    the consumer's `np.stack` builds a batch tensor.

    Version-pin guard: refuses to load if the manifest's featurizer or
    action-space versions don't match the live code. ADR-0011's
    `expected_*_version` mechanism on `ParquetBCDataset` is the model.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.json under {data_dir}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["featurizer_version"] != FEATURIZER_VERSION:
            raise RuntimeError(
                f"featurizer version mismatch: bundle="
                f"{self.manifest['featurizer_version']!r} live="
                f"{FEATURIZER_VERSION!r}"
            )
        self.type_order: list[str] = self.manifest["type_order"]
        self.feature_dim: int = self.manifest["feature_dim"]
        self.counts: dict[str, int] = self.manifest["counts"]
        self.head_logit_dims: dict[str, int] = self.manifest["head_logit_dims"]

        # Build the meta dtype from the manifest's recorded fields so a
        # future bundle with extra fields can still load (forward-compat
        # within reason).
        self._meta_dtype = np.dtype([
            (n, np.dtype(t)) for n, t in self.manifest["meta_dtype"]
        ])

        # Feature reconstruction plan (schema v2: indicator columns packed,
        # continuous columns kept f32). Mirrors MemmapBCDataset.
        cont_cols = list(self.manifest["continuous_feature_columns"])
        cont_set = set(cont_cols)
        bin_cols = [c for c in range(self.feature_dim) if c not in cont_set]
        self._n_bin = len(bin_cols)
        self._feat_bits_bytes = (self._n_bin + 7) // 8
        self._n_cont = len(cont_cols)
        self._bin_cols = np.asarray(bin_cols, dtype=np.intp)
        self._cont_cols = np.asarray(cont_cols, dtype=np.intp)

        # Memmap each per-type file.
        self._feat_bits: dict[str, np.memmap] = {}
        self._feat_cont: dict[str, np.memmap] = {}
        self._mask: dict[str, np.memmap] = {}
        self._meta: dict[str, np.memmap] = {}
        for type_name, info in self.manifest["files"].items():
            n_rows = self.counts[type_name]
            mask_dim = self.head_logit_dims[type_name]
            self._feat_bits[type_name] = np.memmap(
                self.data_dir / info["feat_bits"],
                dtype=np.uint8, mode="r",
                shape=(n_rows, self._feat_bits_bytes),
            )
            self._feat_cont[type_name] = np.memmap(
                self.data_dir / info["feat_cont"],
                dtype=np.float32, mode="r",
                shape=(n_rows, self._n_cont),
            )
            self._mask[type_name] = np.memmap(
                self.data_dir / info["legal_mask"],
                dtype=np.uint8, mode="r",
                shape=(n_rows, (mask_dim + 7) // 8),  # packbits, schema v2
            )
            self._meta[type_name] = np.memmap(
                self.data_dir / info["meta"],
                dtype=self._meta_dtype, mode="r",
                shape=(n_rows,),
            )

        # Walk order.dat to get the (type_idx, row_idx) sequence.
        self._order = np.memmap(
            self.data_dir / self.manifest["order_file"],
            dtype=np.uint32, mode="r",
            shape=(self.manifest["total"], 2),
        )

    @property
    def n_rows(self) -> int:
        return int(self.manifest["total"])

    def _reconstruct_row(self, type_name: str, row_idx: int) -> np.ndarray:
        feat = np.empty(self.feature_dim, dtype=np.float32)
        feat[self._bin_cols] = np.unpackbits(
            self._feat_bits[type_name][row_idx]
        )[: self._n_bin]
        feat[self._cont_cols] = self._feat_cont[type_name][row_idx]
        return feat

    def __iter__(self) -> Iterator[BCExample]:
        type_order = self.type_order
        # Snapshot to locals — saves dict lookup overhead per row.
        masks = self._mask
        meta = self._meta
        order = self._order
        mask_dims = self.head_logit_dims
        for i in range(len(order)):
            type_idx = int(order[i, 0])
            row_idx = int(order[i, 1])
            type_name = type_order[type_idx]
            m = meta[type_name][row_idx]
            game_won_code = int(m["game_won"])
            yield BCExample(
                decision_type=type_name,
                features=self._reconstruct_row(type_name, row_idx),
                target=int(m["target"]),
                legal_mask=np.unpackbits(masks[type_name][row_idx])[
                    : mask_dims[type_name]
                ].astype(bool),
                sample_weight=float(m["sample_weight"]),
                skill_decile=int(m["skill_decile"]),
                round_outcome=float(m["round_outcome"]),
                game_won=(None if game_won_code == -1 else bool(game_won_code)),
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=50_000,
                   help="Examples to consume before starting the timer.")
    p.add_argument("--measure", type=int, default=200_000,
                   help="Examples to consume in the timed window.")
    p.add_argument("--arch", choices=sorted(ARCH_PRESETS), default="current")
    args = p.parse_args()

    arch = ARCH_PRESETS[args.arch]
    print(f"arch={args.arch}  {arch}")

    device = torch.device(args.device)

    # -----------------------------------------------------------------
    # Step 1 — open the memmapped dataset. No data loaded into RAM yet.
    # -----------------------------------------------------------------
    print(f"opening memmap dataset at {args.data_dir} ...")
    ds = MemmapBCDataset(args.data_dir)
    print(f"  total {ds.n_rows} rows, counts={ds.counts}")
    if ds.n_rows < args.warmup + args.measure:
        raise SystemExit(
            f"dataset has only {ds.n_rows} rows but --warmup + --measure = "
            f"{args.warmup + args.measure} — re-materialise with a larger "
            f"--max-examples or reduce the bench window"
        )

    # -----------------------------------------------------------------
    # Step 2 — build the model (random init, throughput-only bench).
    # -----------------------------------------------------------------
    model = BCModel(
        feature_dim=ds.feature_dim,
        skill_buckets=11,
        skill_dim=arch["skill_dim"],
        trunk_hidden=arch["trunk_hidden"],
        trunk_depth=arch["trunk_depth"],
        trunk_out_dim=arch["trunk_out_dim"],
        head_hidden=arch["head_hidden"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params on {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=arch["lr"])
    head_weights = {h: 1.0 for h in HEAD_LOGIT_DIMS}
    batch_size = arch["batch_size"]

    head_buffers: dict[str, list[BCExample]] = defaultdict(list)

    def _fire(head: str, batch: list[BCExample]) -> None:
        tensors = _to_tensors(batch)
        if device.type != "cpu":
            tensors = {k: v.to(device, non_blocking=True) for k, v in tensors.items()}
        out = model(tensors["features"], tensors["skill_decile"])
        logits = out[head]
        per_head_loss = masked_cross_entropy(
            logits, tensors["target"], tensors["legal_mask"],
            tensors["sample_weight"],
        )
        total = head_weights[head] * per_head_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    # -----------------------------------------------------------------
    # Step 3 — iterate once: warmup window, then timed measure window.
    # -----------------------------------------------------------------
    it = iter(ds)
    print(f"warmup: consuming {args.warmup} examples ...")
    t_warm = time.perf_counter()
    for _ in range(args.warmup):
        ex = next(it)
        head_buf = head_buffers[ex.decision_type]
        head_buf.append(ex)
        if len(head_buf) >= batch_size:
            _fire(ex.decision_type, head_buf)
            head_buf.clear()
    if device.type != "cpu":
        torch.cuda.synchronize()
    warm_elapsed = time.perf_counter() - t_warm
    print(f"  warmup: {args.warmup} ex in {warm_elapsed:.2f}s "
          f"({args.warmup / max(1e-9, warm_elapsed):.0f} ex/s incl first-page reads)")

    print(f"measuring: consuming {args.measure} examples ...")
    consumed = 0
    last_n = 0
    last_t = time.perf_counter()
    t0 = last_t
    progress_every = max(batch_size * 4, args.measure // 10)
    while consumed < args.measure:
        ex = next(it)
        head_buf = head_buffers[ex.decision_type]
        head_buf.append(ex)
        consumed += 1
        if len(head_buf) >= batch_size:
            _fire(ex.decision_type, head_buf)
            head_buf.clear()
        if consumed - last_n >= progress_every:
            now = time.perf_counter()
            inst = (consumed - last_n) / max(1e-9, now - last_t)
            print(f"  ... {consumed}/{args.measure}  inst={inst:.0f} ex/s")
            last_n = consumed
            last_t = now
    # Drain partial buffers.
    for head, head_buf in list(head_buffers.items()):
        if head_buf:
            _fire(head, head_buf)
            head_buf.clear()
    if device.type != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    rate = args.measure / max(1e-9, elapsed)
    print(f"\nDELIVERED ({args.arch}): {args.measure} ex in {elapsed:.2f}s "
          f"= {rate:.0f} ex/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
