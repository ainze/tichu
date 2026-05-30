"""BC training loop, CSV logging, and version-pinned checkpointing.

Two consumer shapes are supported:

  - `train_one_epoch(...)` over `Iterable[BCExample]` — the ADR-0011 path.
    Works with `SyntheticBCDataset`, `ParquetBCDataset`, and (for
    interop) `MemmapBCDataset.__iter__`. Per-example route into per-head
    buffers; `_to_tensors` builds the batch from a `list[BCExample]`.
  - `train_one_epoch_batched(...)` over the
    `Iterator[(decision_type, dict[str, np.ndarray])]` shape produced by
    `MemmapBCDataset.iter_batches`. The fast path of
    [ADR-0014](../../../docs/adr/0014-pre-featurise-bc-corpus.md):
    batches arrive already-stacked from the memmap, so the consumer
    skips per-example dataclass construction, `np.stack`, and the
    per-head buffering machinery. Same loss, CSV, checkpoint contract.

Both paths fire one optimizer step per batch and write one CSV row per
step. The batched path's progress bar counts examples (computed from
the yielded batch sizes) so the bar's meaning is identical.
"""

import csv
import io
import queue
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from tqdm import tqdm

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.dataset import BCExample
from tichu_training.bc.heads import BCModel, HEAD_LOGIT_DIMS
from tichu_training.bc.loss import masked_cross_entropy
from tichu_training.checkpoint import Checkpoint
from tichu_training.featurizer import FEATURIZER_VERSION


_CSV_FIELDS = (
    ["step", "loss_total"]
    + [f"loss_{h}" for h in HEAD_LOGIT_DIMS]
    + [f"acc_{h}" for h in HEAD_LOGIT_DIMS]
)


# Drop tqdm's built-in `{rate_fmt}` because we compute and display
# our own ex/s in the postfix (a 100-batch moving average rather than
# tqdm's EMA). Showing two ex/s readouts would be confusing.
_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
)


class _BarSmoothing:
    """Sliding-window stats for the training progress bar.

    Loss and ex/s are both jumpy at per-batch granularity — loss
    because of inter-head variance (one wish batch swings the readout)
    and ex/s because each batch's wall-time is noisy (disk fault,
    Python GC tick, GPU stream sync). A fixed-size window settles
    both: values older than `window` batches drop out cleanly, so
    the readout reflects recent state rather than the run's history.

    Why a true window over an EMA: tqdm's built-in `smoothing` is an
    EMA, which never forgets. An EMA-equivalent half-life of 100
    samples (smoothing ~ 0.02) still gives ~10% weight to data from
    250 batches ago — fine for monitoring overall trend but not what
    the user asked for here.
    """

    def __init__(self, window: int = 100) -> None:
        self._losses: deque[float] = deque(maxlen=window)
        self._times: deque[float] = deque(maxlen=window)
        self._sizes: deque[int] = deque(maxlen=window)

    def push(self, loss: float, batch_size: int) -> None:
        self._losses.append(loss)
        self._times.append(time.monotonic())
        self._sizes.append(batch_size)

    def loss_mean(self) -> float:
        if not self._losses:
            return float("nan")
        return sum(self._losses) / len(self._losses)

    def ex_per_sec(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        # Slight overcount: the first sample's batch_size was processed
        # *before* times[0] was recorded, so dividing by (times[-1] -
        # times[0]) attributes that work to a too-short span. Over 100
        # batches the bias is <1%; not worth the complication.
        return sum(self._sizes) / span


def _to_tensors(batch: Sequence[BCExample]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(np.stack([e.features for e in batch])),
        "target": torch.tensor([e.target for e in batch], dtype=torch.long),
        "legal_mask": torch.from_numpy(np.stack([e.legal_mask for e in batch])),
        "sample_weight": torch.tensor(
            [e.sample_weight for e in batch], dtype=torch.float32
        ),
        "skill_decile": torch.tensor(
            [e.skill_decile for e in batch], dtype=torch.long
        ),
    }


def train_one_epoch(
    model: BCModel,
    examples: Iterable[BCExample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    log_path: Path,
    head_weights: dict[str, float] | None = None,
    show_progress: bool = True,
    checkpoint_every_batches: int = 0,
    checkpoint_fn: Callable[[int], None] | None = None,
) -> float:
    """Train for one epoch over `examples`. Returns the final total loss.

    Accepts any `Iterable[BCExample]` — a list, a generator, or a streaming
    dataset. Examples are routed into per-head buffers; a buffer fires a
    training step the moment it fills to `batch_size`. Partial buffers at
    the end of the iterable fire one final step each.

    A tqdm progress bar is shown by default. When `examples` exposes
    `n_rows` (the case for ParquetBCDataset), the bar is bounded; otherwise
    it counts up. The postfix shows the last loss and per-head batch
    counts so you can see all three heads making progress.
    """
    head_weights = head_weights or {h: 1.0 for h in HEAD_LOGIT_DIMS}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    head_buffers: dict[str, list[BCExample]] = defaultdict(list)
    head_batch_counts: dict[str, int] = {h: 0 for h in HEAD_LOGIT_DIMS}
    last_total = float("inf")
    step = _next_step(log_path)
    stats = _BarSmoothing()

    # Snapshot the model's device once. Tensors built on the CPU side from
    # numpy arrays must be moved here before the forward pass when the
    # model is on CUDA.
    model_device = next(model.parameters()).device

    def _fire(head: str, batch: Sequence[BCExample]) -> None:
        nonlocal last_total, step
        tensors = _to_tensors(batch)
        if model_device.type != "cpu":
            tensors = {k: v.to(model_device, non_blocking=True) for k, v in tensors.items()}
        out = model(tensors["features"], tensors["skill_decile"])
        logits = out[head]
        per_head_loss = masked_cross_entropy(
            logits, tensors["target"], tensors["legal_mask"], tensors["sample_weight"],
        )
        total = head_weights[head] * per_head_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits.masked_fill(~tensors["legal_mask"], float("-inf")).argmax(dim=-1)
            acc = (pred == tensors["target"]).float().mean().item()
        row = {f: 0.0 for f in _CSV_FIELDS}
        row["step"] = step
        row["loss_total"] = float(total.detach())
        row[f"loss_{head}"] = float(per_head_loss.detach())
        row[f"acc_{head}"] = acc
        # Append immediately so a crash mid-epoch leaves valid partial progress
        # on disk and live monitors can tail the file during long runs.
        _append_csv(log_path, [row])
        last_total = row["loss_total"]
        head_batch_counts[head] = head_batch_counts.get(head, 0) + 1
        step += 1
        if (
            checkpoint_every_batches
            and checkpoint_fn is not None
            and step % checkpoint_every_batches == 0
        ):
            checkpoint_fn(step)

    total_known = getattr(examples, "n_rows", None) if show_progress else None
    bar = tqdm(
        total=total_known,
        unit="ex",
        dynamic_ncols=True,
        desc="train",
        disable=not show_progress,
        bar_format=_BAR_FORMAT,
    ) if show_progress else None

    try:
        try:
            for example in examples:
                buf = head_buffers[example.decision_type]
                buf.append(example)
                if len(buf) >= batch_size:
                    bs = len(buf)
                    _fire(example.decision_type, buf)
                    buf.clear()
                    stats.push(last_total, bs)
                    if bar is not None:
                        bar.set_postfix(
                            {
                                "loss": f"{stats.loss_mean():.3f}",
                                "ex/s": f"{stats.ex_per_sec():.0f}",
                                **{f"b_{h}": head_batch_counts[h] for h in HEAD_LOGIT_DIMS},
                            },
                            refresh=False,
                        )
                if bar is not None:
                    bar.update(1)
        finally:
            # Drain partial buffers — runs on normal completion AND on
            # KeyboardInterrupt / dataset exception so already-pulled
            # examples still produce a row. _fire writes per-batch, so
            # rows fired before any crash are already on disk.
            for head, buf in head_buffers.items():
                if buf:
                    bs = len(buf)
                    _fire(head, buf)
                    buf.clear()
                    stats.push(last_total, bs)
        if bar is not None:
            bar.set_postfix({
                "loss": f"{stats.loss_mean():.3f}",
                "ex/s": f"{stats.ex_per_sec():.0f}",
                **{f"b_{h}": head_batch_counts[h] for h in HEAD_LOGIT_DIMS},
            })
    finally:
        if bar is not None:
            bar.close()

    return last_total


def _prefetch_pinned(
    batches: Iterator[tuple[str, dict[str, np.ndarray]]],
    *,
    pin: bool,
    lookahead: int = 2,
) -> Iterator[tuple[str, dict[str, torch.Tensor]]]:
    """Wrap numpy batches as tensors on a background thread.

    With `pin=True`, the tensors land in pinned (page-locked) host
    memory so the consumer's subsequent `.to(cuda, non_blocking=True)`
    is an actual async H2D — overlapping with the previous batch's
    GPU compute instead of serialising as a `.to(...)` from pageable
    memory does.

    The producer runs `lookahead` batches ahead. The memmap fancy
    index (inside `iter_batches`) and the pin_memory memcpy both
    happen here, hidden behind the GPU step rather than between
    steps. With `pin=False` (CPU device) it's still a worker that
    moves numpy→tensor off the hot path.

    Errors in the producer surface on the next `__next__` call to
    the consumer; the worker is a daemon thread so a hard interrupt
    on the main loop won't hang process exit.
    """
    q: queue.Queue = queue.Queue(maxsize=lookahead)
    sentinel = object()

    def producer() -> None:
        try:
            for head, batch in batches:
                tensors = {
                    "features": torch.from_numpy(batch["features"]),
                    "target": torch.from_numpy(batch["target"]),
                    "legal_mask": torch.from_numpy(batch["legal_mask"]),
                    "sample_weight": torch.from_numpy(batch["sample_weight"]),
                    "skill_decile": torch.from_numpy(batch["skill_decile"]),
                }
                if pin:
                    tensors = {k: v.pin_memory() for k, v in tensors.items()}
                q.put((head, tensors))
        except BaseException as exc:
            q.put(("__error__", exc))
        finally:
            q.put(sentinel)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        head, payload = item
        if head == "__error__":
            raise payload
        yield head, payload


def train_one_epoch_batched(
    model: BCModel,
    batches: Iterator[tuple[str, dict[str, np.ndarray]]],
    optimizer: torch.optim.Optimizer,
    *,
    log_path: Path,
    head_weights: dict[str, float] | None = None,
    show_progress: bool = True,
    total_examples: int | None = None,
    checkpoint_every_batches: int = 0,
    checkpoint_fn: Callable[[int], None] | None = None,
) -> float:
    """Fast-path BC epoch over pre-stacked per-head batches.

    `batches` yields `(decision_type, batch_dict)` where `batch_dict`
    contains numpy arrays sized `(B, ...)` keyed exactly like the
    output of `_to_tensors`. Typically produced by
    `MemmapBCDataset.iter_batches`.

    Same loss / CSV-row / checkpoint semantics as `train_one_epoch` —
    one optimizer step per yielded batch, one row per step. The
    progress bar counts *examples* (sum of yielded batch sizes), not
    batches, so its meaning matches the per-example path even though
    no per-example route happens here.

    Why this exists: per the 2026-05-29 perf measurement, the
    `__iter__` path through `MemmapBCDataset` delivers ~43% of the
    in-RAM Phase A ceiling for the current full-BC arch — the gap is
    Python object construction (BCExample dataclass + `np.stack` in
    `_to_tensors`). The batched path bypasses both, recovering most of
    the ceiling. See ADR-0014.
    """
    head_weights = head_weights or {h: 1.0 for h in HEAD_LOGIT_DIMS}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    head_batch_counts: dict[str, int] = {h: 0 for h in HEAD_LOGIT_DIMS}
    last_total = float("inf")
    step = _next_step(log_path)
    stats = _BarSmoothing()
    model_device = next(model.parameters()).device

    def _fire(head: str, tensors: dict[str, torch.Tensor]) -> None:
        nonlocal last_total, step
        # Tensors arrive pre-wrapped (and pinned, for CUDA) from
        # `_prefetch_pinned`. With pinned host memory,
        # `non_blocking=True` makes the H2D actually async, so it
        # overlaps with the previous batch's GPU step rather than
        # blocking the main thread.
        if model_device.type != "cpu":
            tensors = {
                k: v.to(model_device, non_blocking=True)
                for k, v in tensors.items()
            }
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
        with torch.no_grad():
            pred = logits.masked_fill(
                ~tensors["legal_mask"], float("-inf"),
            ).argmax(dim=-1)
            acc = (pred == tensors["target"]).float().mean().item()
        row = {f: 0.0 for f in _CSV_FIELDS}
        row["step"] = step
        row["loss_total"] = float(total.detach())
        row[f"loss_{head}"] = float(per_head_loss.detach())
        row[f"acc_{head}"] = acc
        _append_csv(log_path, [row])
        last_total = row["loss_total"]
        head_batch_counts[head] = head_batch_counts.get(head, 0) + 1
        step += 1
        if (
            checkpoint_every_batches
            and checkpoint_fn is not None
            and step % checkpoint_every_batches == 0
        ):
            checkpoint_fn(step)

    bar = tqdm(
        total=total_examples,
        unit="ex",
        dynamic_ncols=True,
        desc="train",
        disable=not show_progress,
        bar_format=_BAR_FORMAT,
    ) if show_progress else None

    # Pin only when shipping to CUDA. Pinned host memory is a real
    # resource (and bounded by the OS); allocating it for CPU runs
    # gains nothing. Lookahead of 2 hides one full gather + pin
    # behind one GPU step — enough to close the visible gap.
    prefetched = _prefetch_pinned(batches, pin=(model_device.type == "cuda"))
    try:
        try:
            for head, tensors in prefetched:
                bs = tensors["target"].shape[0]
                _fire(head, tensors)
                stats.push(last_total, bs)
                if bar is not None:
                    bar.set_postfix(
                        {
                            "loss": f"{stats.loss_mean():.3f}",
                            "ex/s": f"{stats.ex_per_sec():.0f}",
                            **{f"b_{h}": head_batch_counts[h] for h in HEAD_LOGIT_DIMS},
                        },
                        refresh=False,
                    )
                    bar.update(bs)
        finally:
            # No per-head partial-buffer drain — `batches` is the source
            # of truth for batch boundaries (iter_batches handles
            # drop_last). Nothing to flush here.
            pass
        if bar is not None:
            bar.set_postfix({
                "loss": f"{stats.loss_mean():.3f}",
                "ex/s": f"{stats.ex_per_sec():.0f}",
                **{f"b_{h}": head_batch_counts[h] for h in HEAD_LOGIT_DIMS},
            })
    finally:
        if bar is not None:
            bar.close()

    return last_total


def _next_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1  # subtract header


def _append_csv(log_path: Path, rows: Iterable[dict[str, float]]) -> None:
    new = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_FIELDS))
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str | Path,
) -> None:
    buf = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        buf,
    )
    Checkpoint(
        featurizer_version=FEATURIZER_VERSION,
        action_space_version=ACTION_SPACE_VERSION,
        payload=buf.getvalue(),
    ).save(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    cp = Checkpoint.load(
        path,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    state = torch.load(io.BytesIO(cp.payload), weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])
