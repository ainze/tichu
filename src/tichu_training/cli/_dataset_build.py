"""Shared `dataset: memmap` construction for the standalone trainers.

`train_calls` / `train_schupfen` / `train_belief` each grew a near-identical
`memmap` branch in their `_build_dataset`: open the task's packed-bundle reader
over `data_dir`, honour an optional `materialize` flag, return the dataset. This
collapses that branch to one place so the bundle-reading contract (ADR-0020 /
ADR-0021) lives once.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping


def build_memmap_dataset(
    reader_cls: Callable[..., Any],
    dataset_kwargs: Mapping[str, Any],
    *,
    force_materialize: bool = False,
    **reader_kwargs: Any,
) -> Any:
    """Open `reader_cls` over a packed bundle for a `dataset: memmap` config.

    `dataset_kwargs` is the config's `dataset_kwargs` block (carries `data_dir`
    and any version-pin overrides). A `materialize: true` key in it — popped
    here — forces the examples into an in-memory list; so does
    `force_materialize` (for trainers whose loop has no `iter_batches` fast
    path and must re-read every epoch). Otherwise the re-iterable dataset
    object is returned, matching the parquet branch. Extra `reader_kwargs`
    (e.g. the calls bundle's `call_type`) pass straight to the reader.
    """
    kwargs = dict(dataset_kwargs)
    materialize = force_materialize or bool(kwargs.pop("materialize", False))
    ds = reader_cls(**kwargs, **reader_kwargs)
    return list(ds) if materialize else ds
