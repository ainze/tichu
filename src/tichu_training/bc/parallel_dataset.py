"""Multiprocessing-parallel variant of ParquetBCDataset.

The sequential `ParquetBCDataset` is single-threaded: parse, replay,
featurize, legal-mask, yield. On a 12-core machine running 100k games of
BSW data, that pegs one core and leaves 11 idle — total CPU utilisation
stays around 40% (1 core for the data loop + a few MKL threads for each
batch's brief forward/backward).

`ParallelParquetBCDataset` shards the manifest across N worker processes.
Each worker owns a stable hash-based slice of the game-id space, runs the
existing per-game replay loop on its slice, and pushes `BCExample`s
through a bounded `multiprocessing.Queue` into the trainer. CPU
utilisation scales toward 100% × N cores; wall-clock per epoch drops
roughly proportionally on a CPU-bound workload.

The queue is bounded — once it fills, workers block on `put()` and back
off naturally. Workers check a shared `stop_event` periodically so the
trainer can shut them down on early exit (e.g. when paired with
`--max-examples` or Ctrl+C).
"""

import logging
import multiprocessing as mp
import os
import queue as queue_module
from pathlib import Path
from typing import Iterator

from tichu_training.bc.dataset import (
    BCExample,
    ParquetBCDataset,
    _DEFAULT_RECENCY_CUTOFF,
    _DEFAULT_RECENCY_WEIGHT,
    _KIND_TO_DECISION_TYPE,
)


log = logging.getLogger("tichu_training.bc.parallel_dataset")


# Sentinel sent on the queue by each worker when it has drained its slice.
# A bare object() doesn't pickle reliably across processes — use a string
# sentinel that's distinct from any BCExample.
_WORKER_DONE = "__WORKER_DONE__"


def _shard_manifest(
    manifest: dict[str, set[int]], num_workers: int,
) -> list[dict[str, set[int]]]:
    """Split the (game_id -> set[round_id]) manifest into `num_workers`
    disjoint shards by stable hash. Hash modulo means the same game
    always lands in the same shard — useful for reproducibility and for
    keeping the (round-replay) cost balanced across workers."""
    shards: list[dict[str, set[int]]] = [dict() for _ in range(num_workers)]
    for game_id, rounds in manifest.items():
        # `hash()` is stable within a Python session but randomised across
        # sessions for str (PYTHONHASHSEED). For our purposes that's fine —
        # we don't need cross-run reproducibility of the shard assignment,
        # only that within one run a game lands in exactly one shard.
        idx = hash(game_id) % num_workers
        shards[idx][game_id] = rounds
    return shards


def _worker_loop(
    worker_id: int,
    archive_path: str,
    manifest_subset: dict[str, set[int]],
    skill_lookup: dict[str, int],
    recency_cutoff: int,
    recency_weight: float,
    neutral_decile: int,
    queue: "mp.Queue",
    stop_event,
) -> None:
    """Per-worker entry point. Iterates the worker's slice of the archive,
    replays each in-manifest round, yields BCExamples through `queue`.
    Sends `_WORKER_DONE` when its slice is exhausted (or `stop_event`
    fires)."""
    # Deferred imports inside the worker so the parent's import cost
    # doesn't double-count and so the worker boots fast.
    try:
        from tichu_training.bsw.archive import iter_archive
        from tichu_training.bsw.parser import parse_tch
        from tichu_training.bsw.replay import replay_round
        from tichu_training.action_space import bc_target_for_concrete, legal_mask
        from tichu_training.featurizer import featurize
        from tichu_engine.state import DragonGivePending

        game_ids = set(manifest_subset.keys())
        for stem, text in iter_archive(Path(archive_path), game_ids=game_ids):
            if stop_event.is_set():
                break
            if stem not in manifest_subset:
                continue
            try:
                game = parse_tch(text, game_id=stem)
            except Exception:  # noqa: BLE001
                continue
            sample_weight = _sample_weight_for(
                stem, recency_cutoff, recency_weight,
            )
            valid_rounds = manifest_subset[stem]
            for parsed_round in game.rounds:
                if stop_event.is_set():
                    break
                if parsed_round.round_index not in valid_rounds:
                    continue
                replay = replay_round(parsed_round)
                if replay.final_state is None:
                    continue
                team_outcome = float(
                    parsed_round.ergebnis[0] - parsed_round.ergebnis[1],
                )
                for (parsed_action, concrete), pre_state, cached_actions in zip(
                    replay.decisions,
                    replay.pre_decision_states,
                    replay.legal_actions_at,
                ):
                    if pre_state is None:
                        continue
                    decision_type = _KIND_TO_DECISION_TYPE.get(parsed_action.kind)
                    if decision_type is None:
                        continue
                    player = parsed_action.player
                    if not 0 <= player < 4:
                        continue
                    private = pre_state.private_view(player)
                    features = featurize(private)
                    if decision_type == "dragon_assignment":
                        pending = pre_state.public.pending_decision
                        if not isinstance(pending, DragonGivePending):
                            continue
                        try:
                            target = bc_target_for_concrete(
                                decision_type, concrete,
                                winner_seat=pending.winner,
                            )
                        except ValueError:
                            continue
                    else:
                        try:
                            target = bc_target_for_concrete(
                                decision_type, concrete,
                            )
                        except ValueError:
                            continue
                    mask = legal_mask(
                        decision_type, pre_state, player,
                        cached_actions=cached_actions,
                    )
                    if not mask[target]:
                        continue
                    handle = parsed_round.handles[player]
                    skill = skill_lookup.get(handle, neutral_decile)
                    example = BCExample(
                        decision_type=decision_type,
                        features=features,
                        target=int(target),
                        legal_mask=mask,
                        sample_weight=sample_weight,
                        skill_decile=skill,
                        round_outcome=team_outcome,
                    )
                    # Blocking put with periodic stop_event check so the
                    # worker can be torn down mid-game when the trainer
                    # stops consuming.
                    while not stop_event.is_set():
                        try:
                            queue.put(example, timeout=0.5)
                            break
                        except queue_module.Full:
                            continue
    except Exception as exc:  # noqa: BLE001
        # Send the exception across the queue so the parent can re-raise
        # with context rather than hanging on a dropped worker.
        try:
            queue.put(("__WORKER_ERROR__", worker_id, repr(exc)), timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            queue.put(_WORKER_DONE, timeout=5.0)
        except Exception:  # noqa: BLE001
            pass


def _sample_weight_for(game_id: str, cutoff: int, low_weight: float) -> float:
    try:
        gid = int(game_id)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if gid >= cutoff else low_weight


class ParallelParquetBCDataset:
    """Worker-parallel variant of ParquetBCDataset.

    Construction reuses ParquetBCDataset for manifest build + skill
    lookup (the cheap parts that happen once). `__iter__` spawns
    `num_workers` processes, each handling a disjoint slice of the
    manifest, and yields BCExamples from a shared bounded queue.

    `num_workers` defaults to `max(1, os.cpu_count() - 1)` — leave one
    core for the trainer + model.
    """

    def __init__(
        self,
        shards_dir: str | Path,
        *,
        archive_path: str | Path,
        expected_featurizer_version: str,
        expected_action_space_version: str,
        ratings_path: str | Path | None = None,
        recency_cutoff_game_id: int = _DEFAULT_RECENCY_CUTOFF,
        recency_weight: float = _DEFAULT_RECENCY_WEIGHT,
        skill_buckets: int = 10,
        num_workers: int | None = None,
        queue_maxsize: int | None = None,
    ) -> None:
        # Reuse ParquetBCDataset for the (relatively cheap) one-time
        # manifest build + ratings load + version-pin verification.
        self._sequential = ParquetBCDataset(
            shards_dir,
            archive_path=archive_path,
            expected_featurizer_version=expected_featurizer_version,
            expected_action_space_version=expected_action_space_version,
            ratings_path=ratings_path,
            recency_cutoff_game_id=recency_cutoff_game_id,
            recency_weight=recency_weight,
            skill_buckets=skill_buckets,
        )
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 2) - 1)
        self.num_workers = int(num_workers)
        # Queue size: a few batches per worker keeps the pipeline warm
        # without bloating RAM. ~66 KB per BCExample × queue_maxsize is
        # the upper bound on inflight memory.
        if queue_maxsize is None:
            queue_maxsize = max(1024, self.num_workers * 256)
        self.queue_maxsize = int(queue_maxsize)

    @property
    def n_rows(self) -> int:
        # The trainer's tqdm reads this for the bounded progress bar.
        # The underlying parquet row count is an upper bound on the
        # actual yielded BCExample count (some rows are dropped at
        # iteration time — None pre_decision_states, etc.).
        return self._sequential.n_rows

    @property
    def manifest_size(self) -> int:
        return self._sequential.manifest_size

    def __iter__(self) -> Iterator[BCExample]:
        shards = _shard_manifest(
            self._sequential._manifest, self.num_workers,
        )
        log.info(
            "spawning %d worker(s); shard sizes: %s",
            self.num_workers,
            [len(s) for s in shards],
        )
        # `spawn` (the default on Windows) is the safest cross-platform
        # start method. `fork` would also work on Linux but we want a
        # uniform contract.
        ctx = mp.get_context("spawn")
        queue: "mp.Queue" = ctx.Queue(maxsize=self.queue_maxsize)
        stop_event = ctx.Event()
        workers: list[mp.Process] = []
        for i, shard in enumerate(shards):
            p = ctx.Process(
                target=_worker_loop,
                name=f"bc-worker-{i}",
                args=(
                    i,
                    str(self._sequential.archive_path),
                    shard,
                    self._sequential._skill_lookup,
                    self._sequential.recency_cutoff_game_id,
                    self._sequential.recency_weight,
                    self._sequential._neutral_decile,
                    queue,
                    stop_event,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        done_count = 0
        try:
            while done_count < self.num_workers:
                try:
                    item = queue.get(timeout=1.0)
                except queue_module.Empty:
                    # Periodically check that workers are still alive —
                    # a crashed worker without a sentinel would otherwise
                    # hang us forever.
                    if all(not p.is_alive() for p in workers):
                        # All dead; drain any remaining queue items and
                        # then bail.
                        while True:
                            try:
                                item = queue.get(timeout=0.1)
                            except queue_module.Empty:
                                break
                            if item == _WORKER_DONE:
                                done_count += 1
                            elif isinstance(item, tuple) and item and item[0] == "__WORKER_ERROR__":
                                _, wid, msg = item
                                raise RuntimeError(
                                    f"BC dataset worker {wid} crashed: {msg}"
                                )
                            else:
                                yield item
                        break
                    continue
                if item == _WORKER_DONE:
                    done_count += 1
                    continue
                if isinstance(item, tuple) and item and item[0] == "__WORKER_ERROR__":
                    _, wid, msg = item
                    raise RuntimeError(
                        f"BC dataset worker {wid} crashed: {msg}"
                    )
                yield item
        finally:
            # Tell workers to stop and join. Done both on normal exhaustion
            # and on early termination (e.g. trainer hit --max-examples or
            # the consumer raised).
            stop_event.set()
            for p in workers:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2.0)
