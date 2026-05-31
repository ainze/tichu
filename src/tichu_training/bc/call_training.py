"""Datasets + training loop for the Tichu/Grand Tichu call networks.

`SyntheticCallDataset` mixes positives and negatives at a configurable rate
for the smoke path. `ParquetCallDataset` is the production source — it reuses
the play shard as a manifest of validated rounds (ADR-0008 makes validation
round-granular, so the play shard's `(game_id, round_id)` set equals every
other decision-type shard's) and synthesises negatives at iteration time by
enumerating all four seats per validated round. The call shards from #007
(positive-only) are not read by this dataset; they remain for analysis.

Per-game replay + featurise is the CPU-bound bottleneck (featurize ~half the
cost, unaffected by the Tichu early-stop). `ParquetCallDataset(workers=N)`
fans that work over a ProcessPoolExecutor; the shared per-game routine
`_call_examples_for_game` keeps `workers=1` and `workers=N` byte-identical.
"""

import csv
import hashlib
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from tichu_training.bc.call_model import CallNetwork


log = logging.getLogger("tichu_training.bc.call_training")


_DEFAULT_RECENCY_CUTOFF: int = 1855844
_DEFAULT_RECENCY_WEIGHT: float = 0.5
_NEUTRAL_SKILL_DECILE: int = 10  # matches ParquetBCDataset


def _all_seats_have_first_play(replay_result) -> bool:
    """Early-stop predicate for the Tichu replay: True once every seat 0..3
    has at least one non-Pass Play with a non-None pre-decision state.

    Rescans the (short, at-stop-time) decisions list each call rather than
    threading incremental state through `replay_round` — the loop halts after
    trick 1's first go-around, so the list has only a handful of entries."""
    seats: set[int] = set()
    for (action, _concrete), pre_state in zip(
        replay_result.decisions, replay_result.pre_decision_states,
    ):
        if pre_state is not None and action.kind == "play":
            seats.add(action.player)
            if len(seats) == 4:
                return True
    return False


@dataclass
class CallExample:
    features: np.ndarray
    target: int  # 0 = no-call, 1 = call
    skill_decile: int
    sample_weight: float = 1.0
    # game_id: stable key for game-level train/val splitting (string, hashed),
    # AND provenance carried into the materialised bundle meta (ADR-0020).
    # round_id is bundle provenance only. Empty/0 for synthetic / standalone.
    game_id: str = ""
    round_id: int = 0


class SyntheticCallDataset(Iterable[CallExample]):
    """Deterministic call examples; mixes positives at the requested rate."""

    def __init__(
        self,
        *,
        seed: int = 0,
        n_examples: int = 200,
        positive_rate: float = 0.3,
        feature_dim: int | None = None,
        skill_buckets: int = 10,
        binary_features: bool = False,
    ) -> None:
        if feature_dim is None:
            from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
            feature_dim = FEATURIZER_OUTPUT_DIM
        self.seed = seed
        self.n_examples = n_examples
        self.positive_rate = positive_rate
        self.feature_dim = feature_dim
        self.skill_buckets = skill_buckets
        # 0/1 indicator columns + float continuous columns — the value
        # contract the materialised bundle's bit-packer requires. Mirrors
        # `SyntheticBCDataset`. Default False keeps the all-Gaussian features.
        self.binary_features = binary_features

    def __iter__(self) -> Iterator[CallExample]:
        rng = np.random.default_rng(self.seed)
        cont_idx = None
        if self.binary_features:
            from tichu_training.featurizer import CONTINUOUS_FEATURE_COLUMNS
            cont_idx = np.asarray(
                [c for c in CONTINUOUS_FEATURE_COLUMNS if c < self.feature_dim],
                dtype=np.intp,
            )
        for _ in range(self.n_examples):
            target = int(rng.random() < self.positive_rate)
            if self.binary_features:
                features = (rng.random(self.feature_dim) < 0.5).astype(np.float32)
                if cont_idx.size:
                    features[cont_idx] = rng.standard_normal(
                        cont_idx.size
                    ).astype(np.float32)
            else:
                features = rng.standard_normal(self.feature_dim).astype(np.float32)
            # 10% cold-start; otherwise rated.
            if rng.random() < 0.1:
                skill = self.skill_buckets
            else:
                skill = int(rng.integers(0, self.skill_buckets))
            yield CallExample(features=features, target=target, skill_decile=skill)


def _sample_weight(game_id: str, recency_cutoff: int, recency_weight: float) -> float:
    try:
        gid = int(game_id)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if gid >= recency_cutoff else recency_weight


def _call_examples_for_game(
    stem: str,
    text: str,
    valid_rounds: set[int],
    *,
    call_type: str,
    skill_lookup: dict[str, int],
    neutral_decile: int,
    recency_cutoff: int,
    recency_weight: float,
) -> list[CallExample]:
    """Produce every CallExample for one game's validated rounds.

    Module-level and self-contained (no `self`) so it is picklable and can run
    in a ProcessPoolExecutor worker. Both the serial and parallel iteration
    paths call this, which guarantees `workers=1` and `workers=N` yield
    identical examples (only the per-game order may differ in parallel).
    """
    from tichu_training.bc.call_emit import (
        grand_tichu_examples_for_round,
        tichu_examples_for_round,
    )
    from tichu_training.bsw.parser import parse_tch

    out: list[CallExample] = []
    try:
        game = parse_tch(text, game_id=stem)
    except Exception:  # noqa: BLE001
        return out
    sample_weight = _sample_weight(stem, recency_cutoff, recency_weight)

    if call_type == "tichu":
        from tichu_training.bsw.replay import replay_round

    for parsed_round in game.rounds:
        if parsed_round.round_index not in valid_rounds:
            continue

        if call_type == "grand_tichu":
            out.extend(grand_tichu_examples_for_round(
                parsed_round,
                skill_lookup=skill_lookup,
                neutral_decile=neutral_decile,
                sample_weight=sample_weight,
                game_id=stem,
                round_id=parsed_round.round_index,
            ))
            continue

        # Tichu: featurise at each seat's first non-Pass Play state (ADR-0018).
        # Early-stop once all four seats have made a first non-Pass Play — the
        # rest of the round is unused, and the per-step legal-action
        # enumeration is the dominant replay cost. The consolidated parse pass
        # instead reuses its full replay (see call_emit.tichu_examples_for_round).
        replay = replay_round(
            parsed_round, early_stop=_all_seats_have_first_play,
        )
        if replay.final_state is None:
            continue  # round failed replay; manifest disagrees, skip
        out.extend(tichu_examples_for_round(
            parsed_round, replay,
            skill_lookup=skill_lookup,
            neutral_decile=neutral_decile,
            sample_weight=sample_weight,
            game_id=stem,
            round_id=parsed_round.round_index,
        ))
    return out


# Per-worker config, populated once per process by the pool initializer so the
# (potentially large) skill_lookup dict is pickled once per worker, not once
# per submitted game.
_CALL_WORKER_CFG: dict = {}


def _init_call_worker(cfg: dict) -> None:
    _CALL_WORKER_CFG.clear()
    _CALL_WORKER_CFG.update(cfg)


def _call_worker(chunk: list[tuple[str, str, tuple[int, ...]]]) -> list[CallExample]:
    """Process K games per task to amortise the per-submit pickle / IPC cost.

    A single-game submit costs O(ms) in dispatcher overhead — non-trivial when
    each game's worker time is ~80ms — and caps throughput well below worker
    capacity. Bundling K games per task cuts the per-game overhead K-fold;
    benched at ~3x throughput on a 12-core box at K=10.
    """
    out: list[CallExample] = []
    for stem, text, valid_rounds in chunk:
        out.extend(_call_examples_for_game(
            stem, text, set(valid_rounds), **_CALL_WORKER_CFG,
        ))
    return out


class ParquetCallDataset(Iterable[CallExample]):
    """Archive-driven negative synthesis for a single call type.

    Manifest is built from the play shard (per Q6 of the design discussion:
    ADR-0008 makes validation round-granular, so the play shard's
    `(game_id, round_id)` pairs cover every validated round — including those
    where nobody called the relevant Tichu type, which contribute only
    negatives). At iteration time, for each validated round:

      * Grand-Tichu: build a synthetic deal-time `GameState` from
        `parsed_round.pre_deal_hands` (8 cards each), featurise each seat's
        `private_view`, target = (seat in `grand_tichu_callers`).
      * Tichu: featurise at each seat's first non-Pass Play (ADR-0018).

    Skill decile and recency weight follow `ParquetBCDataset`'s contract.
    `workers > 1` fans per-game work over a ProcessPoolExecutor.
    """

    def __init__(
        self,
        shards_dir: str | Path,
        *,
        archive_path: str | Path,
        call_type: str,
        expected_featurizer_version: str,
        expected_action_space_version: str,
        ratings_path: str | Path | None = None,
        recency_cutoff_game_id: int = _DEFAULT_RECENCY_CUTOFF,
        recency_weight: float = _DEFAULT_RECENCY_WEIGHT,
        skill_buckets: int = 10,
        workers: int = 1,
        chunk_size: int = 10,
    ) -> None:
        if call_type not in ("grand_tichu", "tichu"):
            raise ValueError(
                f"call_type must be 'grand_tichu' or 'tichu', got {call_type!r}"
            )
        self.shards_dir = Path(shards_dir)
        self.archive_path = Path(archive_path)
        self.call_type = call_type
        self.expected_featurizer_version = expected_featurizer_version
        self.expected_action_space_version = expected_action_space_version
        self.recency_cutoff_game_id = recency_cutoff_game_id
        self.recency_weight = recency_weight
        self.skill_buckets = skill_buckets
        self._neutral_decile = skill_buckets
        # workers <= 1 → serial inline iteration; >1 → ProcessPoolExecutor
        # fan-out over games (featurize + replay are CPU-bound and dominant).
        self.workers = max(1, int(workers))
        # Per-task game count for the parallel path. Larger = less per-submit
        # pickle/IPC overhead, but fewer in-flight tasks for load balancing.
        # K=10 is a reasonable default for ~80ms-per-game workers.
        self.chunk_size = max(1, int(chunk_size))

        self._manifest: dict[str, set[int]] = self._build_manifest()
        if ratings_path:
            self._skill_lookup: dict[str, int] = self._load_skill_lookup(ratings_path)
        else:
            log.info(
                "no ratings_path supplied; all examples will use the "
                "Neutral Skill Decile (=%d)", self._neutral_decile,
            )
            self._skill_lookup = {}

    def _build_manifest(self) -> dict[str, set[int]]:
        # Deferred to avoid circular imports at module load.
        from tichu_training.records import load_shards

        log.info("building call-dataset manifest from play shard under %s …",
                 self.shards_dir)
        table = load_shards(
            self.shards_dir, "play",
            expected_featurizer_version=self.expected_featurizer_version,
            expected_action_space_version=self.expected_action_space_version,
        )
        manifest: dict[str, set[int]] = {}
        game_ids = table.column("game_id").to_pylist()
        round_ids = table.column("round_id").to_pylist()
        for g, r in zip(game_ids, round_ids):
            if g is None or r is None or g == "":
                continue
            manifest.setdefault(g, set()).add(int(r))
        log.info("manifest built: %d games, %d (game_id, round_id) pairs",
                 len(manifest), sum(len(rs) for rs in manifest.values()))
        return manifest

    @staticmethod
    def _load_skill_lookup(ratings_path) -> dict[str, int]:
        log.info("loading skill lookup from %s …", ratings_path)
        table = pq.read_table(
            Path(ratings_path), columns=["player_handle", "skill_decile"],
        )
        handles = table.column("player_handle").to_pylist()
        deciles = table.column("skill_decile").to_pylist()
        lookup = {h: int(d) for h, d in zip(handles, deciles) if h and d is not None}
        log.info("skill lookup: %d handles", len(lookup))
        return lookup

    @property
    def manifest_size(self) -> int:
        return sum(len(rs) for rs in self._manifest.values())

    def _cfg(self) -> dict:
        return dict(
            call_type=self.call_type,
            skill_lookup=self._skill_lookup,
            neutral_decile=self._neutral_decile,
            recency_cutoff=self.recency_cutoff_game_id,
            recency_weight=self.recency_weight,
        )

    def _items(self) -> Iterator[tuple[str, str, tuple[int, ...]]]:
        from tichu_training.bsw.archive import iter_archive

        game_ids = set(self._manifest.keys())
        pbar = tqdm(
            iter_archive(self.archive_path, game_ids=game_ids),
            total=len(game_ids),
            desc=f"streaming archive ({self.call_type})",
            unit="game",
        )
        for stem, text in pbar:
            if stem not in self._manifest:
                continue
            yield stem, text, tuple(sorted(self._manifest[stem]))

    def __iter__(self) -> Iterator[CallExample]:
        cfg = self._cfg()
        if self.workers <= 1:
            for stem, text, valid in self._items():
                yield from _call_examples_for_game(stem, text, set(valid), **cfg)
            return

        # Parallel: sliding-window dispatch keeps `workers * 2` chunks in
        # flight, where each chunk is `chunk_size` games. Amortising the
        # per-submit pickle/IPC cost over K games (K=10 default) is what
        # gets throughput from ~2.5x to ~7x on a 12-core box. skill_lookup
        # is pickled once per worker via the pool initializer.
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        from itertools import islice

        def _chunked(it, n: int):
            while True:
                chunk = list(islice(it, n))
                if not chunk:
                    return
                yield chunk

        chunks = _chunked(self._items(), self.chunk_size)
        max_pending = self.workers * 2
        pending: set = set()
        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_call_worker,
            initargs=(cfg,),
        ) as pool:
            def _top_up(n: int) -> None:
                for chunk in islice(chunks, n):
                    pending.add(pool.submit(_call_worker, chunk))

            _top_up(max_pending)
            while pending:
                done, still = wait(pending, return_when=FIRST_COMPLETED)
                pending = still
                for fut in done:
                    yield from fut.result()
                _top_up(len(done))

    def _sample_weight_for(self, game_id: str) -> float:
        return _sample_weight(
            game_id, self.recency_cutoff_game_id, self.recency_weight,
        )


def _stack(examples: Sequence[CallExample]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(np.stack([e.features for e in examples])),
        "target": torch.tensor([e.target for e in examples], dtype=torch.long),
        "skill_decile": torch.tensor([e.skill_decile for e in examples], dtype=torch.long),
        "sample_weight": torch.tensor([e.sample_weight for e in examples], dtype=torch.float32),
    }


def _batched(it: Iterable[CallExample], batch_size: int) -> Iterator[list[CallExample]]:
    """Yield successive `batch_size`-sized lists from a possibly-streaming iterable.
    The last batch may be short; empty iterables yield nothing."""
    batch: list[CallExample] = []
    for ex in it:
        batch.append(ex)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_one_call_epoch(
    network: CallNetwork,
    examples: Iterable[CallExample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    log_path: Path,
    desc: str | None = None,
) -> float:
    """One pass over `examples` (any `Iterable[CallExample]` — list, generator,
    or `ParquetCallDataset`). One pass = one epoch; multi-epoch is the caller's
    responsibility (re-iterate the source, or wrap in `list(...)` once).

    Shows a tqdm progress bar; the postfix is the rolling mean loss/accuracy
    over the last 100 batches. With a `Sized` container the bar shows ETA; for
    streaming sources the total is unknown so it shows rate + count only.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    step = _next_step(log_path)
    last_loss = float("inf")
    rows: list[dict[str, float]] = []

    # tqdm total: number of batches if examples has __len__, else None.
    try:
        n_examples = len(examples)  # type: ignore[arg-type]
        total_batches: int | None = (n_examples + batch_size - 1) // batch_size
    except TypeError:
        total_batches = None

    pbar = tqdm(
        _batched(examples, batch_size),
        total=total_batches, desc=desc or "train", unit="batch",
    )
    # Rolling window over the last 100 batches for a stable postfix readout.
    recent_loss: deque[float] = deque(maxlen=100)
    recent_acc: deque[float] = deque(maxlen=100)
    n_batches = 0
    for batch in pbar:
        tensors = _stack(batch)
        logits = network(tensors["features"], tensors["skill_decile"])
        per_sample = F.cross_entropy(logits, tensors["target"], reduction="none")
        loss = (per_sample * tensors["sample_weight"]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == tensors["target"]).float().mean().item()
        batch_loss = float(loss.detach())
        rows.append({"step": step, "loss": batch_loss, "accuracy": acc})
        last_loss = batch_loss
        step += 1
        n_batches += 1
        recent_loss.append(batch_loss)
        recent_acc.append(acc)
        if n_batches % 10 == 0:
            pbar.set_postfix(
                loss=f"{sum(recent_loss) / len(recent_loss):.4f}",
                acc=f"{sum(recent_acc) / len(recent_acc):.3f}",
            )
    pbar.close()

    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "loss", "accuracy"])
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return last_loss


def _next_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1


def calling_rate_by_decile(
    network: CallNetwork, examples: Sequence[CallExample]
) -> dict[int, float]:
    """Predicted P(call) > 0.5 → 1, grouped by skill_decile."""
    if not examples:
        return {}
    tensors = _stack(examples)
    with torch.no_grad():
        probs = torch.softmax(network(tensors["features"], tensors["skill_decile"]), dim=-1)
        call_prob = probs[:, 1]
        calls = (call_prob > 0.5).long()
    by_decile: dict[int, list[int]] = defaultdict(list)
    deciles = tensors["skill_decile"].tolist()
    for d, c in zip(deciles, calls.tolist()):
        by_decile[int(d)].append(int(c))
    return {d: sum(v) / len(v) for d, v in by_decile.items()}


def split_examples_by_game(
    examples: Sequence[CallExample],
    *,
    val_frac: float,
    seed: int = 0,
) -> tuple[list[CallExample], list[CallExample]]:
    """Deterministic train/val split that keeps whole games on one side.

    Examples within a single game span all 4 seats and (for tichu) hands
    drawn from the same deal. Splitting at the example level would leak
    that within-game correlation into val. Hashing on `game_id` ensures
    every example from one game lands on the same side.

    `val_frac=0` returns `(all_examples, [])`. Examples with empty
    `game_id` (the synthetic dataset) all hash to the same bucket and
    so all land on one side.
    """
    if val_frac <= 0:
        return list(examples), []
    if val_frac >= 1:
        return [], list(examples)
    threshold = int(round(val_frac * 10_000))
    salt = str(seed).encode()
    train: list[CallExample] = []
    val: list[CallExample] = []
    for ex in examples:
        h = hashlib.blake2b(
            salt + b":" + ex.game_id.encode(), digest_size=4,
        ).digest()
        bucket = int.from_bytes(h, "big") % 10_000
        (val if bucket < threshold else train).append(ex)
    return train, val


def _binary_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney U with average-rank tie handling.

    Returns NaN when one class is absent (AUC undefined). No sklearn
    dependency — pure numpy.
    """
    n = probs.shape[0]
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs, kind="stable")
    sorted_probs = probs[order]
    sorted_labels = labels[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0  # 1-indexed average rank
        i = j + 1
    sum_ranks_pos = float(ranks[sorted_labels == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_call_examples(
    network: CallNetwork,
    examples: Sequence[CallExample],
    *,
    batch_size: int,
) -> dict[str, float]:
    """Unweighted val-set metrics: loss, accuracy, AUC, n, positive fraction.

    Loss is unweighted cross-entropy (recency weights are a training-time
    knob, not a held-out metric). AUC is the right call-quality signal
    given the heavy class imbalance — accuracy is dominated by the
    majority class.
    """
    if not examples:
        return {"n": 0.0, "loss": float("nan"), "accuracy": float("nan"),
                "auc": float("nan"), "pos_frac": float("nan")}
    network.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batched(iter(examples), batch_size):
            tensors = _stack(batch)
            logits = network(tensors["features"], tensors["skill_decile"])
            loss = F.cross_entropy(logits, tensors["target"], reduction="sum")
            pred = logits.argmax(dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            total_loss += float(loss.item())
            total_correct += int((pred == tensors["target"]).sum().item())
            total_n += int(tensors["target"].shape[0])
            all_probs.append(probs.cpu().numpy())
            all_labels.append(tensors["target"].cpu().numpy())
    network.train()
    probs_arr = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels).astype(np.int64)
    return {
        "n": float(total_n),
        "loss": total_loss / total_n,
        "accuracy": total_correct / total_n,
        "auc": _binary_auc(probs_arr, labels_arr),
        "pos_frac": float(labels_arr.mean()),
    }


def write_calling_rate_csv(
    network: CallNetwork, examples: Sequence[CallExample], path: Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rates = calling_rate_by_decile(network, examples)
    by_decile_n: dict[int, int] = defaultdict(int)
    for e in examples:
        by_decile_n[e.skill_decile] += 1
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["decile", "n", "calling_rate"])
        writer.writeheader()
        for decile in sorted(rates.keys()):
            writer.writerow({
                "decile": decile,
                "n": by_decile_n[decile],
                "calling_rate": rates[decile],
            })
