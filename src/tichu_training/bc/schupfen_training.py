"""Schupfen Network datasets + training loop.

Mirrors `bc/call_training.py` (which trains the Tichu / Grand-Tichu Call
Networks) but for the structured 3-tuple schupfen target.

Two dataset sources:

* `SyntheticSchupfenDataset` — random features + random hand mask of 14
  slots + random distinct 3-tuple target drawn from the hand mask.
  Used by the smoke config; no parquet, no archive.

* `ParquetSchupfenDataset` — production source. Uses the play shard as
  the validated-rounds manifest (same as `ParquetCallDataset`), streams
  the BSW archive, and per validated round emits 4 examples — one per
  seat — featurised at a synthetic pre-schupfen `GameState` built from
  `parsed_round.start_hands`. `grand_tichu_callers` is populated from
  the parsed round (the calls are public by the time schupfen happens);
  `tichu_callers` is left empty (Tichu featurises at first-non-pass-play
  per ADR-0018, so at schupfen time no Tichu calls have been recorded
  in the pipeline's timeline). Targets are the seat's three schupfen
  cards mapped through `card_slots.card_slot`.
"""

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.card_slots import CARD_SLOTS


@dataclass
class SchupfenExample:
    features: np.ndarray            # shape (FEATURIZER_OUTPUT_DIM,)
    hand_mask: np.ndarray           # shape (56,), 1.0 for cards held, 0.0 otherwise
    target: np.ndarray              # shape (3,) int — (slot_to_next, slot_to_partner, slot_to_previous)
    skill_decile: int
    sample_weight: float = 1.0
    # Provenance (ADR-0020): the source (game_id, round_id). Populated by the
    # consolidated parse pass; 0 for synthetic / standalone examples that
    # don't carry it. Materialised into the bundle meta for per-game filtering.
    game_id: int = 0
    round_id: int = 0


class SyntheticSchupfenDataset(Iterable[SchupfenExample]):
    """Deterministic schupfen examples for the smoke path.

    Each example: random features, a random 14-of-56 hand mask, and a
    random distinct 3-tuple target drawn from the hand mask. The target
    being inside the hand mask is what makes the masked cross-entropy
    loss learnable on synthetic data.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        n_examples: int = 200,
        feature_dim: int | None = None,
        skill_buckets: int = 10,
        binary_features: bool = False,
    ) -> None:
        if feature_dim is None:
            from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
            feature_dim = FEATURIZER_OUTPUT_DIM
        self.seed = seed
        self.n_examples = n_examples
        self.feature_dim = feature_dim
        self.skill_buckets = skill_buckets
        # When True, indicator columns are 0/1 and only the columns in
        # `CONTINUOUS_FEATURE_COLUMNS` carry arbitrary floats — the value
        # contract the materialised bundle's bit-packer requires (it guards
        # against non-0/1 indicator columns). Default False keeps the
        # historical all-Gaussian features. Mirrors `SyntheticBCDataset`.
        self.binary_features = binary_features

    def __iter__(self) -> Iterator[SchupfenExample]:
        rng = np.random.default_rng(self.seed)
        cont_idx = None
        if self.binary_features:
            from tichu_training.featurizer import CONTINUOUS_FEATURE_COLUMNS
            cont_idx = np.asarray(
                [c for c in CONTINUOUS_FEATURE_COLUMNS if c < self.feature_dim],
                dtype=np.intp,
            )
        for _ in range(self.n_examples):
            if self.binary_features:
                features = (rng.random(self.feature_dim) < 0.5).astype(np.float32)
                if cont_idx.size:
                    features[cont_idx] = rng.standard_normal(
                        cont_idx.size
                    ).astype(np.float32)
            else:
                features = rng.standard_normal(self.feature_dim).astype(np.float32)
            hand_slots = rng.choice(56, size=14, replace=False)
            hand_mask = np.zeros(56, dtype=np.float32)
            hand_mask[hand_slots] = 1.0
            target = rng.choice(hand_slots, size=3, replace=False).astype(np.int64)
            if rng.random() < 0.1:
                skill = self.skill_buckets   # Neutral Skill Decile
            else:
                skill = int(rng.integers(0, self.skill_buckets))
            yield SchupfenExample(
                features=features,
                hand_mask=hand_mask,
                target=target,
                skill_decile=skill,
            )


def _stack(examples: Sequence[SchupfenExample]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(np.stack([e.features for e in examples])),
        "hand_mask": torch.from_numpy(np.stack([e.hand_mask for e in examples])),
        "target": torch.from_numpy(np.stack([e.target for e in examples])).long(),
        "skill_decile": torch.tensor([e.skill_decile for e in examples], dtype=torch.long),
        "sample_weight": torch.tensor([e.sample_weight for e in examples], dtype=torch.float32),
    }


def _batched(it: Iterable[SchupfenExample], batch_size: int) -> Iterator[list[SchupfenExample]]:
    batch: list[SchupfenExample] = []
    for ex in it:
        batch.append(ex)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


_NEUTRAL_SKILL_DECILE: int = 10
_DEFAULT_RECENCY_CUTOFF: int = 1855844
_DEFAULT_RECENCY_WEIGHT: float = 0.5


def _sample_weight_for(game_id: str, recency_cutoff: int, recency_weight: float) -> float:
    try:
        gid = int(game_id)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if gid >= recency_cutoff else recency_weight


class ParquetSchupfenDataset(Iterable[SchupfenExample]):
    """Production source — same archive + manifest pattern as
    `ParquetCallDataset`, simpler because schupfen needs no replay.

    Manifest is built from the play shard (validated rounds, per
    ADR-0008 round-granular validation). For each validated round, four
    `SchupfenExample`s are emitted — one per seat — via
    `_schupfen_examples_for_round`. No legal-action enumeration, no
    early-stop predicate: schupfen is fully determined by `start_hands`.
    """

    def __init__(
        self,
        shards_dir,
        *,
        archive_path,
        expected_featurizer_version: str,
        expected_action_space_version: str,
        ratings_path=None,
        recency_cutoff_game_id: int = _DEFAULT_RECENCY_CUTOFF,
        recency_weight: float = _DEFAULT_RECENCY_WEIGHT,
        skill_buckets: int = 10,
        workers: int = 1,
    ) -> None:
        from pathlib import Path as _Path
        self.shards_dir = _Path(shards_dir)
        self.archive_path = _Path(archive_path)
        self.expected_featurizer_version = expected_featurizer_version
        self.expected_action_space_version = expected_action_space_version
        self.recency_cutoff_game_id = recency_cutoff_game_id
        self.recency_weight = recency_weight
        self.skill_buckets = skill_buckets
        self._neutral_decile = skill_buckets
        self.workers = max(1, int(workers))
        self._manifest: dict[str, set[int]] = self._build_manifest()
        if ratings_path:
            self._skill_lookup = self._load_skill_lookup(ratings_path)
        else:
            self._skill_lookup = {}

    def _build_manifest(self) -> dict[str, set[int]]:
        from tichu_training.records import load_shards
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
        return manifest

    @staticmethod
    def _load_skill_lookup(ratings_path) -> dict[str, int]:
        import pyarrow.parquet as pq
        from pathlib import Path as _Path
        table = pq.read_table(
            _Path(ratings_path), columns=["player_handle", "skill_decile"],
        )
        handles = table.column("player_handle").to_pylist()
        deciles = table.column("skill_decile").to_pylist()
        return {h: int(d) for h, d in zip(handles, deciles) if h and d is not None}

    def __iter__(self) -> Iterator[SchupfenExample]:
        from tichu_training.bsw.archive import iter_archive
        from tichu_training.bsw.parser import parse_tch

        game_ids = set(self._manifest.keys())
        pbar = tqdm(
            iter_archive(self.archive_path, game_ids=game_ids),
            total=len(game_ids), desc="streaming archive (schupfen)", unit="game",
        )
        for stem, text in pbar:
            if stem not in self._manifest:
                continue
            try:
                game = parse_tch(text, game_id=stem)
            except Exception:  # noqa: BLE001
                continue
            sw = _sample_weight_for(stem, self.recency_cutoff_game_id, self.recency_weight)
            valid_rounds = self._manifest[stem]
            for parsed_round in game.rounds:
                if parsed_round.round_index not in valid_rounds:
                    continue
                yield from _schupfen_examples_for_round(
                    parsed_round,
                    skill_lookup=self._skill_lookup,
                    neutral_decile=self._neutral_decile,
                    sample_weight=sw,
                )


def _schupfen_examples_for_round(
    parsed_round,
    *,
    skill_lookup: dict[str, int],
    neutral_decile: int,
    sample_weight: float,
) -> list[SchupfenExample]:
    """Build one SchupfenExample per seat for a single validated round.

    Featurise moment: synthetic pre-schupfen `GameState` from
    `parsed_round.start_hands`. `grand_tichu_callers` is populated
    (calls are public by schupfen time — round order is Grand-Tichu →
    Schupfen → Tichu → Tricks). `tichu_callers` is left empty (the
    pipeline times Tichu calls at first-non-pass-play per ADR-0018,
    so no Tichu calls are recorded in this featurise moment).
    """
    import dataclasses

    from tichu_engine.state import GameState, PublicState, SchupfenPending, Trick
    from tichu_training.card_slots import card_slot
    from tichu_training.featurizer import featurize

    out: list[SchupfenExample] = []
    public = PublicState(
        current_player=0,  # overridden per acting seat below; see loop comment.
        hand_sizes=(14, 14, 14, 14),
        scores=(0, 0),
        trick=Trick.empty(),
        pending_decision=SchupfenPending(submitted=(None, None, None, None)),
        grand_tichu_callers=parsed_round.grand_tichu_callers,
        # tichu_callers: left at default frozenset() — empty at schupfen time
    )
    state = GameState(hands=parsed_round.start_hands, public=public)
    schupfen_by_seat = {a.player: a for a in parsed_round.schupfen}
    for seat in range(4):
        action = schupfen_by_seat.get(seat)
        if action is None:
            continue   # malformed round; skip
        # current_player MUST equal the acting seat to match inference. The
        # featurizer one-hots current_player AND grand_tichu_callers at ABSOLUTE
        # slots, so the head can only localise the caller relative to itself via
        # (grand_caller - current_player) % 4 — e.g. to learn "the caller is my
        # partner". The engine advances current_player through the seats during
        # schupfen (engine._apply_schupfen), so the served state at seat S's turn
        # has current_player == S. Pinning it to 0 here made that relationship
        # unlearnable for seats 1..3 (3/4 of examples): the head could not tell
        # "partner called" from "opponent called" and collapsed to "any grand
        # call -> give Dog to partner". See tests/.../test_schupfen_training.py.
        seat_public = dataclasses.replace(public, current_player=seat)
        seat_state = dataclasses.replace(state, public=seat_public)
        private = seat_state.private_view(seat)
        features = featurize(private)
        hand_mask = np.zeros(CARD_SLOTS, dtype=np.float32)
        for c in parsed_round.start_hands[seat]:
            hand_mask[card_slot(c)] = 1.0
        target = np.array(
            [card_slot(action.schupfen_to_next),
             card_slot(action.schupfen_to_partner),
             card_slot(action.schupfen_to_previous)],
            dtype=np.int64,
        )
        handle = parsed_round.handles[seat]
        skill = skill_lookup.get(handle, neutral_decile)
        out.append(SchupfenExample(
            features=features,
            hand_mask=hand_mask,
            target=target,
            skill_decile=skill,
            sample_weight=sample_weight,
        ))
    return out


def _next_step(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1


def _masked_logits(logits: torch.Tensor, hand_mask: torch.Tensor) -> torch.Tensor:
    """Apply the hand mask by setting off-hand slots to -inf — the
    softmax inside cross_entropy then puts zero mass off-hand."""
    neg_inf = torch.finfo(logits.dtype).min
    return logits.masked_fill(hand_mask == 0, neg_inf)


def train_one_schupfen_epoch(
    network: SchupfenNetwork,
    examples: Iterable[SchupfenExample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    log_path: Path,
    desc: str | None = None,
) -> float:
    """One pass over `examples`. Loss = sum of three hand-masked
    cross-entropies (one per direction) × `sample_weight`, batch-mean.
    Heads are independent at train time; distinctness across heads is
    resolved at decode time, not in the loss."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    step = _next_step(log_path)
    last_loss = float("inf")
    rows: list[dict[str, float]] = []

    try:
        n_examples = len(examples)  # type: ignore[arg-type]
        total_batches: int | None = (n_examples + batch_size - 1) // batch_size
    except TypeError:
        total_batches = None

    device = next(network.parameters()).device
    pbar = tqdm(
        _batched(examples, batch_size),
        total=total_batches, desc=desc or "train", unit="batch",
    )
    recent_loss: deque[float] = deque(maxlen=100)
    recent_acc: deque[float] = deque(maxlen=100)
    n_batches = 0
    for batch in pbar:
        tensors = {k: v.to(device) for k, v in _stack(batch).items()}
        head_logits = network(tensors["features"], tensors["skill_decile"])
        per_sample = torch.zeros(len(batch), dtype=head_logits[0].dtype, device=device)
        per_dir_correct = 0
        for d, logits_d in enumerate(head_logits):
            masked = _masked_logits(logits_d, tensors["hand_mask"])
            per_sample = per_sample + F.cross_entropy(
                masked, tensors["target"][:, d], reduction="none",
            )
            with torch.no_grad():
                pred = masked.argmax(dim=-1)
                per_dir_correct += (pred == tensors["target"][:, d]).float().sum().item()
        loss = (per_sample * tensors["sample_weight"]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = per_dir_correct / (3 * len(batch))   # mean per-direction top-1
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


_VAL_BUCKETS = 10_000


def _schupfen_val_bucket(game_id: np.ndarray, seed: int) -> np.ndarray:
    """Vectorised game-level splitmix64 hash → bucket in [0, _VAL_BUCKETS).

    Same family as `call_training._val_bucket`: a game's rows never straddle
    the train/val boundary, so a seat's same-deal siblings can't leak. game_id
    0 (synthetic / empty sentinel) all lands in one bucket."""
    with np.errstate(over="ignore"):
        z = game_id.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return (z % np.uint64(_VAL_BUCKETS)).astype(np.int64)


def train_one_schupfen_epoch_batched(
    network: SchupfenNetwork,
    batches: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    *,
    log_path: Path,
    desc: str | None = None,
    total_batches: int | None = None,
    val_frac: float = 0.0,
    val_seed: int = 0,
) -> tuple[float, dict[str, float] | None]:
    """Streaming epoch over pre-stacked numpy batch dicts (the memmap fast
    path). Three hand-masked cross-entropies × sample_weight, batch-mean, one
    step.csv row per batch — never materialises the slice.

    Returns `(last_loss, val_metrics)`. When `val_frac > 0` a game-level
    holdout (game_id hash fixed by `val_seed`, identical every epoch) is carved
    per batch; held-out rows are forwarded in the SAME sweep (no grad) to
    accumulate an ONLINE val estimate — `{"n", "loss" (unweighted 3-CE NLL),
    "accuracy" (per-direction top-1)}`. `val_metrics` is None when val_frac<=0.
    Early-epoch val rows see a less-trained net, so it is a monitoring signal
    (mirrors `train_one_call_epoch_batched`)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    step = _next_step(log_path)
    last_loss = float("inf")
    rows: list[dict[str, float]] = []
    threshold = int(round(val_frac * _VAL_BUCKETS)) if val_frac > 0 else 0

    device = next(network.parameters()).device
    pbar = tqdm(batches, total=total_batches, desc=desc or "train", unit="batch")
    recent_loss: deque[float] = deque(maxlen=100)
    recent_acc: deque[float] = deque(maxlen=100)
    n_batches = 0
    v_nll = 0.0
    v_correct = 0
    v_n = 0
    for batch in pbar:
        if threshold > 0 and "game_id" in batch:
            buckets = _schupfen_val_bucket(batch["game_id"], val_seed)
            train_np = buckets >= threshold
            val_np = ~train_np
        else:
            train_np = np.ones(batch["target"].shape[0], dtype=bool)
            val_np = np.zeros_like(train_np)

        # ---- train step (train-side rows) ----
        if train_np.any():
            feats = torch.from_numpy(np.ascontiguousarray(batch["features"][train_np])).to(device)
            hand_mask = torch.from_numpy(np.ascontiguousarray(batch["hand_mask"][train_np])).to(device)
            target = torch.from_numpy(np.ascontiguousarray(batch["target"][train_np])).long().to(device)
            skill = torch.from_numpy(np.ascontiguousarray(batch["skill_decile"][train_np])).long().to(device)
            sw = torch.from_numpy(np.ascontiguousarray(batch["sample_weight"][train_np])).float().to(device)
            head_logits = network(feats, skill)
            bsz = target.shape[0]
            per_sample = torch.zeros(bsz, dtype=head_logits[0].dtype, device=device)
            per_dir_correct = 0
            for d, logits_d in enumerate(head_logits):
                masked = _masked_logits(logits_d, hand_mask)
                per_sample = per_sample + F.cross_entropy(
                    masked, target[:, d], reduction="none",
                )
                with torch.no_grad():
                    per_dir_correct += (masked.argmax(dim=-1) == target[:, d]).float().sum().item()
            loss = (per_sample * sw).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = per_dir_correct / (3 * bsz)
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

        # ---- online val (held-out rows, no grad, same sweep) ----
        if val_np.any():
            with torch.no_grad():
                vfeats = torch.from_numpy(np.ascontiguousarray(batch["features"][val_np])).to(device)
                vmask = torch.from_numpy(np.ascontiguousarray(batch["hand_mask"][val_np])).to(device)
                vtarget = torch.from_numpy(np.ascontiguousarray(batch["target"][val_np])).long().to(device)
                vskill = torch.from_numpy(np.ascontiguousarray(batch["skill_decile"][val_np])).long().to(device)
                vlogits = network(vfeats, vskill)
                vbsz = vtarget.shape[0]
                vper = torch.zeros(vbsz, dtype=vlogits[0].dtype, device=device)
                for d, logits_d in enumerate(vlogits):
                    masked = _masked_logits(logits_d, vmask)
                    vper = vper + F.cross_entropy(masked, vtarget[:, d], reduction="none")
                    v_correct += int((masked.argmax(dim=-1) == vtarget[:, d]).sum().item())
                v_nll += float(vper.sum().item())
                v_n += vbsz
    pbar.close()

    with log_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "loss", "accuracy"])
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if val_frac <= 0 or v_n == 0:
        return last_loss, None
    return last_loss, {
        "n": float(v_n),
        "loss": v_nll / v_n,
        "accuracy": v_correct / (3 * v_n),
    }
