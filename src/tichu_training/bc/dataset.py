"""BC training data sources.

`SyntheticBCDataset` synthesises (features, target, legal_mask) tuples
deterministically from a seed — used by the smoke test. The featurizer is
not invoked; features are drawn from the seeded RNG so the smoke test stays
fast and engine-independent.

`ParquetBCDataset` loads parquet shards produced by #007 and validates the
two version columns against the loader's expectation, raising
`VersionMismatchError` on drift. Yielding featurized examples from parquet
requires re-replaying the round from the .tch source; that path is a stub
in v1 and a follow-up will land the full reconstruction pipeline once
featurization-caching is decided.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from tichu_training.bc.heads import HEAD_LOGIT_DIMS
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.records import load_shards


@dataclass
class BCExample:
    decision_type: str
    features: np.ndarray
    target: int
    legal_mask: np.ndarray
    sample_weight: float
    skill_decile: int  # index in [0, num_buckets]; num_buckets is the neutral row
    # Team-0 minus team-1 Ergebnis for the round the decision was made in.
    # Used as the reward signal for AWR offline refinement (#012). Defaults
    # to 0.0 — BC training ignores it.
    round_outcome: float = 0.0


class SyntheticBCDataset(Iterable[BCExample]):
    """Generate `n_per_head` deterministic examples per decision_type."""

    def __init__(
        self,
        *,
        seed: int = 0,
        n_per_head: int = 100,
        feature_dim: int | None = None,
        skill_buckets: int = 10,
    ) -> None:
        self.seed = seed
        self.n_per_head = n_per_head
        # Default to the real featurizer's output dim if not specified.
        if feature_dim is None:
            from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
            feature_dim = FEATURIZER_OUTPUT_DIM
        self.feature_dim = feature_dim
        self.skill_buckets = skill_buckets

    def __iter__(self) -> Iterator[BCExample]:
        rng = np.random.default_rng(self.seed)
        for decision_type, k in HEAD_LOGIT_DIMS.items():
            for _ in range(self.n_per_head):
                features = rng.standard_normal(self.feature_dim).astype(np.float32)
                # 50% of actions legal (minimum 1).
                legal = rng.random(k) < 0.5
                if not legal.any():
                    legal[rng.integers(0, k)] = True
                legal_idx = np.flatnonzero(legal)
                target = int(rng.choice(legal_idx))
                # Pick a skill bucket (or neutral with 10% probability).
                if rng.random() < 0.1:
                    skill = self.skill_buckets  # neutral / cold-start
                else:
                    skill = int(rng.integers(0, self.skill_buckets))
                # Deterministic non-zero outcome with feature dependence so the
                # AWR value baseline has signal to fit. Magnitude (~ ±50) is
                # plausible for Tichu Ergebnis values.
                round_outcome = float(features[: min(4, features.size)].sum() * 10.0)
                yield BCExample(
                    decision_type=decision_type,
                    features=features,
                    target=target,
                    legal_mask=legal.astype(bool),
                    sample_weight=1.0,
                    skill_decile=skill,
                    round_outcome=round_outcome,
                )


class ParquetBCDataset:
    """Read parquet shards and validate version pins.

    Yielding featurized examples requires re-replaying rounds from the
    `.tch` source — that path is deferred to a follow-up so the BC training
    pipeline can ship without coupling the trainer to the parser.
    """

    def __init__(
        self,
        shards_dir: str | Path,
        *,
        expected_featurizer_version: str,
        expected_action_space_version: str,
    ) -> None:
        self.shards_dir = Path(shards_dir)
        self.expected_featurizer_version = expected_featurizer_version
        self.expected_action_space_version = expected_action_space_version
        # Validate at construction: any mismatch fails fast.
        self._n_rows_by_type: dict[str, int] = {}
        for decision_type in HEAD_LOGIT_DIMS:
            try:
                table = load_shards(
                    self.shards_dir,
                    decision_type,
                    expected_featurizer_version=expected_featurizer_version,
                    expected_action_space_version=expected_action_space_version,
                )
            except FileNotFoundError:
                continue
            except VersionMismatchError:
                raise
            self._n_rows_by_type[decision_type] = table.num_rows

    @property
    def n_rows(self) -> int:
        return sum(self._n_rows_by_type.values())
