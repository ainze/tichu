"""BC training data sources.

`SyntheticBCDataset` synthesises (features, target, legal_mask) tuples
deterministically from a seed — used by the smoke test. The featurizer is
not invoked; features are drawn from the seeded RNG so the smoke test stays
fast and engine-independent.

`ParquetBCDataset` is the production training source per [ADR-0011]. It
treats the parquet shards as a **manifest** of validated (game_id,
round_id) pairs — Feature Vectors and legal Intent masks are computed at
iteration time by replaying each round from the BSW archive. Skill Decile
is joined live from the TrueSkill ratings table by `player_handle`.
Schupfen is **not** consumed here — it is served by a standalone Schupfen
Network per [ADR-0012].
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq

from tichu_training.bc.decision_types import HEAD_LOGIT_DIMS
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.records import load_shards


log = logging.getLogger("tichu_training.bc.dataset")


_DEFAULT_RECENCY_CUTOFF: int = 1855844
_DEFAULT_RECENCY_WEIGHT: float = 0.5
_NEUTRAL_SKILL_DECILE: int = 10  # the eleventh row of the Skill Embedding

# Map from BSW ParsedAction.kind to BC decision_type. `play` and `pass` both
# feed the `play` head. Schupfen, Tichu/Grand-Tichu calls are excluded from
# BC (see ADR-0007 / ADR-0012).
_KIND_TO_DECISION_TYPE: dict[str, str] = {
    "play": "play",
    "pass": "play",
    "wish": "wish",
    "dragon_give": "dragon_assignment",
}


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
    # Team-relative game-level outcome: True if the row's acting player's
    # team won the Complete Game, False if it lost, None for an Incomplete
    # Session (no game-level outcome to predict). Used by AWR when
    # `value_target='game'`; None rows are filtered out of that fit.
    game_won: bool | None = None


class SyntheticBCDataset(Iterable[BCExample]):
    """Generate `n_per_head` deterministic examples per decision_type."""

    def __init__(
        self,
        *,
        seed: int = 0,
        n_per_head: int = 100,
        feature_dim: int | None = None,
        skill_buckets: int = 10,
        binary_features: bool = False,
    ) -> None:
        self.seed = seed
        self.n_per_head = n_per_head
        # Default to the real featurizer's output dim if not specified.
        if feature_dim is None:
            from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
            feature_dim = FEATURIZER_OUTPUT_DIM
        self.feature_dim = feature_dim
        self.skill_buckets = skill_buckets
        # When True, emit features that honour the real featurizer's
        # value contract: indicator columns are 0/1, only the columns in
        # `CONTINUOUS_FEATURE_COLUMNS` carry arbitrary floats. Required by
        # consumers that bit-pack the indicator columns (the materialised
        # bundle). Default False keeps the historical all-Gaussian features
        # that the AWR / value-baseline tests depend on.
        self.binary_features = binary_features

    def __iter__(self) -> Iterator[BCExample]:
        rng = np.random.default_rng(self.seed)
        cont_idx = None
        if self.binary_features:
            from tichu_training.featurizer import CONTINUOUS_FEATURE_COLUMNS
            cont_idx = np.asarray(
                [c for c in CONTINUOUS_FEATURE_COLUMNS if c < self.feature_dim],
                dtype=np.intp,
            )
        for decision_type, k in HEAD_LOGIT_DIMS.items():
            for _ in range(self.n_per_head):
                if self.binary_features:
                    features = (
                        rng.random(self.feature_dim) < 0.5
                    ).astype(np.float32)
                    if cont_idx.size:
                        features[cont_idx] = rng.standard_normal(
                            cont_idx.size
                        ).astype(np.float32)
                else:
                    features = rng.standard_normal(self.feature_dim).astype(
                        np.float32
                    )
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
                # Deterministic game_won so AWR's `value_target='game'` path
                # has a fittable signal in synthetic tests. 10% None to
                # exercise the Incomplete-Session filter.
                roll = rng.random()
                if roll < 0.1:
                    game_won: bool | None = None
                else:
                    game_won = round_outcome > 0.0
                yield BCExample(
                    decision_type=decision_type,
                    features=features,
                    target=target,
                    legal_mask=legal.astype(bool),
                    sample_weight=1.0,
                    skill_decile=skill,
                    round_outcome=round_outcome,
                    game_won=game_won,
                )


class ParquetBCDataset(Iterable[BCExample]):
    """Archive-driven, replay-on-the-fly BC dataset per ADR-0011.

    Construction:
      - Loads parquet shards (per decision_type) once. Validates the version
        pins (featurizer_version, action_space_version) against the loader's
        expectation; raises `VersionMismatchError` on mismatch.
      - Builds a `(game_id, round_id)` manifest from the parquet — the set
        of replay-validated rounds that contribute BC training rows.
      - Loads the optional ratings parquet for live `player_handle →
        skill_decile` lookup.

    Iteration:
      - Streams the BSW archive in offset order (`tichu_training.bsw.archive
        .iter_archive`), parses each game, replays each in-manifest round
        once. For every parsed decision whose `decision_type` is in
        `HEAD_LOGIT_DIMS`, builds a PrivateState at the decision boundary,
        featurizes, derives the target via `bc_target_for_concrete`, builds
        the legal mask via `legal_mask`, and yields a `BCExample`.

    Excludes (per ADR-0007 / ADR-0012):
      - `call_tichu` / `call_grand_tichu` rows (feed `train_calls`)
      - `schupfen` rows (feed `train_schupfen`)
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
    ) -> None:
        self.shards_dir = Path(shards_dir)
        self.archive_path = Path(archive_path)
        self.expected_featurizer_version = expected_featurizer_version
        self.expected_action_space_version = expected_action_space_version
        self.recency_cutoff_game_id = recency_cutoff_game_id
        self.recency_weight = recency_weight
        self.skill_buckets = skill_buckets
        self._neutral_decile = skill_buckets  # the 11th row

        # Manifest: game_id (str) → set of valid round_index (int).
        # Built from the play shard since per ADR-0008 round-level validity
        # is all-or-nothing across decision types in a round.
        self._manifest: dict[str, set[int]] = self._build_manifest()
        self._skill_lookup: dict[str, int] = (
            self._load_skill_lookup(ratings_path) if ratings_path else {}
        )

    def _build_manifest(self) -> dict[str, set[int]]:
        """Read the play shard's (game_id, round_id) columns; union across
        shards is redundant because ADR-0008 validates at Round granularity.
        Counts and version pins also come from `load_shards` so a
        `VersionMismatchError` fires before any iteration.

        Progress is logged at INFO so a 100k-game manifest build (~30s)
        isn't silent.
        """
        log.info(
            "building manifest from parquet shards under %s …", self.shards_dir,
        )
        manifest: dict[str, set[int]] = {}
        n_rows_by_type: dict[str, int] = {}
        for decision_type in HEAD_LOGIT_DIMS:
            try:
                table = load_shards(
                    self.shards_dir,
                    decision_type,
                    expected_featurizer_version=self.expected_featurizer_version,
                    expected_action_space_version=self.expected_action_space_version,
                )
            except FileNotFoundError:
                log.info("  %s: no shard found, skipping", decision_type)
                continue
            n_rows_by_type[decision_type] = table.num_rows
            log.info(
                "  %s: %d rows; extracting (game_id, round_id) …",
                decision_type, table.num_rows,
            )
            game_ids = table.column("game_id").to_pylist()
            round_ids = table.column("round_id").to_pylist()
            for g, r in zip(game_ids, round_ids):
                if g is None or r is None or g == "":
                    continue
                manifest.setdefault(g, set()).add(int(r))
        log.info(
            "manifest built: %d games, %d (game_id, round_id) pairs",
            len(manifest), sum(len(rs) for rs in manifest.values()),
        )
        self._n_rows_by_type = n_rows_by_type
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
    def n_rows(self) -> int:
        return sum(self._n_rows_by_type.values())

    @property
    def manifest_size(self) -> int:
        return sum(len(rs) for rs in self._manifest.values())

    def __iter__(self) -> Iterator[BCExample]:
        # Deferred imports so the dataset module stays cheap to import.
        from tichu_training.bsw.archive import iter_archive
        from tichu_training.bsw.parser import parse_tch
        from tichu_training.bsw.records import game_team_totals
        from tichu_training.bsw.replay import replay_round
        from tichu_training.action_space import bc_target_for_concrete, legal_mask
        from tichu_training.featurizer import featurize
        from tichu_engine.state import DragonGivePending

        game_ids = set(self._manifest.keys())
        for stem, text in iter_archive(self.archive_path, game_ids=game_ids):
            if stem not in self._manifest:
                continue
            try:
                game = parse_tch(text, game_id=stem)
            except Exception:  # noqa: BLE001
                continue
            sample_weight = self._sample_weight_for(stem)
            # Computed once per game; per-row game_won is team-relative from
            # the row's acting player's seat. None for an Incomplete Session.
            team_totals = game_team_totals(game)
            valid_rounds = self._manifest[stem]
            for parsed_round in game.rounds:
                if parsed_round.round_index not in valid_rounds:
                    continue
                replay = replay_round(parsed_round)
                if replay.final_state is None:
                    continue  # round failed to replay — manifest disagrees, skip
                team_outcome = float(
                    parsed_round.ergebnis[0] - parsed_round.ergebnis[1],
                )
                for (parsed_action, concrete), pre_state, cached_actions in zip(
                    replay.decisions,
                    replay.pre_decision_states,
                    replay.legal_actions_at,
                ):
                    if pre_state is None:
                        continue  # Tichu/Grand-Tichu passthrough or phantom pass
                    decision_type = _KIND_TO_DECISION_TYPE.get(parsed_action.kind)
                    if decision_type is None:
                        continue  # schupfen, tichu, grand_tichu — not BC's job
                    player = parsed_action.player
                    if not 0 <= player < 4:
                        continue
                    # Build PrivateState at the decision boundary and featurize.
                    private = pre_state.private_view(player)
                    features = featurize(private)
                    # Targets are decision-type-specific. For dragon_assignment
                    # we need the winner seat from the pending decision.
                    if decision_type == "dragon_assignment":
                        pending = pre_state.public.pending_decision
                        if not isinstance(pending, DragonGivePending):
                            continue  # shouldn't happen if replay is consistent
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
                    # Reuse the legal-action frozenset that replay_round
                    # already computed (and cached) for validating this
                    # very decision — saves the duplicate enumeration that
                    # would otherwise happen here.
                    mask = legal_mask(
                        decision_type, pre_state, player,
                        cached_actions=cached_actions,
                    )
                    if not mask[target]:
                        # Defensive: target must be in the legal set. If it's
                        # not, the engine/action-space disagree and we'd be
                        # training on a bug. Skip silently — counted in the
                        # `skipped_target_not_in_mask` accumulator if we add
                        # one in a follow-up.
                        continue
                    handle = parsed_round.handles[player]
                    skill = self._skill_lookup.get(handle, self._neutral_decile)
                    team = player % 2
                    if team_totals is None:
                        game_won: bool | None = None
                    else:
                        game_won = team_totals[team] > team_totals[1 - team]
                    yield BCExample(
                        decision_type=decision_type,
                        features=features,
                        target=int(target),
                        legal_mask=mask,
                        sample_weight=sample_weight,
                        skill_decile=skill,
                        round_outcome=team_outcome,
                        game_won=game_won,
                    )

    def _sample_weight_for(self, game_id: str) -> float:
        try:
            gid = int(game_id)
        except (TypeError, ValueError):
            return 1.0
        return 1.0 if gid >= self.recency_cutoff_game_id else self.recency_weight
