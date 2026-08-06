"""ParquetBCDataset: archive-driven, replay-on-the-fly streaming per ADR-0011.

Builds a tiny end-to-end fixture from the sample/ .tch files:
  1. Compress them into a `.zst + .idx` archive (matching tools/compress.py).
  2. Run `parse_bsw` against that archive to produce parquet shards.
  3. Construct ParquetBCDataset against the archive + shards.
  4. Iterate and verify the yielded BCExamples are well-formed.
"""

import base64
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard as zstd

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.dataset import BCExample, ParquetBCDataset
from tichu_training.bc.heads import HEAD_LOGIT_DIMS
from tichu_training.cli.parse_bsw import main as parse_bsw_main
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def _build_indexed_archive(samples_dir: Path, archive_path: Path) -> None:
    """Build the same indexed-blob archive format `tools/compress.py` writes,
    so iter_archive can read it. Mirrors the fixture in
    tests/training/bsw/test_archive.py."""
    files = sorted(samples_dir.glob("*.tch"), key=lambda p: p.name)
    payloads = [p.read_bytes() for p in files]
    dict_data = zstd.train_dictionary(8192, payloads * 50)
    cctx = zstd.ZstdCompressor(dict_data=dict_data, level=3)
    index: dict[str, list[int]] = {}
    offset = 0
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as out:
        for path, payload in zip(files, payloads):
            blob = cctx.compress(payload)
            out.write(blob)
            index[path.name] = [offset, len(blob)]
            offset += len(blob)
    sidecar = {
        "version": 1, "level": 3,
        "dict": base64.b64encode(dict_data.as_bytes()).decode("ascii"),
        "files": index,
    }
    archive_path.with_suffix(archive_path.suffix + ".idx").write_text(json.dumps(sidecar))


def _build_ratings_parquet(path: Path, handle_to_decile: dict[str, int]) -> None:
    schema = pa.schema([
        ("player_handle", pa.string()),
        ("skill_decile", pa.int32()),
    ])
    table = pa.Table.from_pydict(
        {
            "player_handle": list(handle_to_decile.keys()),
            "skill_decile": list(handle_to_decile.values()),
        },
        schema=schema,
    )
    pq.write_table(table, path)


@pytest.fixture
def smoke_setup(tmp_path: Path):
    """End-to-end fixture: archive + parquet shards from the sample data."""
    archive = tmp_path / "sample.zst"
    _build_indexed_archive(_SAMPLES, archive)

    shards_dir = tmp_path / "shards"
    rc = parse_bsw_main(["--archive", str(archive), "--output", str(shards_dir)])
    assert rc == 0
    return {"archive": archive, "shards_dir": shards_dir, "tmp_path": tmp_path}


def test_constructor_builds_manifest_from_validated_rounds(smoke_setup):
    ds = ParquetBCDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    # At least one round survived replay validation in the sample data.
    assert ds.manifest_size > 0


def test_iteration_yields_well_formed_bc_examples(smoke_setup):
    ds = ParquetBCDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    examples = list(ds)
    assert len(examples) > 0, "expected at least one BCExample from sample data"
    for ex in examples:
        assert isinstance(ex, BCExample)
        # Decision type is one of the THREE BC heads (not schupfen).
        assert ex.decision_type in HEAD_LOGIT_DIMS, ex.decision_type
        # Features are featurize() output dim.
        assert ex.features.shape == (FEATURIZER_OUTPUT_DIM,)
        # Target is in [0, K) for that head.
        k = HEAD_LOGIT_DIMS[ex.decision_type]
        assert 0 <= ex.target < k, (ex.decision_type, ex.target, k)
        # Mask is bool of shape (K,) and the target slot is True.
        assert ex.legal_mask.dtype == bool
        assert ex.legal_mask.shape == (k,)
        assert ex.legal_mask[ex.target], "target must be in the legal set"


def test_no_schupfen_examples_emitted(smoke_setup):
    """ADR-0012: schupfen is a standalone Schupfen Network, not a BC head.
    The dataset must filter schupfen rows out entirely."""
    ds = ParquetBCDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    decision_types = {ex.decision_type for ex in ds}
    assert "schupfen" not in decision_types


def test_skill_decile_joined_from_ratings_parquet(smoke_setup):
    """If a ratings parquet is supplied, examples carry the joined decile.
    Otherwise (or for handles missing from the table) they get Neutral."""
    # Synthesise a ratings table that puts every sample-data handle into
    # decile 7 except one anonymous-or-missing handle.
    archive = smoke_setup["archive"]
    shards_dir = smoke_setup["shards_dir"]
    # Find one handle that appears in the parquet manifest.
    play_table = pq.read_table(shards_dir / "play_00000.parquet", columns=["player_handle"])
    handles = {h for h in play_table.column("player_handle").to_pylist() if h}
    target_handle = next(iter(handles))
    ratings = smoke_setup["tmp_path"] / "ratings.parquet"
    _build_ratings_parquet(ratings, {target_handle: 7})

    ds = ParquetBCDataset(
        shards_dir,
        archive_path=archive,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        ratings_path=ratings,
    )
    examples = list(ds)
    # At least some example was emitted with the joined decile.
    decile_7_examples = [ex for ex in examples if ex.skill_decile == 7]
    assert len(decile_7_examples) > 0, "expected at least one example with decile 7 joined"
    # Other handles map to the Neutral row (=skill_buckets).
    other_examples = [ex for ex in examples if ex.skill_decile != 7]
    assert all(ex.skill_decile == 10 for ex in other_examples), (
        "handles missing from ratings should fall back to Neutral Skill Decile (=10)"
    )


def test_sample_weight_recency_cutoff(smoke_setup):
    """Pre-cutoff game_ids get the low recency weight; post-cutoff get 1.0.
    Both sample games are pre-2015 (game_id 2417500/2417501 vs cutoff 1855844)
    — they're post-cutoff, so sample_weight should be 1.0."""
    ds = ParquetBCDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    examples = list(ds)
    assert all(ex.sample_weight == 1.0 for ex in examples)


def test_n_rows_reports_parquet_shard_totals(smoke_setup):
    ds = ParquetBCDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    # n_rows is the total across BC-consumed shards (play, wish, dragon).
    # It is independent of how many BCExamples actually get yielded — the
    # latter depends on replay-time filtering for None pre_decision_states.
    assert ds.n_rows >= len(list(ds))


def test_forced_play_rows_are_dropped_from_emission_only(smoke_setup):
    """v7 (ADR-0044): drop Forced Play Decisions from the BC corpus.

    A Play Decision with exactly ONE legal Intent teaches nothing: the masked
    softmax puts probability 1 on it, so `nll == 0` and the gradient is exactly
    0. It also inflates reported top-1 and costs ~39% of the bundle's disk.

    INTENT level, not ConcreteAction level — `rollout._forced_action` tests
    `len(legal_actions(state)) == 1`, a strictly smaller set (one "Single 7"
    Intent can have two suited realisations). The BC gradient flows through the
    1809-way Intent softmax, so that is the predicate that matters here.

    EMISSION only: replay still steps every Decision, so wish / dragon rows are
    untouched and nothing downstream of a forced Play shifts.
    """
    def _rows(**kw):
        return list(ParquetBCDataset(
            smoke_setup["shards_dir"],
            archive_path=smoke_setup["archive"],
            expected_featurizer_version=FEATURIZER_VERSION,
            expected_action_space_version=ACTION_SPACE_VERSION,
            **kw,
        ))

    keep = _rows()
    drop = _rows(skip_forced_play=True)

    forced = [e for e in keep
              if e.decision_type == "play" and int(e.legal_mask.sum()) == 1]
    assert forced, "sample data must contain some Forced Play Decisions"

    # Every surviving play row has a real choice ...
    assert all(int(e.legal_mask.sum()) > 1
               for e in drop if e.decision_type == "play")
    # ... exactly the forced ones went ...
    assert (len([e for e in keep if e.decision_type == "play"])
            - len([e for e in drop if e.decision_type == "play"])) == len(forced)
    # ... and no other head lost a single row.
    for head in ("wish", "dragon_assignment"):
        assert (len([e for e in keep if e.decision_type == head])
                == len([e for e in drop if e.decision_type == head])), head
