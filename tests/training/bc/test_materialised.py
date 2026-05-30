"""Roundtrip + version-pin tests for the materialised BC bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tichu_training.bc.dataset import BCExample, SyntheticBCDataset
from tichu_training.bc.heads import HEAD_LOGIT_DIMS
from tichu_training.bc.materialised import (
    MATERIALISED_SCHEMA_VERSION,
    MemmapBCDataset,
    TYPE_ORDER,
    materialise,
)
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION


@pytest.fixture
def materialised_smoke(tmp_path: Path) -> tuple[list[BCExample], Path]:
    """Materialise a SyntheticBCDataset slice to disk and return (source, out_dir)."""
    src = SyntheticBCDataset(seed=42, n_per_head=20, skill_buckets=10, binary_features=True)
    examples = list(src)
    out_dir = tmp_path / "bundle"
    materialise(iter(examples), out_dir, max_examples=len(examples))
    return examples, out_dir


def test_materialise_emits_manifest_and_per_type_files(materialised_smoke):
    _, out_dir = materialised_smoke
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == MATERIALISED_SCHEMA_VERSION
    assert manifest["featurizer_version"] == FEATURIZER_VERSION
    assert manifest["feature_dim"] == FEATURIZER_OUTPUT_DIM
    assert manifest["type_order"] == TYPE_ORDER
    # All three heads should have written files in the synthetic case.
    for head in HEAD_LOGIT_DIMS:
        assert (out_dir / f"{head}_feat_bits.dat").exists()
        assert (out_dir / f"{head}_feat_cont.dat").exists()
        assert (out_dir / f"{head}_legal_mask.dat").exists()
        assert (out_dir / f"{head}_meta.dat").exists()
    assert (out_dir / "order.dat").exists()


def test_features_are_bit_packed_on_disk(materialised_smoke):
    """Feature files must store indicator columns bit-packed (feat_bits) and
    only the continuous columns as f32 (feat_cont) — schema v2. A raw f32
    play row is 224*4 = 896 B; packed it is ceil(214/8) + 10*4 = 27 + 40 =
    67 B (~13x). Locks the format against a silent revert to dense f32.
    """
    _, out_dir = materialised_smoke
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["features_packed"] is True
    cont_cols = manifest["continuous_feature_columns"]
    D = manifest["feature_dim"]
    n_cont = len(cont_cols)
    n_bin = D - n_cont
    bits_bytes = (n_bin + 7) // 8
    counts = manifest["counts"]
    for head in HEAD_LOGIT_DIMS:
        if counts.get(head, 0) == 0:
            continue
        rows = counts[head]
        assert (out_dir / f"{head}_feat_bits.dat").stat().st_size == (
            rows * bits_bytes
        )
        assert (out_dir / f"{head}_feat_cont.dat").stat().st_size == (
            rows * n_cont * 4
        )


def test_materialise_rejects_non_binary_indicator_columns(tmp_path: Path):
    """The writer bit-packs the indicator feature columns, which only works
    if they are 0/1. A stream whose indicator columns carry other values
    (e.g. SyntheticBCDataset's default all-Gaussian features) must raise,
    not silently corrupt — guards against a featurizer layout drift.
    """
    src = list(SyntheticBCDataset(seed=0, n_per_head=4, skill_buckets=10))
    with pytest.raises(ValueError, match="non-0/1"):
        materialise(iter(src), tmp_path / "bad")


def test_legal_mask_is_bit_packed_on_disk(materialised_smoke):
    """The mask files must be packbits-compressed (schema v2): one byte per
    8 actions, not one byte per action. This is the disk win — a raw play
    mask would be 1809 B/row; packed it is 227 B/row (~8x smaller). Locks
    the format so a regression can't silently revert to raw uint8 masks.
    """
    _, out_dir = materialised_smoke
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["legal_mask_packed"] is True
    counts = manifest["counts"]
    for head, mask_dim in HEAD_LOGIT_DIMS.items():
        if counts.get(head, 0) == 0:
            continue
        packed_bytes = (mask_dim + 7) // 8
        expected = counts[head] * packed_bytes
        actual = (out_dir / f"{head}_legal_mask.dat").stat().st_size
        assert actual == expected, (
            f"{head} mask not packed: {actual} B != {expected} B "
            f"({counts[head]} rows x {packed_bytes} B); raw would be "
            f"{counts[head] * mask_dim} B"
        )


def test_roundtrip_iter_yields_same_examples_in_order(materialised_smoke):
    """Source order via SyntheticBCDataset must match the reader's __iter__
    after a materialise pass. Preserves the contract that the bundle is a
    drop-in for Iterable[BCExample]."""
    src_examples, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    read_examples = list(ds)
    assert len(read_examples) == len(src_examples)
    for src, got in zip(src_examples, read_examples):
        assert got.decision_type == src.decision_type
        assert got.target == src.target
        np.testing.assert_array_equal(got.features, src.features)
        np.testing.assert_array_equal(got.legal_mask, src.legal_mask)
        assert got.sample_weight == pytest.approx(src.sample_weight)
        assert got.skill_decile == src.skill_decile
        assert got.round_outcome == pytest.approx(src.round_outcome, abs=1e-4)
        assert got.game_won == src.game_won


def test_iter_batches_preserves_rows_against_iter(materialised_smoke):
    """The fast batched path must contain exactly the same per-row data as
    the per-example __iter__ path. This is the contract that lets training
    swap from `route-per-example + _to_tensors(batch)` to `iter_batches`
    without changing the loss curve."""
    _, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)

    # Collect per-head expected sequences via __iter__.
    expected: dict[str, list[BCExample]] = {h: [] for h in HEAD_LOGIT_DIMS}
    for ex in ds:
        expected[ex.decision_type].append(ex)

    # Walk batched output and re-assemble per-head sequences.
    got: dict[str, dict[str, list]] = {
        h: {"features": [], "target": [], "legal_mask": [],
            "sample_weight": [], "skill_decile": []}
        for h in HEAD_LOGIT_DIMS
    }
    for head, batch in ds.iter_batches(batch_size=8, preserve_order=True):
        got[head]["features"].append(batch["features"])
        got[head]["target"].append(batch["target"])
        got[head]["legal_mask"].append(batch["legal_mask"])
        got[head]["sample_weight"].append(batch["sample_weight"])
        got[head]["skill_decile"].append(batch["skill_decile"])

    for head, exs in expected.items():
        if not exs:
            continue
        feat_got = np.concatenate(got[head]["features"])
        feat_exp = np.stack([e.features for e in exs])
        np.testing.assert_array_equal(feat_got, feat_exp)

        tgt_got = np.concatenate(got[head]["target"])
        tgt_exp = np.array([e.target for e in exs], dtype=np.int64)
        np.testing.assert_array_equal(tgt_got, tgt_exp)

        mask_got = np.concatenate(got[head]["legal_mask"])
        mask_exp = np.stack([e.legal_mask for e in exs])
        np.testing.assert_array_equal(mask_got, mask_exp)


def test_iter_batches_drop_last(materialised_smoke):
    """`drop_last=True` must drop partial trailing batches; default keeps them."""
    _, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    bs = 7  # 20 examples per head / 7 → 2 full + 1 partial
    kept_counts: dict[str, int] = {h: 0 for h in HEAD_LOGIT_DIMS}
    for head, batch in ds.iter_batches(bs, drop_last=False, preserve_order=False):
        kept_counts[head] += len(batch["target"])
    dropped_counts: dict[str, int] = {h: 0 for h in HEAD_LOGIT_DIMS}
    for head, batch in ds.iter_batches(bs, drop_last=True, preserve_order=False):
        dropped_counts[head] += len(batch["target"])
    # Each head has 20 rows; drop_last=True yields 14 (2 full batches of 7).
    for head in HEAD_LOGIT_DIMS:
        assert kept_counts[head] == 20
        assert dropped_counts[head] == 14


def test_version_mismatch_refuses_to_load(materialised_smoke):
    """Reader must refuse a bundle whose featurizer_version doesn't match
    the caller's expectation. Same contract as ParquetBCDataset's pin."""
    _, out_dir = materialised_smoke
    with pytest.raises(VersionMismatchError, match="featurizer_version"):
        MemmapBCDataset(
            out_dir,
            expected_featurizer_version="NOT-A-REAL-VERSION",
        )


def test_schema_version_mismatch_refuses_to_load(materialised_smoke):
    """A future schema bump must invalidate older bundles cleanly."""
    _, out_dir = materialised_smoke
    with pytest.raises(VersionMismatchError, match="schema_version"):
        MemmapBCDataset(
            out_dir,
            expected_schema_version=MATERIALISED_SCHEMA_VERSION + 1,
        )


def test_action_space_version_mismatch_refuses_to_load(materialised_smoke):
    """Action-space changes invalidate target/mask semantics — must be caught."""
    _, out_dir = materialised_smoke
    with pytest.raises(VersionMismatchError, match="action_space_version"):
        MemmapBCDataset(
            out_dir,
            expected_action_space_version="NOT-A-REAL-VERSION",
        )


def test_missing_manifest_raises_filenotfound(tmp_path: Path):
    """An empty directory looks nothing like a bundle; hint at the CLI."""
    with pytest.raises(FileNotFoundError, match="materialise_bc"):
        MemmapBCDataset(tmp_path)


def test_iter_batches_invalid_batch_size_raises(materialised_smoke):
    _, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    with pytest.raises(ValueError, match="batch_size"):
        next(iter(ds.iter_batches(0)))


def test_n_rows_matches_manifest(materialised_smoke):
    src_examples, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    assert ds.n_rows == len(src_examples)
    assert ds.total == ds.n_rows


def test_chunked_write_equivalent_to_unchunked(tmp_path: Path):
    """A bundle materialised with chunk_size=7 must be byte-identical to
    one materialised with a chunk_size larger than the stream — the
    chunking is a memory optimisation, not a semantic change. Tests the
    streaming append path is correct on the row-index bookkeeping
    (which differs between chunked vs single-shot writes).
    """
    src = list(SyntheticBCDataset(seed=11, n_per_head=23, skill_buckets=10, binary_features=True))
    big = tmp_path / "big_chunk"
    tiny = tmp_path / "tiny_chunk"
    materialise(iter(src), big, chunk_size=10_000)  # one chunk per type
    materialise(iter(src), tiny, chunk_size=7)      # many chunks per type

    # All .dat files must be byte-identical.
    for name in (
        "play_feat_bits.dat", "play_feat_cont.dat",
        "play_legal_mask.dat", "play_meta.dat",
        "wish_feat_bits.dat", "wish_feat_cont.dat",
        "wish_legal_mask.dat", "wish_meta.dat",
        "dragon_assignment_feat_bits.dat",
        "dragon_assignment_feat_cont.dat",
        "dragon_assignment_legal_mask.dat",
        "dragon_assignment_meta.dat",
        "order.dat",
    ):
        assert (big / name).read_bytes() == (tiny / name).read_bytes(), (
            f"chunked mismatch in {name}"
        )

    # And the round-tripped streams must match.
    big_examples = list(MemmapBCDataset(big))
    tiny_examples = list(MemmapBCDataset(tiny))
    assert len(big_examples) == len(tiny_examples)
    for a, b in zip(big_examples, tiny_examples):
        assert a.decision_type == b.decision_type
        assert a.target == b.target
        np.testing.assert_array_equal(a.features, b.features)


def test_no_cap_drains_stream(tmp_path: Path):
    """`max_examples=None` (the default) must consume the entire input
    iterable. Production runs rely on this — the CLI default is no-cap."""
    src = list(SyntheticBCDataset(seed=5, n_per_head=11, skill_buckets=10, binary_features=True))
    out_dir = tmp_path / "uncapped"
    counts = materialise(iter(src), out_dir)  # no max_examples kwarg
    assert sum(counts.values()) == len(src)


def test_max_examples_caps_at_target(tmp_path: Path):
    """Cap semantics: at most `max_examples` rows written, even if the
    stream is longer."""
    src = list(SyntheticBCDataset(seed=5, n_per_head=20, skill_buckets=10, binary_features=True))
    assert len(src) > 30  # sanity
    out_dir = tmp_path / "capped"
    counts = materialise(iter(src), out_dir, max_examples=30)
    assert sum(counts.values()) == 30


def test_idempotent_rerun_overwrites_prior_bundle(tmp_path: Path):
    """Running materialise twice into the same out_dir must produce the
    same bundle as one run — the prior .dat files are truncated, not
    appended-to. Without truncation, a re-run would corrupt the bundle.
    """
    src = list(SyntheticBCDataset(seed=3, n_per_head=12, skill_buckets=10, binary_features=True))
    out_dir = tmp_path / "rerun"
    materialise(iter(src), out_dir)
    first_sizes = {
        p.name: p.stat().st_size
        for p in out_dir.iterdir() if p.suffix == ".dat"
    }
    materialise(iter(src), out_dir)  # same data, same out_dir
    second_sizes = {
        p.name: p.stat().st_size
        for p in out_dir.iterdir() if p.suffix == ".dat"
    }
    assert first_sizes == second_sizes
    # And the bundle still reads correctly.
    got = list(MemmapBCDataset(out_dir))
    assert len(got) == len(src)


def test_game_won_none_roundtrips(materialised_smoke):
    """SyntheticBCDataset emits ~10% game_won=None rows; the i1 sentinel
    encoding (-1) must roundtrip through materialise + read."""
    src_examples, out_dir = materialised_smoke
    ds = MemmapBCDataset(out_dir)
    src_nones = sum(1 for e in src_examples if e.game_won is None)
    got_nones = sum(1 for e in ds if e.game_won is None)
    assert src_nones == got_nones
    # And the seed-42 stream must contain at least one None to make this
    # test meaningful.
    assert src_nones > 0
