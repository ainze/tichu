"""`parse_bsw --bundle-out-dir ROOT --bundle-tasks ...` materialises one
bundle per task in the same replay pass (one parse pass, many outputs —
ADR-0020). The bundles land in ROOT/<task>/ subdirectories."""

from pathlib import Path

from tichu_training.bc.call_materialised import MemmapCallDataset
from tichu_training.bc.materialised import MemmapBCDataset
from tichu_training.bc.schupfen_materialised import MemmapSchupfenDataset
from tichu_training.belief.belief_materialised import MemmapBeliefDataset
from tichu_training.cli.parse_bsw import main


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def test_bundle_out_dir_writes_all_task_bundles_in_one_pass(tmp_path):
    parquet_out = tmp_path / "parquet"
    bundle_root = tmp_path / "bundles"
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(parquet_out),
        "--bundle-out-dir", str(bundle_root),
        # default --bundle-tasks all => bc, calls, schupfen
    ])
    assert rc == 0

    # Parquet shards still land as before.
    assert {p.stem for p in parquet_out.glob("*.parquet")}, "no parquet shards written"

    # Each task bundle is readable from its ROOT/<task>/ subdir, with rows.
    bc = MemmapBCDataset(bundle_root / "bc")
    assert bc.total > 0

    schupfen = MemmapSchupfenDataset(bundle_root / "schupfen")
    assert len(schupfen) > 0

    tichu = MemmapCallDataset(bundle_root / "calls", call_type="call_tichu")
    grand = MemmapCallDataset(bundle_root / "calls", call_type="call_grand_tichu")
    assert len(grand) > 0  # one per seat per validated round, always present

    # Provenance is populated by the consolidated pass (0 only for synthetic).
    sample_ids = {int(p.stem) for p in _SAMPLES.glob("*.tch")}
    any_example = next(iter(schupfen))
    assert any_example.game_id in sample_ids
    grand_example = next(iter(grand))
    assert grand_example.game_id in sample_ids

    # Belief is gated OUT of `all` (ADR-0021) — no belief bundle by default.
    assert not (bundle_root / "belief").exists()


def test_belief_is_materialised_only_when_named_explicitly(tmp_path):
    parquet_out = tmp_path / "parquet"
    bundle_root = tmp_path / "bundles"
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(parquet_out),
        "--bundle-out-dir", str(bundle_root),
        "--bundle-tasks", "belief",
    ])
    assert rc == 0
    belief = MemmapBeliefDataset(bundle_root / "belief")
    assert len(belief) > 0
    sample_ids = {int(p.stem) for p in _SAMPLES.glob("*.tch")}
    ex = next(iter(belief))
    assert ex.labels.shape == (3, 56)
    assert ex.mask.shape == (3, 56)
    assert ex.game_id in sample_ids
    assert 0 <= ex.cards_played <= 56
    # belief-only run writes no other task bundle
    assert not (bundle_root / "bc").exists()


def test_bundle_tasks_subset_writes_only_selected(tmp_path):
    parquet_out = tmp_path / "parquet"
    bundle_root = tmp_path / "bundles"
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(parquet_out),
        "--bundle-out-dir", str(bundle_root),
        "--bundle-tasks", "schupfen",
    ])
    assert rc == 0
    assert (bundle_root / "schupfen").is_dir()
    assert not (bundle_root / "bc").exists()
    assert not (bundle_root / "calls").exists()


def test_unknown_bundle_task_is_rejected(tmp_path):
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(tmp_path / "parquet"),
        "--bundle-out-dir", str(tmp_path / "bundles"),
        "--bundle-tasks", "bogus",
    ])
    assert rc == 2
