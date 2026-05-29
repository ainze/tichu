"""`parse_bsw --bundle-out-dir` writes the materialised bundle in the same
replay pass that produces the parquet shards (consolidation handoff)."""

from pathlib import Path

from tichu_training.bc.materialised import MemmapBCDataset
from tichu_training.cli.parse_bsw import main


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def test_bundle_out_dir_flag_writes_both_parquet_and_bundle(tmp_path):
    parquet_out = tmp_path / "parquet"
    bundle_out = tmp_path / "bundle"
    rc = main([
        "--input", str(_SAMPLES),
        "--output", str(parquet_out),
        "--bundle-out-dir", str(bundle_out),
    ])
    assert rc == 0

    # Parquet shards still land as before.
    assert {p.stem for p in parquet_out.glob("*.parquet")}, "no parquet shards written"

    # And the bundle is readable with rows.
    ds = MemmapBCDataset(bundle_out)
    assert ds.total > 0
