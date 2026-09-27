"""Smoke tests for the Behavioral Drift Benchmark CLI (`drift_benchmark`)."""

import textwrap

import pandas as pd

from tichu_training.cli.drift_benchmark import main


def _config(tmp_path, out):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        subject: {{name: rule-a, factory: rule}}
        bc:      {{name: rule-b, factory: rule}}
        pool_seed: 0
        n_deals: 6
        workers: 1
        n_boot: 50
        floor: 0
        output_dir: {out.as_posix()}
    """), encoding="utf-8")
    return cfg


def test_run_writes_log_summary_and_report(tmp_path):
    out = tmp_path / "run"
    assert main(["--config", str(_config(tmp_path, out))]) == 0
    for f in ("decisions.parquet", "rounds.parquet", "summary.csv", "report.html", "drift.yaml"):
        assert (out / f).exists(), f
    summary = pd.read_csv(out / "summary.csv")
    assert {"metric", "cell", "arm", "rate", "rate_lo", "rate_hi", "bh_survives"} <= set(summary.columns)
    assert "rule-a" in (out / "report.html").read_text(encoding="utf-8")


def test_aa_mode_writes_to_its_own_directory(tmp_path):
    out = tmp_path / "run"
    assert main(["--config", str(_config(tmp_path, out)), "--aa"]) == 0
    assert (out / "aa" / "summary.csv").exists() and (out / "aa" / "report.html").exists()
    assert not (out / "summary.csv").exists()


def test_resummarise_from_an_existing_log_without_replaying(tmp_path):
    out = tmp_path / "run"
    cfg = _config(tmp_path, out)
    main(["--config", str(cfg)])
    (out / "summary.csv").unlink()
    stamp = (out / "decisions.parquet").stat().st_mtime_ns
    assert main(["--config", str(cfg), "--from-log"]) == 0
    assert (out / "summary.csv").exists()
    assert (out / "decisions.parquet").stat().st_mtime_ns == stamp   # not replayed


def _config_with_humans(tmp_path, out):
    from pathlib import Path

    fixtures = Path(__file__).resolve().parents[1] / "training" / "bsw" / "data"
    cfg = _config(tmp_path, out)
    cfg.write_text(cfg.read_text(encoding="utf-8") + textwrap.dedent(f"""\
        human:
          source: {fixtures.as_posix()}
          groups:
            - {{name: humans (all), n_rounds: 30}}
    """), encoding="utf-8")
    return cfg


def test_human_reference_is_logged_cached_and_reported(tmp_path):
    out = tmp_path / "run"
    cfg = _config_with_humans(tmp_path, out)
    assert main(["--config", str(cfg)]) == 0
    human_dir = out / "human" / "humans-all"
    assert (human_dir / "decisions.parquet").exists()
    summary = pd.read_csv(out / "summary.csv")
    assert "humans (all)" in set(summary.arm)
    assert "humans (all)" in (out / "report.html").read_text(encoding="utf-8")

    stamp = (human_dir / "decisions.parquet").stat().st_mtime_ns
    assert main(["--config", str(cfg), "--from-log"]) == 0          # reused, not replayed
    assert (human_dir / "decisions.parquet").stat().st_mtime_ns == stamp


def test_archive_sample_spans_the_whole_archive():
    # Regression: the sample was a stride sized for ~200k Games, read in archive
    # order and cut off after the first ~1k — i.e. a sliver of BSW history.
    # The stride is now sized to the Games the group actually samples.
    from tichu_training.cli.drift_benchmark import _sample_ids

    ids = [str(i) for i in range(1_000_000)]
    picked = _sample_ids(ids, sample_games=1000)
    assert len(picked) == 1000
    positions = sorted(int(i) for i in picked)
    assert positions[0] < 1_000 and positions[-1] > 998_000   # first and last thousandth
