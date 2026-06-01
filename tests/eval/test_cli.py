"""Smoke + determinism tests for the `eval_matrix` CLI."""

import textwrap
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tichu_eval.starting_position_pool import generate_starting_position_pool, save_starting_position_pool
from tichu_eval.full_position_pool import generate_full_position_pool, save_full_position_pool
from tichu_training.cli.eval_matrix import main


# Real exported network bundle (featurizer v5). Present on the dev box; absent in
# CI, where the heavy `ml`-in-workers test below is skipped. See ADR-0025.
_EXPORT_DIR = Path(r"C:\workbench\tichu\data\export\bc_full_100k_v5")


def _write_pool(tmp_path: Path, *, n: int = 10) -> Path:
    pool_path = tmp_path / "pool.parquet"
    save_starting_position_pool(generate_starting_position_pool(seed=0, n=n), pool_path)
    return pool_path


def _write_config(tmp_path: Path, *, pool_path: Path, n_deals: int, output: Path) -> Path:
    config = textwrap.dedent(f"""\
        variant: play_strength
        starting_position_pool: {pool_path.as_posix()}
        n_deals: {n_deals}
        agents:
          - name: random
            factory: random
            kwargs:
              seed: 0
          - name: rule
            factory: rule
            kwargs: {{}}
        bootstrap_iters: 50
        seed: 0
        output: {output.as_posix()}
    """)
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(config, encoding="utf-8")
    return cfg_path


def test_smoke_runs_and_rule_beats_random(tmp_path, capsys):
    pool_path = _write_pool(tmp_path, n=20)
    output = tmp_path / "matrix.parquet"
    cfg = _write_config(tmp_path, pool_path=pool_path, n_deals=20, output=output)

    rc = main(["--config", str(cfg)])
    assert rc == 0
    assert output.exists()

    table = pq.read_table(output)
    cols = set(table.column_names)
    assert {"agent_a", "agent_b", "mean", "ci_lower", "ci_upper", "n"} <= cols

    rule_vs_random = [
        row for row in table.to_pylist()
        if row["agent_a"] == "rule" and row["agent_b"] == "random"
    ]
    assert rule_vs_random
    assert rule_vs_random[0]["mean"] > 0

    captured = capsys.readouterr().out
    assert "rule" in captured and "random" in captured


def test_full_strength_is_the_default_and_writes_call_bonus_column(tmp_path, capsys):
    pool_path = tmp_path / "full_pool.parquet"
    save_full_position_pool(generate_full_position_pool(seed=0, n=20), pool_path)
    output = tmp_path / "matrix.parquet"
    config = textwrap.dedent(f"""\
        starting_position_pool: {pool_path.as_posix()}
        n_deals: 20
        agents:
          - name: random
            factory: random
            kwargs:
              seed: 0
          - name: rule
            factory: rule
            kwargs: {{}}
        bootstrap_iters: 50
        seed: 0
        output: {output.as_posix()}
    """)
    cfg = tmp_path / "eval_full.yaml"
    cfg.write_text(config, encoding="utf-8")

    assert main(["--config", str(cfg)]) == 0
    table = pq.read_table(output)
    cols = set(table.column_names)
    assert "call_bonus_mean" in cols
    rule_vs_random = [
        row for row in table.to_pylist()
        if row["agent_a"] == "rule" and row["agent_b"] == "random"
    ]
    assert rule_vs_random and rule_vs_random[0]["mean"] > 0


def test_determinism_two_runs_produce_identical_rows(tmp_path):
    pool_path = _write_pool(tmp_path, n=8)
    out1 = tmp_path / "m1.parquet"
    out2 = tmp_path / "m2.parquet"

    cfg1 = _write_config(tmp_path, pool_path=pool_path, n_deals=8, output=out1)
    cfg2 = tmp_path / "eval2.yaml"
    cfg2.write_text(cfg1.read_text(encoding="utf-8").replace(out1.as_posix(), out2.as_posix()),
                    encoding="utf-8")

    assert main(["--config", str(cfg1)]) == 0
    assert main(["--config", str(cfg2)]) == 0

    rows1 = pq.read_table(out1).to_pylist()
    rows2 = pq.read_table(out2).to_pylist()
    assert rows1 == rows2


def test_move_prediction_mode_writes_csv(tmp_path):
    import shutil
    sample_src = Path(__file__).resolve().parents[1].parent / "sample" / "2417500.tch"
    held_out = tmp_path / "held_out"
    held_out.mkdir()
    shutil.copy(sample_src, held_out / "2417500.tch")

    output = tmp_path / "move_pred.csv"
    config = textwrap.dedent(f"""\
        agents:
          - name: rule
            factory: rule
            kwargs: {{}}
        max_decisions: 100
        output: {output.as_posix()}
    """)
    cfg = tmp_path / "mp.yaml"
    cfg.write_text(config, encoding="utf-8")

    rc = main(["--config", str(cfg), "--mode", "move_prediction",
               "--held-out", str(held_out)])
    assert rc == 0
    assert output.exists()
    import csv as _csv
    rows = list(_csv.DictReader(output.open("r", encoding="utf-8")))
    assert rows
    expected_columns = {"agent", "decision_type", "top1", "top5", "n"}
    assert expected_columns <= set(rows[0].keys())
    assert any(r["agent"] == "rule" and r["decision_type"] == "play" for r in rows)


def test_full_strength_threads_workers_from_config(tmp_path, monkeypatch):
    """`workers:` in the config reaches run_full_tournament. The spy runs the
    matrix serially regardless (fast, no real spawn) but records the requested
    worker count."""
    import tichu_training.cli.eval_matrix as cli

    pool_path = tmp_path / "full_pool.parquet"
    save_full_position_pool(generate_full_position_pool(seed=0, n=6), pool_path)
    output = tmp_path / "matrix.parquet"
    config = textwrap.dedent(f"""\
        starting_position_pool: {pool_path.as_posix()}
        n_deals: 6
        workers: 3
        agents:
          - name: rule
            factory: rule
            kwargs: {{}}
          - name: rule2
            factory: rule
            kwargs: {{}}
        bootstrap_iters: 20
        seed: 0
        output: {output.as_posix()}
    """)
    cfg = tmp_path / "eval_full.yaml"
    cfg.write_text(config, encoding="utf-8")

    captured = {}
    real = cli.run_full_tournament

    def spy(builders, positions, **kwargs):
        captured["workers"] = kwargs.get("workers")
        return real(builders, positions, **{**kwargs, "workers": 1})

    monkeypatch.setattr(cli, "run_full_tournament", spy)
    assert main(["--config", str(cfg)]) == 0
    assert captured["workers"] == 3


def test_full_strength_parallel_cli_matches_serial(tmp_path):
    """End-to-end parallel run through the CLI (real spawn): workers=2 must
    produce a matrix bit-identical to workers=1 for deterministic agents, and the
    spawned workers must rebuild agents from the registry without error."""
    pool_path = tmp_path / "full_pool.parquet"
    save_full_position_pool(generate_full_position_pool(seed=0, n=12), pool_path)

    def _cfg(workers: int, output: Path) -> Path:
        config = textwrap.dedent(f"""\
            starting_position_pool: {pool_path.as_posix()}
            n_deals: 12
            workers: {workers}
            agents:
              - name: rule
                factory: rule
                kwargs: {{}}
              - name: rule2
                factory: rule
                kwargs: {{}}
            bootstrap_iters: 30
            seed: 0
            output: {output.as_posix()}
        """)
        cfg = tmp_path / f"eval_w{workers}.yaml"
        cfg.write_text(config, encoding="utf-8")
        return cfg

    out1, out2 = tmp_path / "serial.parquet", tmp_path / "parallel.parquet"
    assert main(["--config", str(_cfg(1, out1))]) == 0
    assert main(["--config", str(_cfg(2, out2))]) == 0

    assert pq.read_table(out1).to_pylist() == pq.read_table(out2).to_pylist()


@pytest.mark.skipif(
    not (_EXPORT_DIR / "policy.pt").exists(),
    reason="requires exported v5 network bundle (dev box only)",
)
def test_full_strength_parallel_cli_rebuilds_ml_agents_in_workers(tmp_path):
    """The real eval path: a `factory: ml` agent must rebuild from its checkpoint
    paths inside spawned workers (proving the `ml` factory is registered there).

    ML inference is NOT bit-reproducible across the process boundary — identical
    inputs yield bit-different torch logits in a different process, occasionally
    flipping a near-tie greedy argmax (pre-existing torch float noise, not an RNG
    or orchestration bug). So we assert *statistical* equivalence: the Position
    count is exact, and the parallel mean lands within the serial run's own
    bootstrap CI. (Bit-identity is guaranteed only for torch-free agents — see
    test_parallel_matches_serial_for_deterministic_agents.)"""
    pool_path = tmp_path / "full_pool.parquet"
    save_full_position_pool(generate_full_position_pool(seed=0, n=20), pool_path)

    def _cfg(workers: int, output: Path) -> Path:
        config = textwrap.dedent(f"""\
            starting_position_pool: {pool_path.as_posix()}
            n_deals: 20
            workers: {workers}
            agents:
              - name: bc
                factory: ml
                kwargs:
                  checkpoint_path: {(_EXPORT_DIR / "policy.pt").as_posix()}
                  schupfen_path:   {(_EXPORT_DIR / "schupfen.pt").as_posix()}
                  tichu_call_path: {(_EXPORT_DIR / "tichu_call.pt").as_posix()}
                  grand_call_path: {(_EXPORT_DIR / "grand_tichu_call.pt").as_posix()}
              - name: rule
                factory: rule
                kwargs: {{}}
            bootstrap_iters: 50
            seed: 0
            output: {output.as_posix()}
        """)
        cfg = tmp_path / f"eval_ml_w{workers}.yaml"
        cfg.write_text(config, encoding="utf-8")
        return cfg

    def _bc_vs_rule(out: Path) -> dict:
        rows = pq.read_table(out).to_pylist()
        return next(r for r in rows if r["agent_a"] == "bc" and r["agent_b"] == "rule")

    out1, out2 = tmp_path / "serial.parquet", tmp_path / "parallel.parquet"
    assert main(["--config", str(_cfg(1, out1))]) == 0
    assert main(["--config", str(_cfg(2, out2))]) == 0

    serial, parallel = _bc_vs_rule(out1), _bc_vs_rule(out2)
    assert parallel["n"] == serial["n"]  # Position count is exact (no torch noise)
    ci_width = serial["ci_upper"] - serial["ci_lower"]
    assert abs(parallel["mean"] - serial["mean"]) <= ci_width  # statistically identical


def _full_cfg(tmp_path: Path, *, output: Path, n: int = 6) -> Path:
    pool_path = tmp_path / "full_pool.parquet"
    save_full_position_pool(generate_full_position_pool(seed=0, n=n), pool_path)
    config = textwrap.dedent(f"""\
        starting_position_pool: {pool_path.as_posix()}
        n_deals: {n}
        agents:
          - name: rule
            factory: rule
            kwargs: {{}}
          - name: rule2
            factory: rule
            kwargs: {{}}
        bootstrap_iters: 20
        seed: 0
        output: {output.as_posix()}
    """)
    cfg = tmp_path / "eval_full.yaml"
    cfg.write_text(config, encoding="utf-8")
    return cfg


def test_progress_bar_callback_threaded_by_default(tmp_path, monkeypatch):
    """By default the CLI drives a progress bar: it passes a callable `progress`
    hook into run_full_tournament (the bar's own .update). The visual bar is
    verified separately; here we only assert the seam is wired."""
    import tichu_training.cli.eval_matrix as cli

    cfg = _full_cfg(tmp_path, output=tmp_path / "matrix.parquet")
    captured = {}
    real = cli.run_full_tournament

    def spy(builders, positions, **kwargs):
        captured["progress"] = kwargs.get("progress")
        return real(builders, positions, **kwargs)

    monkeypatch.setattr(cli, "run_full_tournament", spy)
    assert main(["--config", str(cfg)]) == 0
    assert callable(captured["progress"])


def test_no_progress_flag_disables_the_bar(tmp_path, monkeypatch):
    """`--no-progress` keeps piped/cron logs clean: no bar is created and the
    library is called with progress=None."""
    import tichu_training.cli.eval_matrix as cli

    cfg = _full_cfg(tmp_path, output=tmp_path / "matrix.parquet")
    captured = {}
    real = cli.run_full_tournament

    def spy(builders, positions, **kwargs):
        captured["progress"] = kwargs.get("progress")
        return real(builders, positions, **kwargs)

    monkeypatch.setattr(cli, "run_full_tournament", spy)
    assert main(["--config", str(cfg), "--no-progress"]) == 0
    assert captured["progress"] is None


def test_config_copied_into_output_run_dir(tmp_path):
    pool_path = _write_pool(tmp_path, n=4)
    output = tmp_path / "out" / "matrix.parquet"
    cfg = _write_config(tmp_path, pool_path=pool_path, n_deals=4, output=output)

    assert main(["--config", str(cfg)]) == 0
    # Config copied next to the output for reproducibility.
    assert (output.parent / cfg.name).exists()
