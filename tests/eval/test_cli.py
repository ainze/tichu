"""Smoke + determinism tests for the `eval_matrix` CLI."""

import textwrap
from pathlib import Path

import pyarrow.parquet as pq

from tichu_eval.starting_position_pool import generate_starting_position_pool, save_starting_position_pool
from tichu_eval.full_position_pool import generate_full_position_pool, save_full_position_pool
from tichu_training.cli.eval_matrix import main


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


def test_config_copied_into_output_run_dir(tmp_path):
    pool_path = _write_pool(tmp_path, n=4)
    output = tmp_path / "out" / "matrix.parquet"
    cfg = _write_config(tmp_path, pool_path=pool_path, n_deals=4, output=output)

    assert main(["--config", str(cfg)]) == 0
    # Config copied next to the output for reproducibility.
    assert (output.parent / cfg.name).exists()
