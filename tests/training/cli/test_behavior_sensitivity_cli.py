"""Behavior Sensitivity Probe CLI — calibrate → run → summarise, end to end on a
dummy exported policy vs the RuleAgent (tiny, parallel)."""

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

from tichu_training.action_space import ACTION_SPACE_SIZE
from tichu_training.cli.behavior_sensitivity import main
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM

from tests.inference.test_ml_agent import _export_dummy


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("probe")
    artifact = _export_dummy(tmp, feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    config = {
        "champion": {"name": "dummy", "factory": "ml",
                     "kwargs": {"checkpoint_path": str(artifact), "skill_decile": 9}},
        "opponent": {"name": "rule", "factory": "rule"},
        "levers": ["pass_vs_opponent", "single_lead"],
        "pool_seed": 0, "n_deals": 12, "workers": 2, "n_boot": 200, "floor": 0,
        "identity_check_deals": 4,
        "calibration": {"pool_seed": 1000, "n_deals": 6, "share": 0.10, "cap": 8.0},
        "output_dir": str(tmp / "out"),
    }
    path = tmp / "probe.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    for step in ("calibrate", "run", "summarise"):
        assert main([step, "--config", str(path)]) == 0
    return tmp / "out", path


def test_calibration_freezes_a_signed_delta_per_lever(probe):
    out, _ = probe
    deltas = yaml.safe_load((out / "calibration" / "deltas.yaml").read_text(encoding="utf-8"))
    assert set(deltas) == {"pass_vs_opponent", "single_lead"}
    for d in deltas.values():
        assert d["plus"] >= 0 >= d["minus"] and d["situations"] > 0


def test_run_plays_the_reference_and_every_arm(probe):
    out, _ = probe
    assert (out / "reference" / "rounds.parquet").exists()
    for lever in ("pass_vs_opponent", "single_lead"):
        for sign in ("plus", "minus"):
            assert (out / "arms" / f"{lever}_{sign}" / "rounds.parquet").exists()


def test_summarise_writes_one_row_per_arm_and_a_verdict(probe):
    out, _ = probe
    table = pd.read_csv(out / "results.csv")
    assert len(table) == 4 and set(table.sign) == {"+", "-"}
    assert {"d_ev", "d_ev_lo", "d_ev_hi", "d_rate", "slope", "holm"} <= set(table.columns)
    assert "global null" in (out / "verdict.md").read_text(encoding="utf-8")


def test_a_positive_pass_bias_raises_the_pass_rate(probe):
    out, _ = probe
    table = pd.read_csv(out / "results.csv").set_index(["lever", "sign"])
    assert table.loc[("pass_vs_opponent", "+"), "d_rate"] >= 0
    assert table.loc[("pass_vs_opponent", "-"), "d_rate"] <= 0


def test_frozen_deltas_are_not_recalibrated_after_arms_ran(probe):
    _, path = probe
    with pytest.raises(SystemExit):
        main(["calibrate", "--config", str(path), "--force"])
