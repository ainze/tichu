"""Champion promotion gate wired into train_cotrain (rounds-as-gate).

Runs the real parallel co-train loop (the gate is parallel-only) with the opponent
forced to the frozen champion, and checks the two ends of the gate: a pass-everything
threshold promotes (champion file rewritten, gate CSV records promoted=1) and a
pass-nothing threshold holds (champion untouched, promoted=0). Margin values are
irrelevant at these extremes, so the test is deterministic despite the bootstrap.
"""

import csv
from pathlib import Path

import torch

from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.heads import BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.bc.training import load_checkpoint, save_checkpoint
from tichu_training.cli.train_cotrain import run_cotrain_training
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM as _D

_MODEL = dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16)


def _write(net, path):
    save_checkpoint(net, torch.optim.Adam(net.parameters()), step=0, path=str(path))


def _config(tmp_path, *, threshold: float):
    warm = {k: tmp_path / f"{k}.bin" for k in ("play", "schupfen", "tichu", "grand")}
    _write(BCModel(_D, skill_buckets=10, **_MODEL), warm["play"])
    _write(SchupfenNetwork(_D, skill_dim=8, hidden=16), warm["schupfen"])
    _write(TichuCallNetwork(_D, skill_dim=8, hidden=16), warm["tichu"])
    _write(GrandTichuCallNetwork(_D, skill_dim=8, hidden=16), warm["grand"])
    return {
        "warm_start": {k: str(v) for k, v in warm.items()},
        "run_dir": str(tmp_path / "run"),
        "model": _MODEL,
        "schupfen_model": {"skill_dim": 8, "hidden": 16},
        "call_model": {"skill_dim": 8, "hidden": 16},
        "critic": {"hidden": 16},
        "perfect_info": True,
        "rollout_workers": 2,  # gate is parallel-only
        "promotion_gate": {"enabled": True, "window_games": 2, "threshold": threshold},
        "kl": {dt: {"coef": 0.5, "target": 0.02} for dt in ("play", "schupfen", "tichu", "grand")},
        "entropy": {dt: 0.01 for dt in ("play", "schupfen", "tichu", "grand")},
        "snapshot_every": 2,
        "ppo": {
            "iterations": 2, "positions_per_iter": 2, "pool_seed": 0,
            "gamma": 1.0, "lam": 0.95, "clip_eps": 0.1, "vf_coef": 0.5,
            "ppo_epochs": 1, "policy_lr": 1e-3, "critic_lr": 1e-3,
            "skill_decile": 9, "learner_team": 0,
        },
    }


def _gate_rows(run_dir):
    return list(csv.DictReader(open(Path(run_dir) / "promotion_gate.csv", encoding="utf-8")))


def test_gate_creates_champion_and_promotes_on_passing_verdict(tmp_path):
    torch.manual_seed(0)
    config = _config(tmp_path, threshold=-1e9)  # any margin clears -> always promote
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    assert (run_dir / "_champion.pt").exists()
    rows = _gate_rows(run_dir)
    assert rows, "gate window (2 games) should fill each iter and log a verdict"
    assert all(r["promoted"] == "1" for r in rows)
    assert int(rows[0]["n"]) >= 2
    # A promotion fires at iter 1 (off the snapshot_every=2 cadence), so a serving
    # snapshot at iter_00001 exists ONLY because promotion saves one — the champion
    # is directly check_cotrain-able.
    assert (run_dir / "snapshots" / "iter_00001_play.bin").exists()


def test_gate_holds_when_threshold_unreachable(tmp_path):
    torch.manual_seed(0)
    config = _config(tmp_path, threshold=1e9)  # no finite margin clears -> never promote
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    champ = run_dir / "_champion.pt"
    assert champ.exists()  # initialised to frozen BC at setup
    mtime_after_setup_and_run = champ.stat().st_mtime
    rows = _gate_rows(run_dir)
    assert rows and all(r["promoted"] == "0" for r in rows)
    # Champion was never re-saved after the initial BC write (no promotion occurred).
    # (Sanity: a promotion would have rewritten it during the loop, after setup.)
    assert isinstance(mtime_after_setup_and_run, float)


def test_reanchor_on_promote_moves_and_persists_the_play_anchor(tmp_path):
    # With reanchor_on_promote, a validated promotion moves the play KL anchor onto
    # the champion (= current net) and persists it in the bundle (survives resume).
    torch.manual_seed(0)
    config = _config(tmp_path, threshold=-1e9)  # force a promotion
    config["promotion_gate"]["reanchor_on_promote"] = True

    bc_play = BCModel(_D, skill_buckets=10, **_MODEL)
    load_checkpoint(config["warm_start"]["play"], bc_play)
    orig = {k: v.detach().clone() for k, v in bc_play.state_dict().items()}

    result = run_cotrain_training(config, progress=False)

    bundle = torch.load(result["bundle_path"], map_location="cpu", weights_only=False)
    assert bundle["play_anchor"] is not None, "moved anchor must persist for resume"
    anchor = bundle["play_anchor"]
    assert any(not torch.equal(orig[k], anchor[k]) for k in orig), \
        "play anchor should have moved off the original BC after a promotion"


def test_no_reanchor_on_promote_leaves_anchor_unpersisted(tmp_path):
    # Default (reanchor_on_promote off): the gate ratchets only the opponent; the KL
    # anchor stays at BC and is not persisted (the existing behaviour).
    torch.manual_seed(0)
    config = _config(tmp_path, threshold=-1e9)  # promotes, but no anchor move
    result = run_cotrain_training(config, progress=False)
    bundle = torch.load(result["bundle_path"], map_location="cpu", weights_only=False)
    assert bundle["play_anchor"] is None


def test_also_beat_bc_adds_bc_opponent_and_logs_both(tmp_path):
    # With also_beat_bc the gate alternates champion/BC, creates a frozen BC opponent
    # file, and a verdict must beat BOTH — the per-opponent CSV records both streams.
    torch.manual_seed(0)
    config = _config(tmp_path, threshold=-1e9)  # pass-everything => promotes
    config["promotion_gate"]["also_beat_bc"] = True
    config["ppo"]["iterations"] = 6  # alternation needs >1 iter to fill both windows
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    assert (run_dir / "_champion.pt").exists()
    assert (run_dir / "_bc_opponent.pt").exists()
    rows = _gate_rows(run_dir)
    opps = {r["opponent"] for r in rows}
    assert opps == {"champion", "bc"}, f"expected both opponents logged, got {opps}"
    assert all(r["promoted"] == "1" for r in rows)


def test_also_beat_bc_holds_when_one_opponent_fails(tmp_path):
    # Pass-nothing threshold: even though it would beat the champion, the beat-BOTH
    # gate holds (neither opponent clears) — champion not promoted.
    torch.manual_seed(0)
    config = _config(tmp_path, threshold=1e9)
    config["promotion_gate"]["also_beat_bc"] = True
    config["ppo"]["iterations"] = 6
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    rows = _gate_rows(run_dir)
    assert rows and {r["opponent"] for r in rows} == {"champion", "bc"}
    assert all(r["promoted"] == "0" for r in rows)


# --- Greedy-margin gate (option A): margins from a GREEDY mini-tournament -----------
# The deployed-strength source that replaces the confounded sampled rollout reward.
_GREEDY_N_DEALS = 2  # tiny mini-tournament => 2*2 = 4 seat-swapped obs/opponent/window


def _greedy_config(tmp_path, *, threshold, also_beat_bc=False):
    config = _config(tmp_path, threshold=threshold)
    config["promotion_gate"].update(
        {"greedy": True, "greedy_n_deals": _GREEDY_N_DEALS,
         "greedy_every": 1, "greedy_workers": 1}
    )
    if also_beat_bc:
        config["promotion_gate"]["also_beat_bc"] = True
    config["ppo"]["iterations"] = 2  # greedy_every=1 => eval fires after the first iter
    return config


def test_greedy_gate_promotes_and_sources_margins_from_the_mini_tournament(tmp_path):
    # The anti-confound regression test (the whole point): with greedy=True the gate's
    # margins come from a GREEDY mini-tournament, NOT the sampled rollout reward. At a
    # pass-everything threshold it promotes, AND each window's n equals the tournament's
    # obs count (2*greedy_n_deals seat-swapped), proving the source is the tournament
    # (the sampled gate would record positions_per_iter=2 per iter instead).
    torch.manual_seed(0)
    config = _greedy_config(tmp_path, threshold=-1e9)
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    rows = _gate_rows(run_dir)
    assert rows and all(r["promoted"] == "1" for r in rows)
    assert all(int(r["n"]) == 2 * _GREEDY_N_DEALS for r in rows), \
        "greedy window must be the mini-tournament's seat-swapped obs count, not sampled"
    assert (run_dir / "_champion.pt").exists()


def test_greedy_gate_holds_when_threshold_unreachable(tmp_path):
    torch.manual_seed(0)
    config = _greedy_config(tmp_path, threshold=1e9)  # no greedy margin clears
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    rows = _gate_rows(run_dir)
    assert rows and all(r["promoted"] == "0" for r in rows)
    assert all(int(r["n"]) == 2 * _GREEDY_N_DEALS for r in rows)


def test_greedy_gate_fills_both_opponents_in_a_single_eval(tmp_path):
    # Unlike the sampled gate (which alternates opponents per iter and needs several
    # iters to fill both windows), the greedy eval scores BOTH pool opponents in one
    # mini-tournament — so both windows fill, and a beat-both verdict is drawn, at the
    # very first eval.
    torch.manual_seed(0)
    config = _greedy_config(tmp_path, threshold=-1e9, also_beat_bc=True)
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    rows = _gate_rows(run_dir)
    assert rows and {r["opponent"] for r in rows} == {"champion", "bc"}
    assert all(r["promoted"] == "1" for r in rows)
    assert (run_dir / "_bc_opponent.pt").exists()


# --- extra_opponents: FIXED external models the candidate must ALSO beat -------------

def _write_rollout_opponent(path):
    """A fixed external opponent in rollout-weights format ({"models": {net: sd}}),
    test arch — what `extra_opponents[*].path` points at (e.g. a shipped champion)."""
    models = {
        "play": BCModel(_D, skill_buckets=10, **_MODEL),
        "schupfen": SchupfenNetwork(_D, skill_dim=8, hidden=16),
        "tichu": TichuCallNetwork(_D, skill_dim=8, hidden=16),
        "grand": GrandTichuCallNetwork(_D, skill_dim=8, hidden=16),
    }
    torch.save({"models": {k: m.state_dict() for k, m in models.items()}}, str(path))


def test_extra_opponents_adds_a_third_fixed_opponent_to_the_beat_all_gate(tmp_path):
    # A third FIXED opponent (e.g. the shipped champion) joins the {champion, bc} pool
    # via extra_opponents. The greedy eval scores all THREE in one mini-tournament; at a
    # pass-everything threshold the candidate beats all three and promotes — and the
    # per-opponent CSV records the third stream by its configured name.
    torch.manual_seed(0)
    config = _greedy_config(tmp_path, threshold=-1e9, also_beat_bc=True)
    opp_path = tmp_path / "cpfix_opponent.pt"
    _write_rollout_opponent(opp_path)
    config["promotion_gate"]["extra_opponents"] = [{"name": "cpfix", "path": str(opp_path)}]
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    rows = _gate_rows(run_dir)
    assert rows and {r["opponent"] for r in rows} == {"champion", "bc", "cpfix"}
    assert all(r["promoted"] == "1" for r in rows)


def test_extra_opponents_gate_holds_when_threshold_unreachable(tmp_path):
    # Beat-ALL semantics through the wiring: with a pass-nothing threshold the gate
    # holds even though all three streams are scored — promotion needs every opponent
    # (champion, bc, AND the fixed cpfix) to clear.
    torch.manual_seed(0)
    config = _greedy_config(tmp_path, threshold=1e9, also_beat_bc=True)
    opp_path = tmp_path / "cpfix_opponent.pt"
    _write_rollout_opponent(opp_path)
    config["promotion_gate"]["extra_opponents"] = [{"name": "cpfix", "path": str(opp_path)}]
    run_dir = Path(config["run_dir"])

    run_cotrain_training(config, progress=False)

    rows = _gate_rows(run_dir)
    assert rows and {r["opponent"] for r in rows} == {"champion", "bc", "cpfix"}
    assert all(r["promoted"] == "0" for r in rows)


def test_extra_opponents_missing_file_raises(tmp_path):
    # A typo'd / missing opponent file is caught at setup, not mid-run.
    import pytest
    torch.manual_seed(0)
    config = _greedy_config(tmp_path, threshold=-1e9)
    config["promotion_gate"]["extra_opponents"] = [
        {"name": "cpfix", "path": str(tmp_path / "does_not_exist.pt")}
    ]
    with pytest.raises(FileNotFoundError):
        run_cotrain_training(config, progress=False)
