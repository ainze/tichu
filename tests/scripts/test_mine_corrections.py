"""CLI guards for scripts/mine_corrections.py.

The corpus builder's logic lives in `tichu_training.correction.corpus` and is
tested there. What only the script can get wrong is the pool wiring: `round_idx`
indexes a REGENERATED Starting-Position Pool, so a `--rounds` / `--pool-seed`
that does not match the tier-1 sweep verifies every Decision against a different
deal and produces a corpus of confident nonsense.
"""

import pandas as pd
import pytest

from scripts.mine_corrections import main


def _tier1(path, *, max_round):
    pd.DataFrame([
        dict(round_idx=max_round, turn=3, delta=40.0, alt="A", chosen="X"),
        dict(round_idx=0, turn=1, delta=20.0, alt="B", chosen="Y"),
    ]).to_parquet(path, index=False)
    return str(path)


def test_refuses_a_pool_too_small_for_the_tier1_sweep(tmp_path):
    """A tier-1 row the regenerated pool cannot address is proof the two runs are
    not the same sweep. Failing loudly here is the only defence — a mismatched
    SEED at the same size is silent, which is why the message says so."""
    tier1 = _tier1(tmp_path / "tier1.parquet", max_round=1499)
    with pytest.raises(SystemExit) as excinfo:
        main(["--tier1", tier1, "--export-dir", str(tmp_path),
              "--out-dir", str(tmp_path / "out"), "--rounds", "1000"])
    assert "not the same sweep" in str(excinfo.value)


def test_runs_from_a_worktree_without_pythonpath(tmp_path):
    """Regression for a real crash. The editable install resolves `tichu_training`
    to the MAIN checkout, so a worktree run without PYTHONPATH imports this
    worktree's `scripts/` against another tree's `src/` — missing modules at best,
    and silently stale ones at worst (a `blunder_miner` without `candidate_seed`
    reverts tier-2 seeding to the salted hash). The script must resolve its own
    src, the way `diag_pikl` and friends already do."""
    import os
    import subprocess
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tier1 = _tier1(tmp_path / "tier1.parquet", max_round=1499)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.mine_corrections", "--tier1", tier1,
         "--export-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"),
         "--rounds", "1000"],
        capture_output=True, text=True, env=env, cwd=root,
    )
    combined = proc.stdout + proc.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert "not the same sweep" in combined, combined
