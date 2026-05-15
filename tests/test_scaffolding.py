import importlib
import subprocess
import sys


def test_tichu_engine_importable():
    import tichu_engine  # noqa: F401


def test_tichu_ml_importable():
    import tichu_ml  # noqa: F401


def test_tichu_training_importable():
    import tichu_training  # noqa: F401


def test_tichu_inference_importable():
    import tichu_inference  # noqa: F401


_CLI_COMMANDS = [
    "parse_bsw",
    "compute_trueskill",
    "train_bc",
    "train_calls",
    "train_belief",
    "eval_matrix",
    "serve_inference",
]


def test_tichu_engine_has_no_ml_imports():
    # Run in a subprocess so we can inspect sys.modules after import without
    # contamination from whatever the test runner itself has loaded.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tichu_engine, sys; "
            "ml_libs = [m for m in sys.modules if m.split('.')[0] in ('torch', 'numpy', 'tensorflow', 'jax')]; "
            "print(ml_libs)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    loaded = eval(result.stdout.strip())
    assert loaded == [], f"tichu_engine pulled in ML libraries: {loaded}"


def test_cli_entry_points_exit_zero_on_help():
    for cmd in _CLI_COMMANDS:
        result = subprocess.run(
            [sys.executable, "-m", f"tichu_training.cli.{cmd.replace('-', '_')}" if cmd != "serve_inference"
             else f"tichu_inference.cli.serve", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"`{cmd} --help` exited {result.returncode}:\n{result.stderr}"
        )
