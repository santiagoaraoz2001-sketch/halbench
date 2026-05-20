"""Smoke tests for the CLI — argparse wiring + --version."""
import subprocess
import sys


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "halbench", *args],
        capture_output=True, text=True, timeout=20,
    )


def test_cli_version():
    r = _run_cli("--version")
    assert r.returncode == 0
    assert "halbench" in r.stdout.lower()
    assert "2.2.0" in r.stdout


def test_cli_help():
    r = _run_cli("--help")
    assert r.returncode == 0
    # All four subcommands should be advertised
    for cmd in ("run", "score", "verify", "submit"):
        assert cmd in r.stdout


def test_cli_submit_smoke():
    """submit just prints instructions — no file IO. Pass a fake path."""
    r = _run_cli("submit", "/tmp/does_not_exist.jsonl", "--model-id", "test/model")
    assert r.returncode == 0
    assert "leaderboard/submissions" in r.stdout
    assert "test__model" in r.stdout
