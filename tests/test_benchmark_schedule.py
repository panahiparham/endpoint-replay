"""End-to-end tests for the weekly benchmark due-check (experiment.schedule)."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from experiment import schedule  # noqa: E402

NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)


def _run(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)


def _bare_origin(tmp_path: Path) -> tuple[Path, str]:
    """A throwaway bare repo standing in for "origin", with one commit on main."""
    bare = tmp_path / "origin.git"
    _run(["git", "init", "--quiet", "--bare", "-b", "main", str(bare)], tmp_path)
    work = tmp_path / "work"
    _run(["git", "clone", "--quiet", str(bare), str(work)], tmp_path)
    (work / "f.txt").write_text("hi")
    _run(["git", "add", "f.txt"], work)
    _run(["git", "-c", "user.email=t@t.t", "-c", "user.name=t",
          "commit", "--quiet", "-m", "init"], work)
    _run(["git", "push", "--quiet", "origin", "main"], work)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work,
                         capture_output=True, text=True, check=True).stdout.strip()
    return bare, sha


def test_remote_sha_reads_origin_head(tmp_path):
    bare, sha = _bare_origin(tmp_path)
    assert schedule.remote_sha(str(bare), "main", cwd=tmp_path) == sha


def test_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    state = schedule.BenchmarkState(last_sha="abc123", last_run="2026-01-01T00:00:00Z")
    schedule.write_state(path, state)
    assert schedule.read_state(path) == state


def test_read_state_missing_file_returns_none(tmp_path):
    assert schedule.read_state(tmp_path / "missing.json") is None


def test_is_due_first_run_has_no_state():
    assert schedule.is_due(None, "sha", NOW) is True


def test_is_due_same_sha_is_never_due():
    state = schedule.BenchmarkState(last_sha="sha", last_run="2000-01-01T00:00:00Z")
    assert schedule.is_due(state, "sha", NOW) is False


def test_is_due_changed_sha_but_too_soon():
    last_run = (NOW - timedelta(days=3)).isoformat().replace("+00:00", "Z")
    state = schedule.BenchmarkState(last_sha="old", last_run=last_run)
    assert schedule.is_due(state, "new", NOW) is False


def test_is_due_changed_sha_and_enough_days():
    last_run = (NOW - timedelta(days=8)).isoformat().replace("+00:00", "Z")
    state = schedule.BenchmarkState(last_sha="old", last_run=last_run)
    assert schedule.is_due(state, "new", NOW) is True
