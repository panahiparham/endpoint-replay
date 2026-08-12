"""Weekly benchmark due-check: state persistence and the pure due-ness rule.

``benchmarks/state.json`` (committed) records the sha and timestamp of the last
completed benchmark run. :func:`is_due` compares that against ``origin/main``'s
current sha and the elapsed time, so the check itself never touches the
cluster - :func:`remote_sha` reads GitHub over ``git ls-remote``, never Vulcan,
so it needs no MFA.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

__all__ = ["BenchmarkState", "read_state", "write_state", "remote_sha", "is_due"]


@dataclasses.dataclass(frozen=True)
class BenchmarkState:
    """The last completed benchmark run, as recorded in ``state.json``."""

    last_sha: str
    last_run: str  # UTC ISO 8601, e.g. "2026-08-03T12:00:00Z"


def read_state(path: str | Path) -> BenchmarkState | None:
    """Read a benchmark's state file.

    Args:
        path: The ``state.json`` to read.

    Returns:
        The recorded state, or ``None`` if the file doesn't exist yet (the
        benchmark has never completed a run).
    """
    path = Path(path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return BenchmarkState(last_sha=data["last_sha"], last_run=data["last_run"])


def write_state(path: str | Path, state: BenchmarkState) -> None:
    """Write a benchmark's state file, creating its parent directory.

    Args:
        path: The ``state.json`` to write.
        state: The state to record.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(state), indent=2) + "\n")


def remote_sha(
    remote: str = "origin", branch: str = "main", *, cwd: str | Path | None = None
) -> str:
    """The current sha a remote branch points to, without fetching.

    Args:
        remote: The git remote to query.
        branch: The branch to look up.
        cwd: The repo to run ``git`` in, defaulting to the current directory.

    Returns:
        The sha ``remote``'s ``branch`` currently points to.

    Raises:
        RuntimeError: If the remote has no such branch.
    """
    proc = subprocess.run(
        ["git", "ls-remote", remote, f"refs/heads/{branch}"],
        capture_output=True, text=True, cwd=cwd, check=True,
    )
    line = proc.stdout.strip()
    if not line:
        raise RuntimeError(f"{remote} has no branch {branch!r}")
    return line.split()[0]


def is_due(
    state: BenchmarkState | None,
    current_sha: str,
    now: datetime,
    *,
    min_days: int = 7,
) -> bool:
    """Whether a new benchmark run is due.

    Args:
        state: The last completed run, or ``None`` if it has never run.
        current_sha: ``origin/main``'s current sha (:func:`remote_sha`).
        now: The current time, timezone-aware.
        min_days: Minimum days required between two runs.

    Returns:
        ``True`` if this is the first run, or if ``current_sha`` differs from
        the last run's sha *and* at least ``min_days`` have passed since it.
    """
    if state is None:
        return True
    if current_sha == state.last_sha:
        return False
    last_run = datetime.fromisoformat(state.last_run.replace("Z", "+00:00"))
    return now - last_run >= timedelta(days=min_days)
