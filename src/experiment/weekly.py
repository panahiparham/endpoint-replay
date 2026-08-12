"""Weekly-benchmark orchestration: check/dispatch/finish, no chat mediation.

Three questions, checked in this order:

1. Is a dispatch already in flight (``.cluster/<label>.json`` exists)? If so,
   are its jobs still queued, or done?
2. If nothing is in flight, is a fresh run due (:func:`experiment.schedule.is_due`)?
3. Otherwise, idle.

:func:`check_status` answers all three with one :class:`Status`. The actions
a human explicitly triggers from a benchmark's ``weekly.py`` CLI - the Vulcan
MFA step happens before :func:`dispatch`, a PR review happens after - live
here too, reusing :mod:`experiment.schedule`, :mod:`experiment.slurm` and
:mod:`experiment.report`.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiment.report import render_markdown
from experiment.schedule import BenchmarkState, is_due, read_state, remote_sha, write_state
from experiment.slurm import is_queued
from experiment.slurm import dispatch as _slurm_dispatch
from experiment.slurm import sync as _slurm_sync
from experiment.slurm import wipe as _slurm_wipe

__all__ = [
    "STATUS_IDLE", "STATUS_DUE", "STATUS_QUEUED", "STATUS_READY_TO_FINISH",
    "STATUS_UNREACHABLE", "Status", "check_status", "dispatch", "finish", "open_pr",
]

STATUS_IDLE = "idle"
STATUS_DUE = "due"
STATUS_QUEUED = "queued"
STATUS_READY_TO_FINISH = "ready_to_finish"
STATUS_UNREACHABLE = "unreachable"


@dataclasses.dataclass(frozen=True)
class Status:
    """The answer to "what, if anything, should happen next?"."""

    status: str  # one of the STATUS_* constants
    detail: str  # a human-readable reason, for a CLI to print


def check_status(
    *, label: str, state_path: str | Path, repo_root: str | Path,
    config_path: str | Path | None = None,
) -> Status:
    """Whether a dispatch is in flight, or a fresh run is due.

    Args:
        label: The benchmark's experiment label (e.g. ``"bench_core"``).
        state_path: The benchmark's ``state.json`` (last completed run).
        repo_root: The repo root, to find ``.cluster/<label>.json``.
        config_path: The ``cluster.toml`` to read, defaulting to the repo
            root's.

    Returns:
        ``STATUS_UNREACHABLE`` if a dispatch is in flight but Vulcan can't be
        reached (e.g. an expired ssh ControlMaster) - distinct from
        ``STATUS_QUEUED``, since a real connection failure is worth surfacing
        rather than reading as "nothing to do" alongside a legitimately
        still-running sweep. Else ``STATUS_QUEUED``/``STATUS_READY_TO_FINISH``
        if a dispatch is in flight and reachable; ``STATUS_DUE``/
        ``STATUS_IDLE`` from the due-check otherwise.
    """
    repo_root = Path(repo_root)
    if (repo_root / ".cluster" / f"{label}.json").is_file():
        try:
            queued = is_queued(label=label, config_path=config_path)
        except SystemExit as exc:
            return Status(STATUS_UNREACHABLE, f"can't reach Vulcan: {exc}")
        if queued:
            return Status(STATUS_QUEUED, "jobs still in squeue")
        return Status(STATUS_READY_TO_FINISH, "jobs no longer queued - run finish")

    state = read_state(state_path)
    sha = remote_sha(cwd=repo_root)
    if is_due(state, sha, datetime.now(timezone.utc)):
        last = f"{state.last_sha[:7]} on {state.last_run}" if state else "never"
        return Status(STATUS_DUE, f"origin/main is {sha[:7]}, last benchmarked {last}")
    return Status(STATUS_IDLE, "nothing to do")


def dispatch(
    *, label: str, run_py: str | Path, results_dir: str | Path, num_workers: int,
    config_path: str | Path | None = None,
) -> None:
    """Wipe prior results and dispatch a fresh sweep.

    The caller's job to have already gotten a human go-ahead and to have an
    authenticated Vulcan session open (a CLI's ``--yes`` flag and its own
    "did you run ``ssh vulcan true``?" prompt) - this raises the same
    ``SystemExit`` ``slurm.dispatch`` always does if the tree is dirty or the
    ssh session has expired.

    Args:
        label: The benchmark's experiment label.
        run_py: The benchmark's ``run.py``.
        results_dir: The benchmark's local results dir, wiped before dispatch
            (dedup is by ``run_id`` regardless of commit, so a rerun with
            unwiped results would just skip everything).
        num_workers: Passed through as ``--num-workers``.
        config_path: The ``cluster.toml`` to read, defaulting to the repo
            root's.
    """
    shutil.rmtree(results_dir, ignore_errors=True)
    _slurm_wipe(label=label, config_path=config_path)
    _slurm_dispatch(label=label, run_py=Path(run_py), mode="sweep",
                    argv=["--num-workers", str(num_workers)], config_path=config_path)


def finish(
    *, label: str, component_names: list[str], run_py: str | Path,
    results_dir: str | Path, plots_dir: str | Path, readme_dir: str | Path,
    state_path: str | Path, repo_root: str | Path = Path("."),
    config_path: str | Path | None = None,
) -> str:
    """Sync results, render the report, and record the new state.

    Args:
        label: The benchmark's experiment label.
        component_names: The components to report on, in table order.
        run_py: The benchmark's ``run.py``, used to consolidate synced parts.
        results_dir: Where each component's ``<name>.db`` lives.
        plots_dir: Directory to save each component's learning-curve PNG into.
        readme_dir: Where the rendered ``README.md`` is written.
        state_path: The benchmark's ``state.json``.
        repo_root: The repo root, for ``git ls-remote`` (defaults to the
            current directory, since a CLI always runs from the repo root).
        config_path: The ``cluster.toml`` to read, defaulting to the repo
            root's.

    Returns:
        The commit sha this run benchmarked.
    """
    results_dir, readme_dir = Path(results_dir), Path(readme_dir)
    _slurm_sync(label=label, results_dir=results_dir, run_py=Path(run_py),
               config_path=config_path)

    sha = remote_sha(cwd=repo_root)
    run_date = datetime.now(timezone.utc).date().isoformat()
    md = render_markdown(label, component_names, results_dir, Path(plots_dir),
                         readme_dir, sha=sha, run_date=run_date)
    (readme_dir / "README.md").write_text(md)

    write_state(state_path,
               BenchmarkState(last_sha=sha, last_run=datetime.now(timezone.utc).isoformat()))

    # The dispatch this run processed is done - without this, check_status()
    # would find .cluster/<label>.json still there and report ready_to_finish
    # forever, re-running finish every tick indefinitely.
    dispatch_state = Path(repo_root) / ".cluster" / f"{label}.json"
    dispatch_state.unlink(missing_ok=True)

    return sha


def _git(repo_root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo_root, check=True,
                   capture_output=True, text=True)


def open_pr(
    *, label: str, readme_path: str | Path, repo_root: str | Path,
    run_date: str | None = None,
) -> str:
    """Branch, commit the rendered report + state, push, and open a PR.

    Args:
        label: The benchmark's experiment label, named in the PR body.
        readme_path: The rendered ``README.md``, relative to ``repo_root``
            (e.g. ``"benchmarks/core/README.md"``) - committed alongside
            ``benchmarks/state.json`` and ``benchmark_plots/``.
        repo_root: The repo root, where ``git``/``gh`` run.
        run_date: The run's date, defaulting to today (UTC).

    Returns:
        The created PR's URL.
    """
    repo_root = Path(repo_root)
    run_date = run_date or datetime.now(timezone.utc).date().isoformat()
    branch = f"chore/weekly-benchmark-{run_date}"

    # -B (not -b) off a freshly-fetched origin/main: idempotent regardless of
    # what's currently checked out or whether this branch already exists
    # locally (e.g. a prior run of this same day that didn't get this far).
    _git(repo_root, "fetch", "origin", "main")
    _git(repo_root, "checkout", "-B", branch, "origin/main")
    _git(repo_root, "add", str(readme_path), "benchmarks/state.json", "benchmark_plots")
    _git(repo_root, "commit", "-m", f"data: weekly benchmark run {run_date}")
    _git(repo_root, "push", "--force", "-u", "origin", branch)

    proc = subprocess.run(
        ["gh", "pr", "create", "--title", f"Weekly benchmark: {run_date}",
         "--body", f"Automated run of {label}. See {readme_path} for results."],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()
