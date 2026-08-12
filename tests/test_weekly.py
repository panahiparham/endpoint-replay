"""End-to-end tests for experiment.weekly's orchestration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from conftest import Sandbox, _seed_cluster_results, _seed_db, run_py, setup_cluster
from experiment import schedule, weekly  # noqa: E402
from experiment.core import _connect_write, _insert_run  # noqa: E402


def test_idle_when_not_due(tmp_path, monkeypatch):
    schedule.write_state(
        tmp_path / "state.json",
        schedule.BenchmarkState(last_sha="abc", last_run="2026-01-01T00:00:00Z"),
    )
    monkeypatch.setattr(weekly, "remote_sha", lambda **kw: "abc")

    status = weekly.check_status(
        label="bench_core", state_path=tmp_path / "state.json", repo_root=tmp_path,
    )

    assert status.status == weekly.STATUS_IDLE


def test_due_on_a_fresh_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly, "remote_sha", lambda **kw: "def4567")

    status = weekly.check_status(
        label="bench_core", state_path=tmp_path / "state.json", repo_root=tmp_path,
    )

    assert status.status == weekly.STATUS_DUE
    assert "def4567" in status.detail
    assert "never" in status.detail


def test_queued_while_a_dispatch_is_running(tmp_path, monkeypatch):
    (tmp_path / ".cluster").mkdir()
    (tmp_path / ".cluster" / "bench_core.json").write_text("{}")
    monkeypatch.setattr(weekly, "is_queued", lambda **kw: True)

    status = weekly.check_status(
        label="bench_core", state_path=tmp_path / "state.json", repo_root=tmp_path,
    )

    assert status.status == weekly.STATUS_QUEUED


def test_ready_to_finish_once_no_longer_queued(tmp_path, monkeypatch):
    (tmp_path / ".cluster").mkdir()
    (tmp_path / ".cluster" / "bench_core.json").write_text("{}")
    monkeypatch.setattr(weekly, "is_queued", lambda **kw: False)

    status = weekly.check_status(
        label="bench_core", state_path=tmp_path / "state.json", repo_root=tmp_path,
    )

    assert status.status == weekly.STATUS_READY_TO_FINISH


def test_unreachable_when_vulcan_cant_be_reached(tmp_path, monkeypatch):
    """An expired ssh session (SystemExit) is its own status, not "queued" -
    a real connection failure shouldn't read as "nothing to do"."""
    (tmp_path / ".cluster").mkdir()
    (tmp_path / ".cluster" / "bench_core.json").write_text("{}")

    def _boom(**kw):
        raise SystemExit("ssh refused")

    monkeypatch.setattr(weekly, "is_queued", _boom)

    status = weekly.check_status(
        label="bench_core", state_path=tmp_path / "state.json", repo_root=tmp_path,
    )

    assert status.status == weekly.STATUS_UNREACHABLE
    assert "can't reach Vulcan" in status.detail


# --- dispatch (needs the GSP_LOCAL_MODE sandbox: slurm.dispatch's state write
# depends on repo_root(), so this only resolves correctly in a subprocess with
# the sandbox's own copy of experiment/ on PYTHONPATH - see conftest.py) -----

def _dispatch_proc(sandbox: Sandbox, num_workers: int = 2):
    return subprocess.run(
        [sys.executable, "-c",
         "from experiment import weekly; weekly.dispatch("
         "label='toy', run_py='experiments/toy/run.py', "
         "results_dir='experiments/toy/results', "
         f"num_workers={num_workers})"],
        cwd=str(sandbox.repo), env={**os.environ, **sandbox.env},
        capture_output=True, text=True,
    )


def test_dispatch_wipes_prior_results_and_submits_a_fresh_sweep(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")  # a stale prior run
    _seed_cluster_results(sandbox, ["stale-cluster"])
    _seed_db(sandbox.results_dir / "comp.db", ["stale-local"])

    proc = _dispatch_proc(sandbox)

    assert proc.returncode == 0, proc.stderr
    assert not sandbox.results_dir.exists(), "local results must be wiped"
    # dispatch's own remote provisioning recreates results/<label> (its new
    # snapshot's results/ symlinks to it) - what must be gone is the stale db.
    assert not (sandbox.root / "results" / "toy" / "comp.db").exists(), \
        "stale remote results must not survive a wipe"
    assert sandbox.sbatch_calls(), "must have submitted a fresh sweep"


# --- finish (sync/render_markdown/write_state have no repo_root() dependency,
# so this runs in-process against the sandbox via monkeypatched env vars) ----

def test_finish_syncs_renders_and_records_state(sandbox: Sandbox, monkeypatch, tmp_path):
    from experiment import slurm

    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    slurm._ROOT_CACHE.clear()  # keyed on the literal "$HOME/..." string, stale otherwise

    T = 10
    curves = {"reward": [0.0] * (T - 1) + [5.0], "terminated": [0] * (T - 1) + [1],
              "truncated": [0] * T}
    conn = _connect_write(sandbox.root / "results" / "toy" / "comp.db")
    _insert_run(conn, {"run_id": "r0", "config_id": "c", "seed": 0}, {}, curves)
    conn.commit()
    conn.close()

    monkeypatch.setenv("HOME", str(sandbox.home))
    monkeypatch.setenv("GSP_LOCAL_MODE", "1")
    monkeypatch.setattr(weekly, "remote_sha", lambda **kw: "abc1234567")

    readme_dir = sandbox.repo / "experiments" / "toy"
    sha = weekly.finish(
        label="toy", component_names=["comp"],
        run_py=sandbox.repo / "experiments" / "toy" / "run.py",
        results_dir=sandbox.results_dir, plots_dir=tmp_path / "plots",
        readme_dir=readme_dir, state_path=tmp_path / "state.json",
        repo_root=sandbox.repo, config_path=sandbox.repo / "cluster.toml",
    )

    assert sha == "abc1234567"
    assert (sandbox.results_dir / "comp.db").is_file()

    readme = (readme_dir / "README.md").read_text()
    assert "abc1234" in readme and "comp" in readme

    # Dispatch's own state write (a prior run_py sweep --slurm above) must be
    # cleared, or check_status() would report ready_to_finish forever.
    assert not (sandbox.repo / ".cluster" / "toy.json").exists()

    state = schedule.read_state(tmp_path / "state.json")
    assert state.last_sha == "abc1234567"


# --- open_pr (real local git against a bare "origin"; only gh is stubbed) --

GH_STUB = """\
#!/usr/bin/env bash
echo "$@" >> "$GH_CALLS"
echo "https://github.com/fake/online-gsp/pull/1"
"""


def _repo_with_origin(tmp_path: Path) -> Path:
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    repo = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", str(bare), str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t.t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)

    (repo / "benchmarks" / "core").mkdir(parents=True)
    (repo / "benchmarks" / "core" / "README.md").write_text("placeholder\n")
    (repo / "benchmarks" / "state.json").write_text("{}")
    (repo / "benchmark_plots").mkdir()
    (repo / "benchmark_plots" / "x.png").write_bytes(b"fake")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "main"], check=True)
    return repo


def test_open_pr_branches_commits_pushes_and_creates_pr(tmp_path, monkeypatch):
    repo = _repo_with_origin(tmp_path)
    (repo / "benchmarks" / "core" / "README.md").write_text("updated\n")  # this run's report

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(GH_STUB)
    (bin_dir / "gh").chmod(0o755)
    gh_calls = tmp_path / "gh_calls.txt"
    monkeypatch.setenv("GH_CALLS", str(gh_calls))
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    url = weekly.open_pr(
        label="bench_core", readme_path="benchmarks/core/README.md",
        repo_root=repo, run_date="2026-08-17",
    )

    assert url == "https://github.com/fake/online-gsp/pull/1"
    branch = subprocess.run(["git", "-C", str(repo), "branch", "--show-current"],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert branch == "chore/weekly-benchmark-2026-08-17"
    subject = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%s"],
                             capture_output=True, text=True, check=True).stdout.strip()
    assert subject == "data: weekly benchmark run 2026-08-17"
    assert "pr create" in gh_calls.read_text()


def test_open_pr_survives_a_stale_branch_of_the_same_name(tmp_path, monkeypatch):
    """Reproduces the real failure: check_status stuck reporting
    ready_to_finish re-ran finish, and open_pr's old `checkout -b` crashed on
    a branch left over from an earlier, already-merged run of the same day."""
    repo = _repo_with_origin(tmp_path)
    subprocess.run(["git", "-C", str(repo), "branch",
                    "chore/weekly-benchmark-2026-08-17"], check=True)
    # This run's fresh report (finish() already wrote it, on main).
    (repo / "benchmarks" / "core" / "README.md").write_text("fresh content\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(GH_STUB)
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("GH_CALLS", str(tmp_path / "gh_calls.txt"))
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    url = weekly.open_pr(  # must not raise
        label="bench_core", readme_path="benchmarks/core/README.md",
        repo_root=repo, run_date="2026-08-17",
    )

    assert url == "https://github.com/fake/online-gsp/pull/1"
    readme = subprocess.run(["git", "-C", str(repo), "show", "HEAD:benchmarks/core/README.md"],
                            capture_output=True, text=True, check=True).stdout
    assert readme == "fresh content\n"
