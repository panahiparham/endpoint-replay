"""End-to-end tests for the SLURM workflow - without a cluster.

See ``conftest.py`` for the ``sandbox`` fixture this relies on. What is under
test here is the wiring: which commit gets snapshotted, which venv and
PYTHONPATH the jobs are told to use, what sbatch is actually asked for, when a
venv is rebuilt, and that synced databases merge into local ones instead of
replacing them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import (
    Sandbox,
    _commit,
    _git,
    _run_ids,
    _seed_cluster_results,
    _seed_db,
    run_py,
    setup_cluster,
)

# --- setup (requirement 1) ---------------------------------------------------------


def test_setup_creates_the_bare_repo_and_both_venvs(sandbox: Sandbox):
    setup_cluster(sandbox)

    assert (sandbox.root / "endpoint-replay.git" / "HEAD").exists(), "no bare repo"
    for name in ("cpu", "gpu"):
        venv = sandbox.root / "envs" / name / ".venv" / "bin" / "python"
        assert venv.exists(), f"{name} venv was not built"
        assert (sandbox.root / "envs" / name / "lock.sha256").read_text().strip()

    # Exactly two venvs, and the gpu one carries the cuda extra.
    assert sorted(p.name for p in (sandbox.root / "envs").iterdir() if p.is_dir()) == \
        ["cpu", "gpu"]
    syncs = sandbox.uv_syncs()
    assert len(syncs) == 2
    assert any("--extra cuda" in s for s in syncs)
    assert all("--no-install-project" in s for s in syncs), \
        "the venvs must stay deps-only"

    url = _git(sandbox.repo, "remote", "get-url", "cluster-testcluster")
    assert url == str(sandbox.root / "endpoint-replay.git")
    # The scratch snapshot used to build the venvs is cleaned up.
    assert not (sandbox.root / "runs" / ".setup").exists()


def test_setup_is_idempotent_and_does_not_resync(sandbox: Sandbox):
    setup_cluster(sandbox)
    setup_cluster(sandbox)
    assert len(sandbox.uv_syncs()) == 2, "an unchanged lock must not rebuild the venvs"


def test_setup_rejects_a_dirty_tree(sandbox: Sandbox):
    (sandbox.repo / "marker.txt").write_text("uncommitted")
    proc = setup_cluster(sandbox, expect_ok=False)
    assert proc.returncode != 0
    assert "working tree is dirty" in proc.stderr


# --- dispatch (requirement 2) ------------------------------------------------------


def test_sweep_dispatch_snapshots_the_commit_and_queues_array_plus_consolidate(
    sandbox: Sandbox,
):
    setup_cluster(sandbox)
    sha = _git(sandbox.repo, "rev-parse", "HEAD")

    run_py(sandbox, "sweep", "--num-workers", "3", "--slurm")

    (rundir,) = sandbox.run_dirs
    assert rundir.name.startswith("toy_") and rundir.name.endswith(sha[:7])
    assert (rundir / "marker.txt").read_text() == "v1", "snapshot is not the commit"

    # Results are redirected to the experiment's shared cluster directory, which is what
    # makes sync one rsync and lets runs accumulate across commits.
    link = rundir / "experiments" / "toy" / "results"
    assert link.is_symlink()
    assert link.resolve() == (sandbox.root / "results" / "toy").resolve()

    array, consolidate = sandbox.sbatch_calls()
    assert "--array=0-2" in array
    assert "--account=def-test" in array
    assert "--time=01:00:00" in array
    assert "--mem-per-cpu=4G" in array
    assert not any(a.startswith("--gpus") for a in array), \
        "gpus = 0 must not ask for GPUs"
    assert "--dependency=afterok:1001" in consolidate

    wrap = sandbox.wrap_of(array)
    assert f"cd {rundir}" in wrap
    assert f"PYTHONPATH={rundir}/src" in wrap, "the snapshot's own src must win"
    assert f"{sandbox.root}/envs/cpu/.venv/bin/python" in wrap
    assert "experiments/toy/run.py sweep --num-workers 3" in wrap
    # Must reach the compute node unexpanded, for sbatch to substitute per task.
    assert "--worker-index $SLURM_ARRAY_TASK_ID" in wrap
    assert "experiments/toy/run.py consolidate" in sandbox.wrap_of(consolidate)

    state = sandbox.state()
    assert state["sha"] == sha
    assert state["venv"] == "cpu"
    assert state["jobs"] == {"array": "1001", "consolidate": "1002"}


def test_dispatch_rejects_a_dirty_tree(sandbox: Sandbox):
    setup_cluster(sandbox)
    (sandbox.repo / "marker.txt").write_text("uncommitted")

    proc = run_py(sandbox, "sweep", "--num-workers", "2", "--slurm", expect_ok=False)

    assert proc.returncode != 0
    assert "working tree is dirty" in proc.stderr
    assert sandbox.run_dirs == []
    assert sandbox.sbatch_calls() == []


def test_sweep_requires_num_workers(sandbox: Sandbox):
    setup_cluster(sandbox)
    proc = run_py(sandbox, "sweep", "--slurm", expect_ok=False)
    assert "--num-workers" in proc.stderr


def test_passthrough_args_reach_both_jobs(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--component", "comp",
           "--shard-size", "1", "--slurm")

    array, consolidate = sandbox.sbatch_calls()
    array_wrap = sandbox.wrap_of(array)
    assert "--component comp" in array_wrap
    assert "--shard-size 1" in array_wrap
    # The consolidate job must merge the same components the sweep wrote.
    assert "--component comp" in sandbox.wrap_of(consolidate)


def test_single_dispatches_one_job(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "single", "--seeds", "0", "--slurm")

    (call,) = sandbox.sbatch_calls()
    assert "--job-name=single" in call
    assert not any(a.startswith("--array") for a in call)
    assert "run.py single --seeds 0" in sandbox.wrap_of(call)
    assert sandbox.state()["jobs"] == {"single": "1001"}


def test_gpu_experiment_uses_the_gpu_venv_and_asks_for_gpus(sandbox: Sandbox):
    """Per-experiment overrides in cluster.toml select resources and the venv."""
    setup_cluster(sandbox)
    # Relabel the toy experiment so [experiments.gpu_toy] applies to it.
    run_py_path = sandbox.repo / "experiments" / "toy" / "run.py"
    run_py_path.write_text(
        run_py_path.read_text().replace('label="toy"', 'label="gpu_toy"')
    )
    _commit(sandbox.repo, "gpu label")

    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")

    (array, consolidate) = sandbox.sbatch_calls()
    assert "--gpus-per-node=2" in array
    assert "--time=12:00:00" in array, "per-experiment override must beat the default"
    assert f"{sandbox.root}/envs/gpu/.venv/bin/python" in sandbox.wrap_of(array)

    # Merging sqlite parts uses no GPU, so the dependent job must not queue for one.
    assert not any(a.startswith("--gpus") for a in consolidate)
    assert "--time=12:00:00" in consolidate, "other resources are still inherited"


def test_per_experiment_account_override_beats_the_default(sandbox: Sandbox):
    """[experiments.<label>] account overrides [cluster] account for that label only."""
    setup_cluster(sandbox)
    run_py_path = sandbox.repo / "experiments" / "toy" / "run.py"
    run_py_path.write_text(
        run_py_path.read_text().replace('label="toy"', 'label="acct_toy"')
    )
    _commit(sandbox.repo, "acct label")

    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")

    array, consolidate = sandbox.sbatch_calls()
    assert "--account=acct-test" in array
    assert "--account=acct-test" in consolidate
    assert "--account=def-test" not in array


def test_dry_run_submits_nothing_and_leaves_no_run_dir(sandbox: Sandbox):
    setup_cluster(sandbox)
    proc = run_py(sandbox, "sweep", "--num-workers", "3", "--slurm-dry-run")

    assert sandbox.sbatch_calls() == []
    assert sandbox.run_dirs == []
    assert not (sandbox.repo / ".cluster" / "toy.json").exists()
    assert "--array=0-2" in proc.stdout, "the sbatch commands should still be shown"
    assert "--dependency=afterok:" in proc.stdout


def test_plan_with_slurm_stays_local_and_reports_resources(sandbox: Sandbox):
    setup_cluster(sandbox)
    before = sandbox.sbatch_calls()

    proc = run_py(sandbox, "plan", "--slurm")

    assert "2 shards" in proc.stdout, "plan output must be unchanged"
    assert "venv cpu" in proc.stdout
    assert sandbox.sbatch_calls() == before, "plan must not submit anything"
    assert sandbox.run_dirs == [], "plan must not snapshot anything"


def test_venv_is_rebuilt_only_when_the_lock_changes(sandbox: Sandbox):
    setup_cluster(sandbox)
    assert len(sandbox.uv_syncs()) == 2

    (sandbox.repo / "marker.txt").write_text("v2")
    _commit(sandbox.repo, "code only")
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    assert len(sandbox.uv_syncs()) == 2, "a code change must not rebuild a venv"

    (sandbox.repo / "uv.lock").write_text("version = 1\nchanged = true\n")
    _commit(sandbox.repo, "bump deps")
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    assert len(sandbox.uv_syncs()) == 3, "a lock change must re-sync before running"

    # Each snapshot still carries its own source.
    assert {(d / "marker.txt").read_text() for d in sandbox.run_dirs} == {"v2"}


# --- sync (requirement 3) ----------------------------------------------------------


def test_sync_merges_cluster_results_without_overwriting_local_ones(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1", "cluster-2"])
    _seed_db(sandbox.results_dir / "comp.db", ["local-0"])

    run_py(sandbox, "sync")

    assert _run_ids(sandbox.results_dir / "comp.db") == {
        "local-0", "cluster-1", "cluster-2",
    }
    assert not list((sandbox.results_dir / "comp.parts").glob("*.db")), \
        "consolidate removes the parts it merged"


def test_sync_creates_the_local_db_when_there_is_none(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1"])

    run_py(sandbox, "sync")

    assert _run_ids(sandbox.results_dir / "comp.db") == {"cluster-1"}


def test_sync_is_idempotent(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1", "cluster-2"])

    run_py(sandbox, "sync")
    run_py(sandbox, "sync")

    assert _run_ids(sandbox.results_dir / "comp.db") == {"cluster-1", "cluster-2"}


def test_sync_needs_no_run_id_and_spans_commits(sandbox: Sandbox):
    """Results are per-experiment on the cluster, so one sync covers every commit."""
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["from-commit-1"])

    (sandbox.repo / "marker.txt").write_text("v2")
    _commit(sandbox.repo, "second commit")
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    # The second snapshot writes into the same shared dir.
    _seed_db(sandbox.root / "results" / "toy" / "comp.db", ["from-commit-2"])

    run_py(sandbox, "sync")

    assert _run_ids(sandbox.results_dir / "comp.db") == {
        "from-commit-1", "from-commit-2",
    }


def test_sync_warns_when_a_sweep_is_still_in_flight(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1"])
    (sandbox.root / "results" / "toy" / "comp.parts").mkdir()

    proc = run_py(sandbox, "sync")

    assert "still in flight" in proc.stderr
    assert _run_ids(sandbox.results_dir / "comp.db") == {"cluster-1"}


def test_sync_errors_before_anything_has_run(sandbox: Sandbox):
    setup_cluster(sandbox)
    proc = run_py(sandbox, "sync", expect_ok=False)
    assert "nothing on the cluster" in proc.stderr


# --- wipe --------------------------------------------------------------------------

def test_wipe_removes_the_remote_results_dir(sandbox: Sandbox, monkeypatch):
    from experiment import slurm

    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1"])
    monkeypatch.setenv("HOME", str(sandbox.home))
    monkeypatch.setenv("GSP_LOCAL_MODE", "1")
    slurm._ROOT_CACHE.clear()  # keyed on the literal "$HOME/..." string, stale otherwise

    slurm.wipe(label="toy", config_path=sandbox.repo / "cluster.toml")

    assert not (sandbox.root / "results" / "toy").exists()


def test_wipe_is_a_no_op_when_nothing_is_there(sandbox: Sandbox, monkeypatch):
    from experiment import slurm

    setup_cluster(sandbox)
    monkeypatch.setenv("HOME", str(sandbox.home))
    monkeypatch.setenv("GSP_LOCAL_MODE", "1")
    slurm._ROOT_CACHE.clear()

    slurm.wipe(label="toy", config_path=sandbox.repo / "cluster.toml")  # must not raise


# --- is_queued -----------------------------------------------------------------------

def _is_queued_proc(sandbox: Sandbox, env: dict | None = None):
    return subprocess.run(
        [sys.executable, "-c",
         "from experiment import slurm; print(slurm.is_queued(label='toy'))"],
        cwd=str(sandbox.repo), env={**os.environ, **(env or sandbox.env)},
        capture_output=True, text=True,
    )


def test_is_queued_is_false_once_nothing_is_in_squeue(sandbox: Sandbox):
    """QUIET_STUB's squeue always reports nothing queued or running."""
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")

    proc = _is_queued_proc(sandbox)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_is_queued_is_true_while_squeue_reports_a_job(sandbox: Sandbox, tmp_path: Path):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    busy_bin = tmp_path / "busybin"
    busy_bin.mkdir()
    (busy_bin / "squeue").write_text(
        "#!/usr/bin/env bash\necho '1001 sweep RUNNING 0:01 node1'\n"
    )
    (busy_bin / "squeue").chmod(0o755)
    env = {**sandbox.env, "PATH": f"{busy_bin}:{sandbox.env['PATH']}"}

    proc = _is_queued_proc(sandbox, env)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True"


def test_is_queued_requires_a_dispatch_first(sandbox: Sandbox):
    setup_cluster(sandbox)

    proc = _is_queued_proc(sandbox)

    assert proc.returncode != 0
    assert "nothing dispatched yet" in proc.stderr


# --- status and logs ---------------------------------------------------------------


def test_status_reports_the_recorded_jobs(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")

    proc = run_py(sandbox, "status")

    assert "1001,1002" in proc.stdout
    assert "squeue" in proc.stdout and "sacct" in proc.stdout


def test_logs_fetches_the_run_directory_output(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    (rundir,) = sandbox.run_dirs
    (rundir / "logs" / "sweep-1001_0.out").write_text("task 0 output\n")

    listing = run_py(sandbox, "logs")
    assert "sweep-1001_0.out" in listing.stdout

    shown = run_py(sandbox, "logs", "0")
    assert "task 0 output" in shown.stdout


def test_an_expired_ssh_session_says_so(sandbox: Sandbox, monkeypatch, tmp_path: Path):
    """Alliance MFA cannot be answered by a subprocess, so an expired ControlMaster
    surfaces on whatever remote call runs first. That must read as "log in again", not
    as a failure of that particular command."""
    from experiment import slurm

    ssh = tmp_path / "authbin" / "ssh"
    ssh.parent.mkdir()
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'parham1@vulcan.alliancecan.ca: Permission denied '"
        "'(keyboard-interactive).' >&2\n"
        "exit 255\n"
    )
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{ssh.parent}:{os.environ['PATH']}")
    monkeypatch.delenv("GSP_LOCAL_MODE", raising=False)   # exercise the real ssh path

    cfg = slurm.load_config(sandbox.repo / "cluster.toml")
    with pytest.raises(SystemExit) as excinfo:
        slurm._ssh(cfg, "printf hello")

    message = str(excinfo.value)
    assert "ssh to testcluster was refused" in message
    assert "ssh testcluster true" in message, "must name the fix, not just the symptom"
    assert "printf hello" not in message, \
        "must not blame the command that happened to run"


def test_cluster_modes_require_a_dispatch_first(sandbox: Sandbox):
    setup_cluster(sandbox)
    for mode in ("status", "logs"):
        proc = run_py(sandbox, mode, expect_ok=False)
        assert "nothing dispatched yet" in proc.stderr
