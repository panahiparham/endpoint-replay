"""End-to-end tests for the weekly benchmark suite's definition (benchmarks/core)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_BENCH_CORE = _REPO / "benchmarks" / "core"

sys.path.insert(0, str(_BENCH_CORE))
import config as bench_core_config  # noqa: E402


def test_components_are_five_dqn_ones_with_ten_seeds_each():
    names = {c.name for c in bench_core_config.COMPONENTS}
    assert names == {
        "dqn_acrobot", "dqn_mountaincar", "dqn_cartpole",
        "dqn_pinball_easy", "dqn_catch",
    }
    for c in bench_core_config.COMPONENTS:
        assert c.seeds == list(range(10))
        assert c.base.AGENT == "dqn"


def test_plan_reports_five_components_and_fifty_runs():
    """Not asserting on pending/done: this is a real host's actual local
    results/, not a fresh checkout, so it may already have some runs done."""
    proc = subprocess.run(
        [sys.executable, str(_BENCH_CORE / "run.py"), "plan"],
        cwd=str(_REPO), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "total: 50 runs across 5 component(s)" in proc.stdout
