"""
Run: DQN on Classic Control environments (MountainCar, Cartpole, Acrobot).

Cheat sheet:
    Plan: uv run experiments/classic_control/run.py plan
    Sweep: uv run experiments/classic_control/run.py sweep --num-workers 6
    Single: uv run experiments/classic_control/run.py single --component dqn_cartpole
    Merge Results (sweep does it for you): uv run experiments/classic_control/run.py
    consolidate
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force single-threaded CPU XLA BEFORE the first jax import (below, via main), so
# N local worker processes don't each grab every core and thrash.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
).strip()

# Bootstrap sys.path: repo root (for `import main`) and this dir (for `import config`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import run_experiment
from main import ExperimentConfig, build

from config import COMPONENTS, LABEL, RESULTS_DIR


def entry() -> None:
    run_experiment(
        build_fn=build,
        config_cls=ExperimentConfig,
        components=COMPONENTS,
        results_dir=RESULTS_DIR,
        label=LABEL,
    )


if __name__ == "__main__":
    entry()
