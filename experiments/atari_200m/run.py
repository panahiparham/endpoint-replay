"""Run the ``atari_200m`` experiment (components: dqn_pong, random_pong).

Thin wrapper: hands ``main`` and this experiment's components to the shared harness
with per-component ``vmappable=False`` (Atari's ale-py FFI can't be vmapped, so seeds
run one per process). Each component saves to its own ``results/<name>.db`` store.
Requires the Atari extra - see ``scripts/install_ale_wheel.sh``.

    uv run python experiments/atari_200m/run.py plan
    uv run python experiments/atari_200m/run.py sweep --num-workers 2   # 2 components x
    1 seed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
).strip()

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import run_experiment  # noqa: E402
from main import ExperimentConfig, build  # noqa: E402

from config import COMPONENTS, LABEL, RESULTS_DIR  # noqa: E402


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
