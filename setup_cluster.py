"""Bootstrap the SLURM cluster for running experiments.

Creates the bare repo on the cluster, installs uv, detects the SLURM account,
and builds both the cpu and gpu venvs. Safe to re-run (idempotent). Must be
re-run after a scratch purge or major uv.lock changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap sys.path to import experiment from src/.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    """Parse args and invoke cluster setup."""
    parser = argparse.ArgumentParser(
        description="Bootstrap the SLURM cluster for running experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "cluster.toml",
        help="cluster/slurm configuration file",
    )
    args = parser.parse_args()

    from experiment.slurm import setup

    setup(config_path=args.config)


if __name__ == "__main__":
    main()
