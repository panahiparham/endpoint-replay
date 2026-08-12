"""
CLI: check/dispatch/finish the weekly benchmark - no chat session needed.

Cheat sheet:
    Check:    uv run benchmarks/core/weekly.py check
    Dispatch: uv run benchmarks/core/weekly.py dispatch --yes
    Finish:   uv run benchmarks/core/weekly.py finish

`check`'s exit code is what a cron job or a human branches on:
    0  - idle, or a dispatch is still queued: nothing to do
    10 - a fresh run is due: run `ssh vulcan true`, then `dispatch --yes`
    20 - a dispatch finished: run `finish`
    30 - a dispatch is in flight but Vulcan can't be reached right now
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import weekly

from config import COMPONENTS, LABEL, RESULTS_DIR

STATE_PATH = _REPO_ROOT / "benchmarks" / "state.json"
PLOTS_DIR = _REPO_ROOT / "benchmark_plots"
README_DIR = Path(__file__).resolve().parent
RUN_PY = Path(__file__).resolve().parent / "run.py"

_EXIT_BY_STATUS = {
    weekly.STATUS_IDLE: 0,
    weekly.STATUS_QUEUED: 0,
    weekly.STATUS_DUE: 10,
    weekly.STATUS_READY_TO_FINISH: 20,
    weekly.STATUS_UNREACHABLE: 30,
}


def _check() -> int:
    status = weekly.check_status(label=LABEL, state_path=STATE_PATH, repo_root=_REPO_ROOT)
    print(f"[{status.status}] {status.detail}")
    return _EXIT_BY_STATUS[status.status]


def _dispatch(num_workers: int) -> int:
    weekly.dispatch(label=LABEL, run_py=RUN_PY, results_dir=RESULTS_DIR,
                    num_workers=num_workers)
    print(f"[{LABEL}] dispatched - run `check` again once the jobs finish")
    return 0


def _finish() -> int:
    component_names = [c.name for c in COMPONENTS]
    sha = weekly.finish(label=LABEL, component_names=component_names, run_py=RUN_PY,
                        results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR,
                        readme_dir=README_DIR, state_path=STATE_PATH, repo_root=_REPO_ROOT)
    readme_path = (README_DIR / "README.md").relative_to(_REPO_ROOT)
    url = weekly.open_pr(label=LABEL, readme_path=readme_path, repo_root=_REPO_ROOT)
    print(f"[{LABEL}] benchmarked {sha[:7]}, opened {url}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="weekly.py")
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("check")
    dispatch_ap = sub.add_parser("dispatch")
    dispatch_ap.add_argument(
        "--yes", action="store_true",
        help="confirm dispatching - the human go-ahead this design gates on. "
             "Requires an authenticated Vulcan session (`ssh vulcan true` first).",
    )
    dispatch_ap.add_argument("--num-workers", type=int, default=len(COMPONENTS))
    sub.add_parser("finish")
    args = ap.parse_args()

    if args.mode == "check":
        return _check()
    if args.mode == "dispatch":
        if not args.yes:
            print("refusing to dispatch without --yes (and an authenticated "
                  "`ssh vulcan true` session)", file=sys.stderr)
            return 1
        return _dispatch(args.num_workers)
    return _finish()


if __name__ == "__main__":
    sys.exit(main())
