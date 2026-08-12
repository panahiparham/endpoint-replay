"""Markdown report rendering for a benchmark run.

Turns a benchmark experiment's stored results into a markdown report: a
summary table (mean +/- 95% bootstrap CI over
:func:`experiment.plotting.weighted_lifetime_return`) and a learning-curve plot
per component. Plots are saved wherever the caller points ``plots_dir`` - point
it at a *committed* directory, not one literally named ``plots/``
(``**/plots/`` is gitignored; see the repo's ``benchmark_plots/`` convention).

Generic over any benchmark experiment (not hardcoded to ``benchmarks/core``),
so a future benchmark folder reuses it rather than duplicating this logic.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment.core import load_curves, load_runs
from experiment.plotting import (
    bootstrap_mean_ci,
    plot_mean_ci,
    seed_grids_for,
    style,
    weighted_lifetime_return,
)

__all__ = ["plot_component", "render_markdown"]


def plot_component(
    db_path: Path, name: str, plots_dir: Path, *, grid_points: int = 200
) -> Path | None:
    """Save one component's learning curve (mean +/- 95% bootstrap CI).

    Args:
        db_path: The component's consolidated ``<name>.db``.
        name: The component's name, used for the plot's filename and title.
        plots_dir: Directory to save the PNG into (created if missing).
        grid_points: Timesteps the learning curve is interpolated onto.

    Returns:
        The saved PNG's path, or ``None`` if the component has no runs yet.
    """
    df = load_runs(db_path, write_csv=False)
    if df.is_empty():
        return None
    max_t = max(len(load_curves(db_path, rid)["reward"]) for rid in df["run_id"])
    grid = np.linspace(0, max_t, grid_points)
    stack = seed_grids_for(db_path, grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_mean_ci(ax, grid, stack, name, "C0")
    style(ax)
    ax.set_title(name)

    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _summary_row(db_path: Path, name: str) -> str:
    """One markdown table row: mean +/- 95% CI weighted-lifetime return."""
    df = load_runs(db_path, write_csv=False)
    if df.is_empty():
        return f"| {name} | - | - | 0 |"
    values = np.array([
        weighted_lifetime_return(*(
            load_curves(db_path, rid)[k] for k in ("reward", "terminated", "truncated")
        ))
        for rid in df["run_id"]
    ])
    mean, lo, hi = (arr[0] for arr in bootstrap_mean_ci(values.reshape(-1, 1)))
    return f"| {name} | {mean:.1f} | [{lo:.1f}, {hi:.1f}] | {len(df)} |"


def render_markdown(
    label: str,
    component_names: list[str],
    results_dir: Path,
    plots_dir: Path,
    readme_dir: Path,
    sha: str,
    run_date: str,
) -> str:
    """Render a benchmark run's report.

    Args:
        label: The benchmark's ``LABEL`` (e.g. ``"bench_core"``).
        component_names: The components to report on, in table order.
        results_dir: Where each component's ``<name>.db`` lives.
        plots_dir: Directory to save each component's learning-curve PNG into.
        readme_dir: Where the returned markdown will be written (e.g.
            ``benchmarks/core/``), used to compute each plot's relative link.
        sha: The commit this run benchmarked.
        run_date: The run's date, e.g. ``"2026-08-10"``.

    Returns:
        The report body, ready to write to a benchmark's ``README.md``.
    """
    lines = [
        f"# {label}",
        "",
        f"Weekly benchmark run - {run_date}, commit `{sha[:7]}`.",
        "",
        "| Component | Mean lifetime return | 95% CI | Seeds |",
        "| --- | --- | --- | --- |",
    ]
    for name in component_names:
        lines.append(_summary_row(results_dir / f"{name}.db", name))

    lines.append("")
    for name in component_names:
        png = plot_component(results_dir / f"{name}.db", name, plots_dir)
        if png is None:
            continue
        rel = Path(os.path.relpath(png, readme_dir)).as_posix()
        lines += [f"## {name}", "", f"![{name}]({rel})", ""]
    return "\n".join(lines) + "\n"
