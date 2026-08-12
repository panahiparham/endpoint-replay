"""End-to-end tests for the benchmark report renderer (experiment.report)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from experiment import report  # noqa: E402
from experiment.core import _connect_write, _insert_run  # noqa: E402

T = 20


def _seed_component(db_path: Path, returns: list[float]) -> None:
    """A fake component: one seed per return, each a single-episode run."""
    conn = _connect_write(db_path)
    for seed, ret in enumerate(returns):
        curves = {
            "reward": np.zeros(T), "terminated": np.zeros(T), "truncated": np.zeros(T),
        }
        curves["terminated"][-1] = 1
        curves["reward"][-1] = ret
        _insert_run(conn, {"run_id": f"r{seed}", "config_id": "c", "seed": seed},
                    {}, curves)
    conn.commit()
    conn.close()


def test_plot_component_saves_a_png(tmp_path):
    _seed_component(tmp_path / "comp.db", [10.0, 11.0, 12.0])
    plots_dir = tmp_path / "plots"

    path = report.plot_component(tmp_path / "comp.db", "comp", plots_dir)

    assert path == plots_dir / "comp.png"
    assert path.is_file()


def test_plot_component_returns_none_without_runs(tmp_path):
    _connect_write(tmp_path / "empty.db").close()
    assert report.plot_component(tmp_path / "empty.db", "empty", tmp_path) is None


def test_render_markdown_reports_mean_ci_and_plot_link(tmp_path):
    _seed_component(tmp_path / "results" / "comp.db", [10.0, 11.0, 12.0])
    readme_dir = tmp_path / "benchmarks" / "core"
    readme_dir.mkdir(parents=True)

    md = report.render_markdown(
        "bench_core", ["comp"], tmp_path / "results", tmp_path / "plots",
        readme_dir, sha="abc1234567", run_date="2026-08-10",
    )

    assert "commit `abc1234`" in md
    assert "| comp | 11.0 | [10.0, 12.0] | 3 |" in md
    assert "![comp](../../plots/comp.png)" in md


def test_render_markdown_placeholder_row_for_an_empty_component(tmp_path):
    _connect_write(tmp_path / "results" / "empty.db").close()

    md = report.render_markdown(
        "bench_core", ["empty"], tmp_path / "results", tmp_path / "plots",
        tmp_path, sha="abc1234567", run_date="2026-08-10",
    )

    assert "| empty | - | - | 0 |" in md
    assert "![empty]" not in md
