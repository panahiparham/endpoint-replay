"""Lightweight, framework-free RL experiment infrastructure.

See :mod:`experiment.core` for the full docstring. Each folder under
``experiments/`` supplies a declarative ``config.py`` (a list of named
:class:`Component` s in ``COMPONENTS``, each a ``base`` config + ``sweep`` +
``seeds`` + ``shard_size``) and a thin ``run.py`` that hands ``main`` (the
interaction loop in ``main.py``) and those components to :func:`run_experiment`.
Plotting helpers live in :mod:`experiment.plotting` (kept separate so this
package stays import-light - it imports matplotlib).
"""

from experiment.core import (
    Component,
    Shard,
    build_global_shards,
    build_shards,
    config_id,
    consolidate,
    expand_grid,
    load_curve,
    load_curves,
    load_runs,
    pending_runs,
    run_experiment,
    run_global_shards,
    run_id,
    run_ids_with_curves,
    run_shards,
    sweep_points,
)

__all__ = [
    "expand_grid",
    "sweep_points",
    "config_id",
    "run_id",
    "Shard",
    "build_shards",
    "Component",
    "build_global_shards",
    "pending_runs",
    "consolidate",
    "run_shards",
    "run_global_shards",
    "run_experiment",
    "load_runs",
    "load_curves",
    "load_curve",
    "run_ids_with_curves",
]
