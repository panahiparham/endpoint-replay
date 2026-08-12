"""Shared, framework-free helpers for defining, running, and collecting RL
experiments. Every experiment stays small: a declarative ``config.py`` (a list of
named :class:`Component` s in ``COMPONENTS``) and a thin ``run.py`` that hands
``main`` to this harness.

Components
----------
An experiment bundles one or more *components*, each a :class:`Component` with its
own ``base`` config, ``sweep``, ``seeds``, ``shard_size`` and ``vmappable`` flag.
Each component's runs are stored in their own ``<results_dir>/<name>.db`` store (with
transient per-worker parts in a sibling ``<name>.parts/`` dir during a sweep), so
sibling components (e.g. ``dqn_easy`` and ``random_easy``) are collected and analysed
independently. A sweep pools every component's shards into one round-robin worker
pool (:func:`build_global_shards`), so a single ``--num-workers`` / SLURM array runs
the whole experiment.

Conventions
-----------
* A *config* is a (possibly nested) ``@dataclass`` instance - e.g. ``ExperimentConfig``
  in ``main.py``, composing an agent config and an env config. Its fields are the
  hyperparameters.
* A *sweep* is a plain ``dict`` mapping a (possibly dotted) field path to a list
  of values, e.g. ``{"ENV_HYPERS.SETTING": ["empty", "box"]}``. It is expanded to the
  Cartesian product of its lists (:func:`expand_grid`) and each combination is
  applied to a ``base`` config (:func:`sweep_points`), yielding config objects.
* A *run* is one row in a per-experiment SQLite database: its exact
  hyperparameters (``config_json``) and any training curves (``curves`` npz blob).
  ``metrics_json`` is left empty - scalar summaries are derived from the curves in
  analysis.

Reproducibility / identity
--------------------------
The seed is a first-class axis. ``config_id`` is a content hash of the config
(``dataclasses.asdict``) with the seed *excluded*, and
``run_id = "<config_id>_s<seed>"``. A run's PRNG is derived from its integer seed
alone (``jax.random.key(seed)``), so results are reproducible regardless of how
work is sharded across workers.

The interaction loop
--------------------
A single function ``main(config, rng) -> pytree`` (in ``main.py``) runs the agent
in the environment. The harness loops over configs and ``jax.vmap``s ``main`` over
a config's seed keys - one vmap call per shard. ``config`` is a concrete,
closed-over Python object, so building the env (host-side numpy/file parsing)
happens at trace time; only the seed is a tracer. The batched result is
decomposed per seed and its ``metrics`` dict stored as curves.

Concurrency
-----------
A shared SQLite file is not safe under concurrent writes, so each worker writes
only its own ``<name>.parts/part-<worker_index>.db``; :func:`consolidate` merges the
parts into the consolidated ``<name>.db`` at the sweep's completion barrier. Reads
(dedup + load) always union ``<name>.db`` with every part, so a run is skipped whether
it lives in the consolidated db or an un-consolidated part.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import itertools
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

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


# --- sweep expansion --------------------------------------------------------

def expand_grid(sweep: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a ``{path: [values...]}`` sweep into the list of every combination.

    >>> expand_grid({"a": [1, 2], "b": ["x"]})
    [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'x'}]
    """
    keys = list(sweep.keys())
    value_lists = [list(sweep[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def _set_path(obj: Any, path: str, value: Any) -> Any:
    """Return a copy of the dataclass ``obj`` with dotted ``path`` set to ``value``.

    ``_set_path(cfg, "ENV_HYPERS.SETTING", "empty")`` -> a new ``cfg`` whose nested
    ``ENV_HYPERS.SETTING`` is ``"empty"`` (via recursive :func:`dataclasses.replace`).
    """
    head, _, rest = path.partition(".")
    if rest:
        return dataclasses.replace(
            obj, **{head: _set_path(getattr(obj, head), rest, value)}
        )
    return dataclasses.replace(obj, **{head: value})


def sweep_points(base: Any, sweep: dict[str, list]) -> list[Any]:
    """Apply every grid combination of a sweep to a base config.

    Args:
        base: The config object each combination is applied to.
        sweep: A ``{dotted_path: [values...]}`` mapping.

    Returns:
        One config object per combination, of the same type as ``base``. An
        empty sweep yields ``[base]``.
    """
    points: list[Any] = []
    for combo in expand_grid(sweep):
        cfg = base
        for path, value in combo.items():
            cfg = _set_path(cfg, path, value)
        points.append(cfg)
    return points


# --- stable identity --------------------------------------------------------

def _canonical(hypers: dict[str, Any]) -> str:
    """Deterministic JSON for a hyper dict: sorted keys, floats rounded to 12
    significant digits (so float noise doesn't change the id). Recurses into
    nested dicts (e.g. ``{"AGENT_HYPERS": {...}, "ENV_HYPERS": {...}}``)."""

    def clean(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(f"{float(v):.12g}")
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {k: clean(v[k]) for k in sorted(v)}
        return v

    return json.dumps(clean(hypers), sort_keys=True)


def _as_point(config: Any) -> dict[str, Any]:
    """The identity/storage dict for a config: ``asdict`` for a dataclass, else
    the mapping itself."""
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return dict(config)


def config_id(config: Any) -> str:
    """Compute the stable id for a configuration.

    Args:
        config: A config object or its ``dataclasses.asdict`` mapping.

    Returns:
        An 8-character id, excluding the seed, so a config keeps one id across
        all of its seeds.
    """
    digest = hashlib.blake2b(_canonical(_as_point(config)).encode(), digest_size=4)
    return digest.hexdigest()


def run_id(config: Any, seed: int) -> str:
    """Compute the stable id for one configuration and seed.

    Args:
        config: A config object or its ``dataclasses.asdict`` mapping.
        seed: The run's integer seed.

    Returns:
        The id ``"<config_id>_s<seed>"``.
    """
    return f"{config_id(config)}_s{int(seed)}"


# --- shards: one config x a chunk of its seeds ------------------------------

@dataclasses.dataclass
class Shard:
    """A config paired with at most ``shard_size`` seeds. All the seeds of a
    shard run together in one ``jax.vmap`` call, so a shard is the unit of
    parallelism: one per local worker process / SLURM array task at a time."""

    config: Any             # a config object (e.g. ExperimentConfig)
    seeds: list[int]        # the subset of seeds evaluated in this shard


def build_shards(
    configs: list[Any],
    seeds: list[int],
    shard_size: int | None = None,
) -> list[Shard]:
    """Chunk each config's seeds into shards.

    Shards cover the full enumeration rather than the pending subset, in a fixed
    order, so membership is stable regardless of what has already run. That is
    what makes a round-robin assignment across workers disjoint and resume-safe.

    Args:
        configs: The configs to shard.
        seeds: The seeds to run for every config.
        shard_size: Maximum seeds per shard, or ``None`` for one shard per config.

    Returns:
        The shards, in ``for cfg ... for chunk ...`` order.
    """
    seeds = [int(s) for s in seeds]
    shards: list[Shard] = []
    for cfg in configs:
        step = shard_size if shard_size else (len(seeds) or 1)
        for i in range(0, len(seeds), step):
            shards.append(Shard(config=cfg, seeds=seeds[i:i + step]))
    return shards


def pending_runs(
    configs: list[Any],
    seeds: list[int],
    db_path: str | Path,
) -> list[tuple[Any, int]]:
    """Find the runs a store does not yet hold.

    Args:
        configs: The configs to check.
        seeds: The seeds to check for every config.
        db_path: The consolidated ``<name>.db``; its parts are read too.

    Returns:
        The ``(config, seed)`` pairs with no stored run.
    """
    done = _existing_run_ids(db_path)
    return [
        (cfg, int(s))
        for cfg in configs
        for s in seeds
        if run_id(cfg, int(s)) not in done
    ]


# --- components: named sub-experiments, each its own sweep + database ---------

@dataclasses.dataclass
class Component:
    """One named sub-experiment: its own ``base`` config, ``sweep``, ``seeds``,
    ``shard_size`` and ``vmappable`` flag. An experiment's ``config.py`` exports a
    list of these (``COMPONENTS``); each component's runs live in their own
    ``<results_dir>/<name>.db`` store, so components with different agents/envs
    (e.g. ``dqn_easy`` and ``random_easy``) are collected and analysed separately.

    ``vmappable`` defaults to ``True``, which is wrong for a stateful env, so
    construction rejects it for any ``ENV`` the registry marks non-vmappable (see
    :data:`environments.ENVIRONMENTS`). Without that check the mistake surfaces only
    at run time, as an error from inside the env's FFI that names no cause."""

    name: str
    base: Any
    sweep: dict[str, list] = dataclasses.field(default_factory=dict)
    seeds: list[int] = dataclasses.field(default_factory=list)
    shard_size: int | None = None
    vmappable: bool = True

    def __post_init__(self) -> None:
        if not self.vmappable:
            return
        # Every ENV this component can run: the base plus anything the sweep sets.
        envs = {getattr(self.base, "ENV", None), *self.sweep.get("ENV", [])} - {None}
        if not envs:
            return  # a config without an ENV field (e.g. the harness's own tests)
        from environments import ENVIRONMENTS  # local: keeps this module import-light

        # Unknown names are left alone; `main` raises its own error for those.
        bad = sorted(
            e for e in envs if e in ENVIRONMENTS and not ENVIRONMENTS[e].vmappable
        )
        if bad:
            raise ValueError(
                f"component {self.name!r} sets vmappable=True but env {bad[0]!r} is "
                "stateful and cannot run under jax.vmap: the agent's boundary-reset "
                "lax.cond gets a per-seed predicate, lowers to a select, and resets "
                "the env on every step. Set vmappable=False (and shard_size=1)."
            )


def build_global_shards(
    components: list[Component],
    *,
    shard_size_override: int | None = None,
) -> list[tuple[Component, Shard]]:
    """Pool every component's shards into one flat, deterministic list.

    A worker pool round-robins over the result: worker ``w`` of ``N`` runs
    ``pairs[w::N]``, so one ``--num-workers N`` or SLURM array spans all
    components while keeping the assignment disjoint and resume-safe.

    Args:
        components: The components to pool, kept in list order.
        shard_size_override: Replaces every component's own ``shard_size``
            when given (this is what ``--shard-size`` sets).

    Returns:
        ``(component, shard)`` pairs, components in list order and each
        component's shards in :func:`build_shards` order.
    """
    pairs: list[tuple[Component, Shard]] = []
    for comp in components:
        configs = sweep_points(comp.base, comp.sweep)
        size = (
            shard_size_override
            if shard_size_override is not None
            else comp.shard_size
        )
        for shard in build_shards(configs, comp.seeds, size):
            pairs.append((comp, shard))
    return pairs


# --- per-component SQLite results store -------------------------------------
#
# A store is a consolidated ``<name>.db`` file plus a sibling ``<name>.parts/`` dir of
# per-worker parts. One row per run. Each worker writes only its own
# <name>.parts/part-<k>.db (so nothing is ever shared-written), and consolidate()
# merges the parts into <name>.db at the completion barrier. Reads (dedup + load)
# union <name>.db + every part.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    config_id    TEXT NOT NULL,
    seed         INTEGER NOT NULL,
    config_json  TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    curves       BLOB
);
"""
_COLUMNS = "run_id, config_id, seed, config_json, metrics_json, curves"


def _parts_dir(db_path: str | Path) -> Path:
    """The directory holding a store's per-worker parts - a sibling of the
    consolidated db file (``results/<name>.db`` -> ``results/<name>.parts/``)."""
    db_path = Path(db_path)
    return db_path.with_name(db_path.stem + ".parts")


def _part_path(db_path: str | Path, worker_index: int) -> Path:
    """The per-worker database written during a sweep (merged by consolidate)."""
    return _parts_dir(db_path) / f"part-{int(worker_index)}.db"


def _part_paths(db_path: str | Path) -> list[Path]:
    parts = _parts_dir(db_path)
    return sorted(parts.glob("part-*.db")) if parts.is_dir() else []


def _db_paths(db_path: str | Path) -> list[Path]:
    """Every database holding runs: ``<name>.db`` then all of its parts."""
    db_path = Path(db_path)
    paths = [db_path] if db_path.exists() else []
    return paths + _part_paths(db_path)


def _connect_write(path: str | Path) -> sqlite3.Connection:
    """Open a database for writing, creating its parent dir and schema. Only one
    process ever writes a given file (a worker its own part, consolidate the
    consolidated db), so there is no cross-process write contention."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60.0)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.executescript(_SCHEMA)
    return conn


def _query_ro(path: str | Path, sql: str, params: tuple = ()) -> list[tuple]:
    """Run a read-only query, returning its rows. Returns ``[]`` if the database
    can't be read cleanly (e.g. a peer worker is mid-write) - a worker never needs
    a peer's rows (shards are disjoint), so a tolerated miss only risks a harmless
    recompute, never a wrong skip."""
    try:
        conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _existing_run_ids(db_path: str | Path) -> set[str]:
    """Global set of already-computed run_ids: the union of the consolidated db
    and every per-worker part. Independent of how work is partitioned, so
    extending a sweep (by hyper or seed) or changing ``--num-workers`` never
    recomputes an existing run - before or after consolidation."""
    done: set[str] = set()
    for path in _db_paths(db_path):
        done |= {r[0] for r in _query_ro(path, "SELECT run_id FROM runs")}
    return done


def _curves_to_blob(curves: dict[str, np.ndarray]) -> bytes | None:
    """Serialize a run's curve arrays to npz bytes (``None`` when there are none)."""
    if not curves:
        return None
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in curves.items()})
    return buf.getvalue()


def _blob_to_curves(blob: bytes | None) -> dict[str, np.ndarray]:
    if not blob:
        return {}
    with np.load(io.BytesIO(blob)) as data:
        return {k: data[k] for k in data.files}


def _insert_run(
    conn: sqlite3.Connection, record: dict, metrics: dict, curves: dict
) -> None:
    """Insert one run. ``INSERT OR IGNORE`` keyed on ``run_id`` makes re-touching an
    existing run a no-op, so reruns/extensions never overwrite or error."""
    conn.execute(
        f"INSERT OR IGNORE INTO runs ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
        (
            record["run_id"],
            record["config_id"],
            int(record["seed"]),
            json.dumps(record),
            json.dumps({k: float(v) for k, v in metrics.items()}),
            _curves_to_blob(curves),
        ),
    )


# --- runner: build a config's train fn host-side, then jit/vmap it ----------

# build(config) -> train, where train(rng) -> pytree with a "metrics" dict of
# per-timestep arrays. The env is built inside build() (host-side); see main.build.
Build = Callable[[Any], Callable[[Any], dict]]


def run_shards(
    build_fn: Build,
    configs: list[Any],
    seeds: list[int],
    db_path: str | Path,
    *,
    shard_size: int | None = None,
    worker_index: int = 0,
    num_workers: int = 1,
    max_shards: int | None = None,
    vmappable: bool = True,
) -> int:
    """Run this worker's share of a sweep, storing to one component's database.

    Seeds whose ``run_id`` is already stored are skipped, so an interrupted run
    recomputes only what is missing. The shard list is identical in every worker,
    so no run is skipped or double-claimed. Only the ``metrics`` dict of the
    result pytree is fetched to host; the rest (e.g. ``runner_state``) is not.

    Args:
        build_fn: Maps a config to its ``train`` function, building the env and
            agent host-side.
        configs: The configs to run.
        seeds: The seeds to run for every config.
        db_path: The consolidated ``<name>.db``. This worker writes only
            ``<name>.parts/part-<worker_index>.db`` beside it.
        shard_size: Maximum seeds per shard, or ``None`` for one shard per config.
        worker_index: This worker's index; it runs ``worker_index::num_workers``.
        num_workers: Total workers sharing the shard list.
        max_shards: Cap on how many of this worker's shards to run.
        vmappable: Run a shard's seeds together as ``jax.jit(jax.vmap(train))``.
            Set False for a stateful env (Atari) to run seeds one at a time, and
            pair it with ``shard_size=1`` plus process-level ``--num-workers``.

    Returns:
        The number of new runs saved.
    """
    db_path = Path(db_path)
    done = _existing_run_ids(db_path)
    mine = build_shards(configs, seeds, shard_size)[worker_index::num_workers]
    if max_shards is not None:
        mine = mine[:max_shards]
    return _run_shard_list(build_fn, mine, db_path,
                           worker_index=worker_index, vmappable=vmappable, done=done)


def _run_shard_list(
    build_fn: Build,
    shards: list[Shard],
    db_path: str | Path,
    *,
    worker_index: int = 0,
    vmappable: bool = True,
    done: set[str] | None = None,
) -> int:
    """Run an explicit list of shards to ``<name>.parts/part-<worker_index>.db``.

    The shared execution core behind :func:`run_shards` (which picks a worker's
    round-robin subset for one config list) and :func:`run_global_shards` (which
    picks a pooled subset spanning components). Seeds whose ``run_id`` is already in
    ``done`` (defaulting to everything already stored for ``db_path``) are
    skipped, so resume computes only the delta. Returns the number of new runs saved.
    jax is imported lazily so this module stays import-light for analysis."""
    import jax
    import jax.numpy as jnp

    db_path = Path(db_path)
    if done is None:
        done = _existing_run_ids(db_path)

    saved = 0
    conn = None
    try:
        for shard in shards:
            cfg = shard.config
            pend = [s for s in shard.seeds if run_id(cfg, s) not in done]
            if not pend:
                continue
            train = build_fn(cfg)  # build env + agent host-side (not under jit)
            if vmappable:
                keys = jax.vmap(jax.random.key)(jnp.asarray(pend))
                out = jax.jit(jax.vmap(train))(keys)
                mb = out["metrics"]  # dict of [S, T] arrays
                per_seed = [
                    {k: np.asarray(v[i]) for k, v in mb.items()}
                    for i in range(len(pend))
                ]
            else:  # non-vmappable env (e.g. Atari): jit train, run seeds one at a time
                run_one = jax.jit(train)
                per_seed = [
                    {
                        k: np.asarray(v)
                        for k, v in run_one(jax.random.key(int(s)))["metrics"].items()
                    }
                    for s in pend
                ]
            if conn is None:  # created lazily so a fully-cached worker writes nothing
                conn = _connect_write(_part_path(db_path, worker_index))
            point = _as_point(cfg)
            cid = config_id(cfg)
            for s, curves in zip(pend, per_seed):
                rid = run_id(cfg, s)
                record = {**point, "seed": int(s), "config_id": cid, "run_id": rid}
                _insert_run(conn, record, {}, curves)  # curves only; metrics empty
                done.add(rid)
            conn.commit()
            saved += len(pend)
    finally:
        if conn is not None:
            conn.close()
    return saved


def run_global_shards(
    build_fn: Build,
    components: list[Component],
    results_dir: str | Path,
    *,
    worker_index: int = 0,
    num_workers: int = 1,
    shard_size_override: int | None = None,
    max_shards: int | None = None,
) -> int:
    """Run this worker's share of a pooled sweep spanning several components.

    Shards come from :func:`build_global_shards` in a fixed, component-tagged
    order. This worker's shards are grouped by component and each group runs
    under that component's own ``vmappable`` flag.

    Args:
        build_fn: Maps a config to its ``train`` function.
        components: The components to pool.
        results_dir: Holds one ``<name>.db`` per component. This worker writes
            only ``<name>.parts/part-<worker_index>.db`` in each.
        worker_index: This worker's index; it runs ``pairs[worker_index::num_workers]``.
        num_workers: Total workers sharing the pooled shard list.
        shard_size_override: Replaces every component's own ``shard_size``.
        max_shards: Cap on how many of this worker's shards to run.

    Returns:
        The total number of new runs saved.
    """
    results_dir = Path(results_dir)
    mine = build_global_shards(components, shard_size_override=shard_size_override)
    mine = mine[worker_index::num_workers]
    if max_shards is not None:
        mine = mine[:max_shards]

    grouped: dict[str, tuple[Component, list[Shard]]] = {}
    order: list[str] = []
    for comp, shard in mine:
        if comp.name not in grouped:
            grouped[comp.name] = (comp, [])
            order.append(comp.name)
        grouped[comp.name][1].append(shard)

    saved = 0
    for name in order:
        comp, shards = grouped[name]
        saved += _run_shard_list(build_fn, shards, results_dir / f"{comp.name}.db",
                                 worker_index=worker_index, vmappable=comp.vmappable)
    return saved


def consolidate(db_path: str | Path) -> int:
    """Merge a store's per-worker parts into its consolidated database.

    Idempotent and single-process; run it at a sweep's completion barrier.

    Args:
        db_path: The consolidated ``<name>.db``. Its ``<name>.parts/part-*.db``
            are merged in and then deleted.

    Returns:
        The number of rows newly merged in.
    """
    db_path = Path(db_path)
    parts = _part_paths(db_path)
    if not parts:
        return 0
    conn = _connect_write(db_path)
    merged = 0
    try:
        for part in parts:
            before = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            conn.execute("ATTACH DATABASE ? AS part", (str(part),))
            conn.execute(f"INSERT OR IGNORE INTO runs SELECT {_COLUMNS} FROM part.runs")
            conn.commit()
            conn.execute("DETACH DATABASE part")
            merged += conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] - before
    finally:
        conn.close()
    for part in parts:  # only reached on a clean merge; a failure keeps the parts
        part.unlink()
    parts_dir = _parts_dir(db_path)
    if parts_dir.is_dir() and not any(parts_dir.iterdir()):
        parts_dir.rmdir()
    return merged


# --- read API (polars) ------------------------------------------------------

def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict to dotted keys, one column per hyper.

    ``{"ENV_HYPERS": {"SETTING": "box"}}`` becomes
    ``{"ENV_HYPERS.SETTING": "box"}``.
    """
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def load_runs(db_path: str | Path, *, write_csv: bool = True) -> pl.DataFrame:
    """Load every run in a component's store as one tidy DataFrame.

    Scalar metric columns are absent by design; derive them from
    :func:`load_curves` in analysis.

    Args:
        db_path: The consolidated ``<name>.db``, e.g.
            ``experiments/pinball/results/dqn_easy.db``. Un-consolidated
            per-worker parts are read too.
        write_csv: Also write ``<name>.csv`` beside the database.

    Returns:
        One row per run, ``run_id`` first, with ``config_json`` flattened to
        dotted columns such as ``ENV_HYPERS.SETTING``.
    """
    db_path = Path(db_path)
    seen: set[str] = set()
    rows: list[dict] = []
    for path in _db_paths(db_path):
        for rid, cfg_json, met_json in _query_ro(
                path, "SELECT run_id, config_json, metrics_json FROM runs"):
            if rid in seen:
                continue
            seen.add(rid)
            rec = _flatten(json.loads(cfg_json))
            rec.update(json.loads(met_json))
            rec.pop("run_id", None)
            rows.append({"run_id": rid, **rec})

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    df = df.select(["run_id", *[c for c in df.columns if c != "run_id"]])
    if write_csv:
        df.write_csv(db_path.with_suffix(".csv"))
    return df


def load_curves(db_path: str | Path, run_id: str) -> dict[str, np.ndarray]:
    """Load every curve array stored for one run.

    Args:
        db_path: The consolidated ``<name>.db``; its parts are searched too.
        run_id: The run to load.

    Returns:
        A ``{name: array}`` mapping, e.g. ``{"reward": ..., "terminated": ...}``,
        empty if the run stored no curves.
    """
    for path in _db_paths(db_path):
        rows = _query_ro(path, "SELECT curves FROM runs WHERE run_id=?", (run_id,))
        if rows and rows[0][0] is not None:
            return _blob_to_curves(rows[0][0])
    return {}


def load_curve(db_path: str | Path, run_id: str, name: str) -> np.ndarray | None:
    """Load one named curve array for a run.

    Args:
        db_path: The consolidated ``<name>.db``; its parts are searched too.
        run_id: The run to load.
        name: The curve to load, e.g. ``"reward"``.

    Returns:
        The array, or ``None`` if the run has no such curve.
    """
    return load_curves(db_path, run_id).get(name)


def run_ids_with_curves(db_path: str | Path) -> list[str]:
    """List the runs in a store that have curve arrays.

    Args:
        db_path: The consolidated ``<name>.db``; its parts are read too.

    Returns:
        The matching run ids, sorted.
    """
    ids: set[str] = set()
    for path in _db_paths(db_path):
        ids |= {r[0] for r in _query_ro(
            path, "SELECT run_id FROM runs WHERE curves IS NOT NULL")}
    return sorted(ids)


# --- CLI shared by every experiment's run.py --------------------------------

def _launch_local_workers(
    *, argv0, num_workers, shard_size, max_shards, components=None
):
    """Spawn ``num_workers`` child processes, each one worker of the sweep. Every
    child re-invokes this run.py (so it inherits the single-threaded-CPU XLA env),
    with a distinct ``--worker-index``. Children round-robin over the same pooled
    global shard list, so they write disjoint shards. Raises if any child exits
    non-zero."""
    import subprocess
    import sys

    cmd = [sys.executable, argv0, "sweep", "--num-workers", str(num_workers)]
    if shard_size is not None:
        cmd += ["--shard-size", str(shard_size)]
    if max_shards is not None:
        cmd += ["--max-shards", str(max_shards)]
    if components:
        cmd += ["--component", *components]
    procs = [
        subprocess.Popen(cmd + ["--worker-index", str(k)])
        for k in range(num_workers)
    ]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker(s) failed with exit codes {codes}")


@dataclass(frozen=True)
class _SlurmFlags:
    """The ``--slurm*`` options, stripped out of a mode's own argv."""

    enabled: bool = False
    dry_run: bool = False
    config: str | None = None


def _split_slurm_flags(argv: list[str]) -> tuple[list[str], _SlurmFlags]:
    """Split ``argv`` into (everything else, the slurm flags).

    Hand-rolled rather than argparse so an experiment's own options - which differ per
    mode and include tyro passthrough in ``single`` - can never be consumed, reordered
    or prefix-abbreviated on their way through.
    """
    rest: list[str] = []
    enabled = dry_run = False
    config: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--slurm":
            enabled = True
        elif arg == "--slurm-dry-run":
            enabled = dry_run = True          # implies --slurm; both is redundant
        elif arg == "--slurm-config":
            i += 1
            if i >= len(argv):
                raise SystemExit("--slurm-config needs a path")
            config = argv[i]
        elif arg.startswith("--slurm-config="):
            config = arg.split("=", 1)[1]
        else:
            rest.append(arg)
        i += 1
    return rest, _SlurmFlags(enabled=enabled, dry_run=dry_run, config=config)


def run_experiment(
    *,
    build_fn: Build,
    config_cls,
    components: list[Component],
    results_dir: str | Path,
    label: str,
    argv: list[str] | None = None,
) -> None:
    """Run the CLI shared by every experiment's run.py.

    The modes are ``single``, ``sweep``, ``plan``, ``consolidate``, ``sync``,
    ``status`` and ``logs``.

    Adding ``--slurm`` to any of the first four runs that same work on the cluster
    instead of here, so the workflow is one command and one flag either way::

        run.py sweep --num-workers 18            # this machine
        run.py sweep --num-workers 18 --slurm    # a SLURM array

    ``sync`` (pull this experiment's cluster results into ``results_dir``), ``status``
    and ``logs`` only ever concern the cluster; they accept ``--slurm`` but do not need
    it. All of it is implemented in :mod:`experiment.slurm`, imported lazily so a local
    run never pays for it.

    An experiment is a list of :class:`Component` s (its ``config.py`` ``COMPONENTS``);
    each component carries its own ``base``, ``sweep``, ``seeds``, ``shard_size`` and
    ``vmappable`` flag, and its runs live in their own ``<results_dir>/<name>.db``
    store. ``build_fn(config) -> train`` builds the env + agent host-side and
    returns the training function (``build`` in ``main.py``); ``config_cls`` is the
    config dataclass (for ``single``-mode tyro parsing).

    ``sweep`` pools every component's shards into one round-robin worker pool
    (:func:`build_global_shards`): with ``--worker-index k`` this process is one
    worker; without it and ``--num-workers > 1`` this process *launches* that many
    single-threaded child workers locally, then auto-consolidates every component.
    A ``--component NAME [NAME ...]`` filter restricts any mode to a subset; a
    ``--shard-size`` override replaces every selected component's own shard size.

    Args:
        build_fn: Maps a config to its ``train`` function (``build`` in main.py).
        config_cls: The config dataclass, for ``single``-mode tyro parsing.
        components: The experiment's ``COMPONENTS``.
        results_dir: Where each component's ``<name>.db`` store lives.
        label: The experiment's name, used for its cluster config section.
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.
    """
    import argparse
    import sys

    import tyro

    argv = sys.argv[1:] if argv is None else list(argv)
    modes = ("single", "sweep", "plan", "consolidate", "sync", "status", "logs")
    if not argv or argv[0] not in modes:
        given = argv[0] if argv else "<none>"
        raise SystemExit(
            f"[{label}] a mode is required as the first argument "
            f"(one of: {', '.join(modes)}); got {given!r}"
        )
    mode = argv[0]
    rest, slurm_flags = _split_slurm_flags(argv[1:])
    results_dir = Path(results_dir)
    by_name = {c.name: c for c in components}
    run_py = Path(sys.argv[0]).resolve()

    if mode in ("sync", "status", "logs"):
        from experiment import slurm

        if mode == "sync":
            slurm.sync(label=label, results_dir=results_dir, run_py=run_py,
                       config_path=slurm_flags.config)
        elif mode == "status":
            slurm.status(label=label, config_path=slurm_flags.config)
        else:
            slurm.logs(label=label, config_path=slurm_flags.config,
                       task=rest[0] if rest else None)
        return

    # `plan` is metadata-only and identical wherever it runs, so --slurm answers it
    # here (instantly) and just adds the resources the job would ask for. Every other
    # mode is real work and goes to the cluster.
    if slurm_flags.enabled and mode != "plan":
        from experiment import slurm

        slurm.dispatch(label=label, run_py=run_py, mode=mode, argv=rest,
                       config_path=slurm_flags.config, dry_run=slurm_flags.dry_run)
        return

    def _select(names) -> list[Component]:
        """Components picked by ``--component`` (all when unset), preserving order."""
        if not names:
            return list(components)
        missing = [n for n in names if n not in by_name]
        if missing:
            raise SystemExit(f"[{label}] unknown component(s) {missing}; "
                             f"defined: {sorted(by_name)}")
        return [by_name[n] for n in names]

    def _db(comp: Component) -> Path:
        return results_dir / f"{comp.name}.db"

    if mode == "consolidate":
        ap = argparse.ArgumentParser(prog="run.py consolidate", add_help=False)
        ap.add_argument("--component", nargs="+", default=None)
        pargs, _ = ap.parse_known_args(rest)
        total = 0
        for comp in _select(pargs.component):
            n = consolidate(_db(comp))
            total += n
            print(f"[{label}:{comp.name}] consolidated {n} run(s) into {_db(comp)}")
        return

    if mode == "plan":
        ap = argparse.ArgumentParser(prog="run.py plan", add_help=False)
        ap.add_argument("--shard-size", type=int, default=None)
        ap.add_argument("--component", nargs="+", default=None)
        pargs, _ = ap.parse_known_args(rest)
        sel = _select(pargs.component)
        tot_runs = tot_shards = tot_pending = 0
        for comp in sel:
            configs = sweep_points(comp.base, comp.sweep)
            size = pargs.shard_size if pargs.shard_size is not None else comp.shard_size
            shards = build_shards(configs, comp.seeds, size)
            pend = pending_runs(configs, comp.seeds, _db(comp))
            runs = len(configs) * len(comp.seeds)
            tot_runs += runs
            tot_shards += len(shards)
            tot_pending += len(pend)
            print(f"[{comp.name}] {len(configs)} configs x {len(comp.seeds)} "
                  f"seeds = {runs} runs; "
                  f"{len(shards)} shards (shard_size={size}); "
                  f"pending {len(pend)}, done {runs - len(pend)}")
        print(f"total: {tot_runs} runs across {len(sel)} component(s); "
              f"{tot_shards} shards "
              f"-> use up to --num-workers {tot_shards}; "
              f"pending {tot_pending}, done {tot_runs - tot_pending}")
        if slurm_flags.enabled:
            from experiment import slurm

            slurm_cfg = slurm.load_config(slurm_flags.config)
            res = slurm.resources_for(slurm_cfg, label)
            asks = " ".join(slurm.resource_flags(res)) or "(no resource flags set)"
            account = res.get("account", slurm_cfg.account)
            print(f"slurm: each task would ask for {asks}; venv {slurm.venv_name(res)}; "
                  f"account {account}")
        return

    if mode == "sweep":
        ap = argparse.ArgumentParser(prog="run.py sweep")
        ap.add_argument("--shard-size", type=int, default=None)
        ap.add_argument("--num-workers", type=int, default=1)
        ap.add_argument("--worker-index", type=int, default=None)
        ap.add_argument("--max-shards", type=int, default=None)
        ap.add_argument("--component", nargs="+", default=None)
        args = ap.parse_args(rest)
        sel = _select(args.component)

        if args.worker_index is None and args.num_workers > 1:
            # This process launches the workers and waits on them -> it owns the
            # whole suite, so it consolidates every component once children finish.
            _launch_local_workers(argv0=sys.argv[0], num_workers=args.num_workers,
                                  shard_size=args.shard_size,
                                  max_shards=args.max_shards,
                                  components=(
                                      [c.name for c in sel]
                                      if args.component else None
                                  ))
            merged = sum(consolidate(_db(c)) for c in sel)
            print(f"[{label}] launched {args.num_workers} local workers; consolidated "
                  f"{merged} run(s) across {len(sel)} component(s)")
            return

        n = run_global_shards(build_fn, sel, results_dir,
                              worker_index=args.worker_index or 0,
                              num_workers=args.num_workers,
                              shard_size_override=args.shard_size,
                              max_shards=args.max_shards)
        # A single-owner run consolidates itself; a lone worker among many leaves
        # its parts for the launcher's consolidate step.
        if args.num_workers <= 1:
            for comp in sel:
                consolidate(_db(comp))
        print(f"[{label}] saved {n} new run(s) across {len(sel)} component(s)")
        return

    # single: run one component's config (with tyro typed overrides) across --seeds.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--component", default=None)
    args, cfg_argv = ap.parse_known_args(rest)
    if args.component is not None:
        if args.component not in by_name:
            raise SystemExit(f"[{label}] unknown component {args.component!r}; "
                             f"defined: {sorted(by_name)}")
        comp = by_name[args.component]
    elif len(components) == 1:
        comp = components[0]
    else:
        raise SystemExit(f"[{label}] single mode needs --component NAME "
                         f"(one of: {sorted(by_name)})")
    cfg = tyro.cli(config_cls, args=cfg_argv, default=comp.base)
    n = run_shards(build_fn, [cfg], args.seeds, _db(comp), vmappable=comp.vmappable)
    consolidate(_db(comp))
    print(f"[{label}:{comp.name}] saved {n} new run(s) to {_db(comp)}")
