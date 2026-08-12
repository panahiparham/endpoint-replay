"""Tests for the experiment harness (``experiment.core``).

These use a tiny fake nested config and a fake ``main`` (a jittable/vmappable
function returning a ``{"metrics": {...}}`` pytree) so the tests exercise the
harness itself - sharding, dedup, workers, consolidate, storage, CLI - without
pulling in the real agents/environments.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiment import (
    Component,
    build_global_shards,
    build_shards,
    config_id,
    consolidate,
    expand_grid,
    load_curves,
    load_runs,
    pending_runs,
    run_experiment,
    run_id,
    run_shards,
    sweep_points,
)
from experiment.core import (
    _connect_write,
    _existing_run_ids,
    _insert_run,
    _part_path,
    _set_path,
)

T = 4  # curve length used by the fake agent


# --- fakes ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Hypers:
    A: float = 1.0
    B: int = 2


@dataclasses.dataclass(frozen=True)
class Cfg:
    NAME: str = "x"
    HYPERS: Hypers = dataclasses.field(default_factory=Hypers)


def fake_main(config: Cfg, rng: jax.Array) -> dict:
    """Jittable/vmappable stand-in for ``main``: constant per-step curves plus a
    non-``metrics`` key the harness must ignore."""
    del rng, config
    return {
        "runner_state": {"buf": jnp.ones((T, 5))},  # must NOT be stored
        "metrics": {
            "reward": jnp.arange(T, dtype=jnp.float32),
            "terminated": jnp.zeros(T, dtype=jnp.float32),
        },
    }


def fake_build(config):
    """build(config) -> train; the harness builds host-side then jits/vmaps train."""
    return lambda rng: fake_main(config, rng)


def _run_ids_in(db_path) -> set[str]:
    return set(load_runs(db_path, write_csv=False)["run_id"].to_list())


# --- sweep expansion --------------------------------------------------------


def test_expand_grid_product():
    assert expand_grid({"a": [1, 2], "b": ["x"]}) == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "x"},
    ]


def test_expand_grid_empty_is_single_point():
    assert expand_grid({}) == [{}]


def test_set_path_nested_and_immutable():
    base = Cfg()
    out = _set_path(base, "HYPERS.A", 9.0)
    assert out.HYPERS.A == 9.0
    assert base.HYPERS.A == 1.0  # original untouched


def test_sweep_points_applies_combos():
    base = Cfg()
    pts = sweep_points(base, {"HYPERS.A": [1.0, 2.0], "NAME": ["p", "q"]})
    assert len(pts) == 4
    assert {(p.NAME, p.HYPERS.A) for p in pts} == {
        ("p", 1.0), ("p", 2.0), ("q", 1.0), ("q", 2.0)
    }
    assert base.HYPERS.A == 1.0  # BASE never mutated


def test_sweep_points_empty_returns_base():
    base = Cfg()
    assert sweep_points(base, {}) == [base]


# --- identity ---------------------------------------------------------------


def test_config_id_stable_and_seed_excluded():
    a = Cfg(HYPERS=Hypers(A=0.5))
    b = Cfg(HYPERS=Hypers(A=0.5))
    assert config_id(a) == config_id(b)
    assert config_id(Cfg(HYPERS=Hypers(A=0.6))) != config_id(a)
    # accepts the asdict mapping too, and gives the same id
    assert config_id(dataclasses.asdict(a)) == config_id(a)


def test_config_id_float_rounding():
    # difference below 12 significant figures collapses to the same id
    assert config_id(Cfg(HYPERS=Hypers(A=1.0))) == config_id(
        Cfg(HYPERS=Hypers(A=1.0 + 1e-14))
    )


def test_run_id_format():
    c = Cfg()
    assert run_id(c, 3) == f"{config_id(c)}_s3"


# --- sharding ---------------------------------------------------------------


def test_build_shards_chunks_by_size():
    configs = sweep_points(Cfg(), {"HYPERS.A": [1.0, 2.0]})  # 2 configs
    shards = build_shards(configs, seeds=list(range(5)), shard_size=2)
    # each config -> ceil(5/2) = 3 shards
    assert len(shards) == 6
    assert [len(s.seeds) for s in shards] == [2, 2, 1, 2, 2, 1]
    # a shard never mixes configs
    assert all(isinstance(s.config, Cfg) for s in shards)


def test_build_shards_none_is_one_per_config():
    configs = sweep_points(Cfg(), {"HYPERS.A": [1.0, 2.0]})
    shards = build_shards(configs, seeds=[0, 1, 2], shard_size=None)
    assert len(shards) == 2
    assert all(s.seeds == [0, 1, 2] for s in shards)


# --- components: pooled global shards ---------------------------------------


def test_build_global_shards_tags_and_orders_by_component():
    comps = [
        Component("a", Cfg(NAME="a"), sweep={"HYPERS.A": [1.0, 2.0]},
                  seeds=[0, 1], shard_size=1),
        Component("b", Cfg(NAME="b"), seeds=[0, 1], shard_size=None),
    ]
    pairs = build_global_shards(comps)
    # component "a": 2 configs x ceil(2/1)=2 shards = 4; component "b": 1 shard
    assert [c.name for c, _ in pairs] == ["a", "a", "a", "a", "b"]
    assert all(isinstance(s, type(pairs[0][1])) for _, s in pairs)
    # a shard_size_override bigger than the seed count collapses every component to
    # one shard per config (2 for "a", 1 for "b")
    pairs2 = build_global_shards(comps, shard_size_override=10)
    assert [c.name for c, _ in pairs2] == ["a", "a", "b"]


def test_global_shards_round_robin_is_disjoint_and_complete():
    comps = [
        Component("a", Cfg(NAME="a"), sweep={"HYPERS.A": [1.0, 2.0]},
                  seeds=[0, 1], shard_size=1),
        Component("b", Cfg(NAME="b"), seeds=[0, 1, 2], shard_size=2),
    ]
    pairs = build_global_shards(comps)
    N = 3
    seen = []
    for w in range(N):
        seen.extend(pairs[w::N])
    # every shard assigned exactly once across workers
    assert len(seen) == len(pairs)
    assert {id(s) for _, s in seen} == {id(s) for _, s in pairs}


# --- run_shards: storage, curves-only, dedup --------------------------------


def test_run_shards_saves_curves_only(tmp_path):
    db = tmp_path / "runs.db"
    configs = [Cfg(HYPERS=Hypers(A=1.0))]
    n = run_shards(fake_build, configs, seeds=[0, 1, 2], db_path=db)
    assert n == 3

    df = load_runs(db, write_csv=False)
    assert df.height == 3
    # nested config flattened to dotted columns; NO metric columns (curves-only)
    assert "HYPERS.A" in df.columns and "NAME" in df.columns
    assert "reward" not in df.columns  # metrics_json is empty by design

    rid = df["run_id"][0]
    curves = load_curves(db, rid)
    assert set(curves) == {"reward", "terminated"}       # runner_state ignored
    assert curves["reward"].shape == (T,)
    np.testing.assert_array_equal(curves["reward"], np.arange(T))


def test_run_shards_dedup_resume(tmp_path):
    db = tmp_path / "runs.db"
    configs = [Cfg()]
    assert run_shards(fake_build, configs, [0, 1], db) == 2
    # second call: everything already present -> nothing new
    assert run_shards(fake_build, configs, [0, 1], db) == 0
    # extending seeds only computes the delta
    assert run_shards(fake_build, configs, [0, 1, 2], db) == 1


def test_pending_runs_shrinks_after_run(tmp_path):
    db = tmp_path / "runs.db"
    configs = sweep_points(Cfg(), {"HYPERS.A": [1.0, 2.0]})
    assert len(pending_runs(configs, [0, 1], db)) == 4
    run_shards(fake_build, configs, [0, 1], db)
    assert pending_runs(configs, [0, 1], db) == []


# --- workers: round-robin is disjoint and complete --------------------------


def test_workers_disjoint_and_complete(tmp_path):
    configs = sweep_points(Cfg(), {"HYPERS.A": [1.0, 2.0, 3.0]})  # 3 configs
    seeds = [0, 1]
    # single-worker reference set
    ref_db = tmp_path / "ref.db"
    run_shards(fake_build, configs, seeds, ref_db)
    ref_ids = _run_ids_in(ref_db)
    assert len(ref_ids) == 6

    # two workers over a fresh store, shard_size=1 so there are several shards each
    w_db = tmp_path / "workers.db"
    n0 = run_shards(fake_build, configs, seeds, w_db, shard_size=1,
                    worker_index=0, num_workers=2)
    n1 = run_shards(fake_build, configs, seeds, w_db, shard_size=1,
                    worker_index=1, num_workers=2)
    assert n0 + n1 == 6  # every run computed exactly once (disjoint, complete)

    # both parts exist before consolidation
    assert _part_path(w_db, 0).exists() and _part_path(w_db, 1).exists()
    assert _run_ids_in(w_db) == ref_ids


# --- non-vmappable path (e.g. Atari: env FFI can't be vmapped) --------------


def test_run_shards_non_vmappable_loop(tmp_path):
    db = tmp_path / "runs.db"
    configs = [Cfg(HYPERS=Hypers(A=1.0))]
    n = run_shards(fake_build, configs, [0, 1, 2], db, vmappable=False)
    assert n == 3
    df = load_runs(db, write_csv=False)
    assert df.height == 3
    curves = load_curves(db, df["run_id"][0])
    assert set(curves) == {"reward", "terminated"}
    np.testing.assert_array_equal(curves["reward"], np.arange(T))


def test_vmappable_and_loop_agree(tmp_path):
    configs = [Cfg(HYPERS=Hypers(A=2.0))]
    run_shards(fake_build, configs, [0, 1], tmp_path / "vmap.db", vmappable=True)
    run_shards(fake_build, configs, [0, 1], tmp_path / "loop.db", vmappable=False)
    assert _run_ids_in(tmp_path / "vmap.db") == _run_ids_in(tmp_path / "loop.db")


# --- consolidate ------------------------------------------------------------


def test_consolidate_merges_and_removes_parts(tmp_path):
    db = tmp_path / "runs.db"
    configs = sweep_points(Cfg(), {"HYPERS.A": [1.0, 2.0]})
    run_shards(fake_build, configs, [0, 1], db, shard_size=1,
               worker_index=0, num_workers=2)
    run_shards(fake_build, configs, [0, 1], db, shard_size=1,
               worker_index=1, num_workers=2)

    merged = consolidate(db)
    assert merged == 4
    assert db.exists()                              # consolidated <name>.db
    assert not (tmp_path / "runs.parts").exists()   # parts dir cleaned up
    # idempotent: nothing left to merge
    assert consolidate(db) == 0
    assert _run_ids_in(db) == {run_id(c, s) for c in configs for s in (0, 1)}


def test_insert_run_is_insert_or_ignore(tmp_path):
    conn = _connect_write(_part_path(tmp_path / "runs.db", 0))
    rec = {"run_id": "abcd_s0", "config_id": "abcd", "seed": 0}
    _insert_run(conn, rec, {}, {"reward": np.arange(3)})
    _insert_run(conn, rec, {}, {"reward": np.arange(3)})  # duplicate run_id
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    conn.close()


def test_existing_run_ids_unions_db_and_parts(tmp_path):
    db = tmp_path / "runs.db"
    configs = [Cfg()]
    run_shards(fake_build, configs, [0], db)               # -> runs.parts/part-0.db
    ids_before = _existing_run_ids(db)
    consolidate(db)                                        # -> runs.db
    ids_after = _existing_run_ids(db)
    assert ids_before == ids_after == {run_id(configs[0], 0)}


# --- run_experiment CLI -----------------------------------------------------


def _kw(tmp_path, **over):
    kw = dict(
        build_fn=fake_build, config_cls=Cfg,
        components=[
            Component("main", Cfg(), sweep={"HYPERS.A": [1.0, 2.0]}, seeds=[0, 1])
        ],
        results_dir=tmp_path, label="test",
    )
    kw.update(over)
    return kw


def test_run_experiment_requires_mode(tmp_path):
    with pytest.raises(SystemExit):
        run_experiment(argv=[], **_kw(tmp_path))
    with pytest.raises(SystemExit):
        run_experiment(argv=["bogus"], **_kw(tmp_path))


def test_run_experiment_plan_reports(tmp_path, capsys):
    run_experiment(argv=["plan"], **_kw(tmp_path))
    out = capsys.readouterr().out
    assert "[main] 2 configs x 2 seeds = 4 runs" in out
    assert "total: 4 runs across 1 component(s)" in out
    assert not (tmp_path / "main.db").exists()  # plan writes nothing


def test_run_experiment_sweep_then_consolidated(tmp_path):
    run_experiment(argv=["sweep"], **_kw(tmp_path))  # num_workers=1 -> auto consolidate
    assert (tmp_path / "main.db").exists()  # per-component store: <name>.db
    assert _run_ids_in(tmp_path / "main.db") == {
        run_id(c, s)
        for c in sweep_points(Cfg(), {"HYPERS.A": [1.0, 2.0]})
        for s in (0, 1)
    }


def test_run_experiment_single_defaults_to_sole_component(tmp_path):
    run_experiment(argv=["single", "--seeds", "0", "1"], **_kw(tmp_path))
    df = load_runs(tmp_path / "main.db", write_csv=False)
    assert df.height == 2  # one config (base) x 2 seeds


def test_run_experiment_single_requires_component_when_many(tmp_path):
    comps = [Component("a", Cfg(NAME="a"), seeds=[0]),
             Component("b", Cfg(NAME="b"), seeds=[0])]
    with pytest.raises(SystemExit):
        run_experiment(argv=["single", "--seeds", "0"],
                       **_kw(tmp_path, components=comps))
    # naming the component works and writes only its store
    run_experiment(argv=["single", "--component", "b", "--seeds", "0"],
                   **_kw(tmp_path, components=comps))
    assert (tmp_path / "b.db").exists()
    assert not (tmp_path / "a.db").exists()


def test_run_experiment_sweep_multi_component_separate_dbs(tmp_path):
    comps = [
        Component("a", Cfg(NAME="a"), sweep={"HYPERS.A": [1.0, 2.0]},
                  seeds=[0, 1], shard_size=1),
        Component("b", Cfg(NAME="b"), seeds=[0, 1, 2]),
    ]
    run_experiment(argv=["sweep"], **_kw(tmp_path, components=comps))
    # each component consolidated into its own <name>.db store
    assert (tmp_path / "a.db").exists()
    assert (tmp_path / "b.db").exists()
    assert _run_ids_in(tmp_path / "a.db") == {
        run_id(c, s)
        for c in sweep_points(Cfg(NAME="a"), {"HYPERS.A": [1.0, 2.0]})
        for s in (0, 1)
    }
    assert _run_ids_in(tmp_path / "b.db") == {
        run_id(Cfg(NAME="b"), s) for s in (0, 1, 2)
    }


def test_run_experiment_sweep_component_filter(tmp_path):
    comps = [Component("a", Cfg(NAME="a"), seeds=[0]),
             Component("b", Cfg(NAME="b"), seeds=[0])]
    run_experiment(argv=["sweep", "--component", "a"],
                   **_kw(tmp_path, components=comps))
    assert (tmp_path / "a.db").exists()
    assert not (tmp_path / "b.db").exists()  # filtered out


# --- plotting helpers (episode returns -> mean + bootstrap CI band) ----------


def test_plotting_returns_and_ci_band():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiment.plotting import (
        bootstrap_mean_ci,
        episode_returns,
        interp_on_grid,
        mean_over_seeds,
        plot_mean_ci,
        style,
    )

    # one run: an episode ends every 4 steps, reward -1/step -> each return is -4
    reward = -np.ones(12)
    terminated = np.zeros(12)
    truncated = np.zeros(12)
    terminated[[3, 7, 11]] = 1
    ends, rets = episode_returns(reward, terminated, truncated)
    assert list(ends) == [3, 7, 11]
    np.testing.assert_array_equal(rets, [-4, -4, -4])

    grid = np.linspace(0, 12, 20)
    stack = np.vstack([interp_on_grid(ends, rets, grid) for _ in range(5)])
    mean, ci_lo, ci_hi = bootstrap_mean_ci(stack, n_boot=200)
    where = ~np.isnan(mean)
    assert where.any()
    np.testing.assert_allclose(mean[where], -4)                 # constant return
    # zero-width band (all seeds equal)
    np.testing.assert_allclose(ci_lo[where], -4)
    np.testing.assert_allclose(mean_over_seeds(stack)[where], -4)

    fig, ax = plt.subplots()
    out = plot_mean_ci(ax, grid, stack, "x", "tab:blue", n_boot=200)
    style(ax, ylim=(-10, 0))
    np.testing.assert_allclose(out[where], -4)
    plt.close(fig)
