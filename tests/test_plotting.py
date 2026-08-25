"""Tests for the analysis-notebook plotting helpers (``experiment.plotting``).

Builds tiny fake stores with the real harness (``sweep_points`` + ``run_shards``,
as in ``test_harness.py``) so the sweep-column plumbing (``AGENT_HYPERS.LR``
flattening etc.) is exercised for real, without pulling in jax agents/envs.
"""

from __future__ import annotations

import dataclasses
import io
import json

import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment.core import _connect_write, load_runs, run_shards, sweep_points
from experiment.plotting import (
    average_lifetime_reward,
    average_lifetime_reward_stack,
    ema_reward,
    ema_reward_grids_for,
    episode_lengths,
    episode_returns,
    seed_grids_for,
    style,
    weighted_lifetime_return,
    weighted_lifetime_return_stack,
)

T = 6  # curve length used by the fake agent


@dataclasses.dataclass(frozen=True)
class Hypers:
    LR: float = 0.1


@dataclasses.dataclass(frozen=True)
class Cfg:
    HYPERS: Hypers = dataclasses.field(default_factory=Hypers)


def fake_build(config):
    """train(rng) -> two episodes (length 2, then 4), reward == LR per step."""
    lr = config.HYPERS.LR

    def train(rng):
        del rng
        return {
            "metrics": {
                "reward": jnp.full((T,), lr),
                "terminated": jnp.zeros((T,)),
                "truncated": jnp.array([0, 1, 0, 0, 0, 1], dtype=jnp.float32),
            }
        }

    return train


# --- episode_lengths / weighted_lifetime_return -----------------------------


def test_episode_lengths():
    # ends are inclusive 0-indexed positions: episode 0 spans indices 0-1 (len
    # 2), episode 1 spans 2-5 (len 4), episode 2 is just index 6 (len 1)
    assert list(episode_lengths(np.array([1, 5, 6]))) == [2, 4, 1]


def test_episode_lengths_corrects_for_dead_steps():
    """Hand-computed run of 3 episodes (lengths 3, 2, 4), each followed by a
    NEXT_STEP dead step (reward 0) except the last. A plain diff on `ends`
    over-counts every episode after the first by exactly one - the dead step
    sitting right before it - unless `dead` is passed to correct for it."""
    reward = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    terminated = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
    truncated = [0] * len(reward)
    dead = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]

    ends, rets = episode_returns(reward, terminated, truncated)
    assert list(ends) == [2, 5, 10]
    assert list(rets) == [3.0, 2.0, 4.0]              # episode_returns is exact

    assert list(episode_lengths(ends)) == [3, 3, 5]   # uncorrected: off by one
                                                        # for every later episode
    assert list(episode_lengths(ends, dead=dead)) == [3, 2, 4]  # corrected


def test_weighted_lifetime_return_weights_by_length():
    # episode 1: length 2, return 2*0.1; episode 2: length 4, return 4*0.2
    reward = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
    terminated = [0, 0, 0, 0, 0, 0]
    truncated = [0, 1, 0, 0, 0, 1]
    got = weighted_lifetime_return(reward, terminated, truncated)
    assert got == pytest.approx((0.2 * 2 + 0.8 * 4) / 6)


def test_weighted_lifetime_return_nan_with_no_completed_episode():
    reward, terminated, truncated = [0.1] * 4, [0] * 4, [0] * 4
    assert np.isnan(weighted_lifetime_return(reward, terminated, truncated))


def test_weighted_lifetime_return_corrects_for_dead_steps():
    """The same dead-step inflation that biases episode_lengths biases the
    length-weighted average too, since it weights by length: episode 1 is
    length 2 (return 10), a dead step follows, then episode 2 is length 5
    (return -10). Uncorrected, episode 2's weight is inflated to 6, pulling
    the weighted average toward its (negative) return more than it should."""
    reward = [10, 0, 0, -2, -2, -2, -2, -2]
    terminated = [0, 1, 0, 0, 0, 0, 0, 1]
    truncated = [0] * len(reward)
    dead = [0, 0, 1, 0, 0, 0, 0, 0]

    wrong = weighted_lifetime_return(reward, terminated, truncated)
    corrected = weighted_lifetime_return(reward, terminated, truncated, dead=dead)

    assert wrong == pytest.approx((10 * 2 + -10 * 6) / 8)       # weights [2, 6]
    assert corrected == pytest.approx((10 * 2 + -10 * 5) / 7)   # weights [2, 5]
    assert wrong != pytest.approx(corrected)


# --- average_lifetime_reward -------------------------------------------------


def test_average_lifetime_reward_is_mean_reward():
    assert average_lifetime_reward([0.0, 1.0, 1.0, -1.0]) == pytest.approx(0.25)


def test_average_lifetime_reward_ignores_episode_boundaries():
    # unlike weighted_lifetime_return, no terminated/truncated needed at all,
    # and a run with zero completed episodes is still a valid mean, not NaN
    assert average_lifetime_reward([2.0, 2.0, 2.0]) == pytest.approx(2.0)


def test_average_lifetime_reward_stack_shape_and_values(tmp_path):
    db = tmp_path / "comp.db"
    configs = sweep_points(Cfg(), {"HYPERS.LR": [0.1, 0.2]})
    run_shards(fake_build, configs, seeds=[0, 1], db_path=db)

    stack = average_lifetime_reward_stack(db, "HYPERS.LR", [0.1, 0.2])
    assert stack.shape == (2, 2)  # 2 seeds x 2 LR values
    # fake_build's reward is constant == LR every step, so the mean is LR itself
    np.testing.assert_allclose(stack[:, 0], 0.1)
    np.testing.assert_allclose(stack[:, 1], 0.2)


# --- ema_reward ---------------------------------------------------------------


def test_ema_reward_first_value_is_unsmoothed():
    assert ema_reward([5.0, 1.0, 1.0], beta=0.5)[0] == pytest.approx(5.0)


def test_ema_reward_matches_manual_recursion():
    reward = [1.0, 0.0, 1.0, 0.0]
    beta = 0.5
    expected = [reward[0]]
    for r in reward[1:]:
        expected.append(beta * expected[-1] + (1 - beta) * r)
    np.testing.assert_allclose(ema_reward(reward, beta=beta), expected)


def test_ema_reward_empty():
    assert ema_reward([]).size == 0


def test_ema_reward_grids_for_shape_and_values(tmp_path):
    db = tmp_path / "comp.db"
    configs = sweep_points(Cfg(), {"HYPERS.LR": [0.1, 0.2]})
    run_shards(fake_build, configs, seeds=[0, 1], db_path=db)

    df = load_runs(db, write_csv=False)
    rids = df.filter(df["HYPERS.LR"] == 0.1)["run_id"].to_list()
    stack = ema_reward_grids_for(db, np.array([0.0, 5.0]), beta=0.5, run_ids=rids)
    assert stack.shape == (2, 2)
    # fake_build's reward is constant == LR every step, so the EMA of it is
    # the same constant regardless of beta
    np.testing.assert_allclose(stack, 0.1)


# --- seed_grids_for(run_ids=...) --------------------------------------------


def test_seed_grids_for_run_ids_filters(tmp_path):
    db = tmp_path / "comp.db"
    configs = sweep_points(Cfg(), {"HYPERS.LR": [0.1, 0.2]})
    run_shards(fake_build, configs, seeds=[0, 1], db_path=db)

    all_grid = seed_grids_for(db, np.array([0.0, 6.0]))
    assert all_grid.shape == (4, 2)  # 2 LRs x 2 seeds

    df = load_runs(db, write_csv=False)
    lo_rids = df.filter(df["HYPERS.LR"] == 0.1)["run_id"].to_list()
    lo_grid = seed_grids_for(db, np.array([0.0, 6.0]), run_ids=lo_rids)
    assert lo_grid.shape == (2, 2)


# --- weighted_lifetime_return_stack -----------------------------------------


def test_weighted_lifetime_return_stack_shape_and_values(tmp_path):
    db = tmp_path / "comp.db"
    configs = sweep_points(Cfg(), {"HYPERS.LR": [0.1, 0.2]})
    run_shards(fake_build, configs, seeds=[0, 1], db_path=db)

    stack = weighted_lifetime_return_stack(db, "HYPERS.LR", [0.1, 0.2])
    assert stack.shape == (2, 2)  # 2 seeds x 2 LR values
    # episode returns are 2*LR (length 2) and 4*LR (length 4): weighted mean
    # = (2*LR*2 + 4*LR*4) / 6 = 10*LR/3
    np.testing.assert_allclose(stack[:, 0], 10 * 0.1 / 3)
    np.testing.assert_allclose(stack[:, 1], 10 * 0.2 / 3)


def _insert_run(db_path, run_id, seed, extra_config, reward, terminated, truncated):
    buf = io.BytesIO()
    np.savez(buf, reward=np.asarray(reward, dtype=np.float32),
              terminated=np.asarray(terminated), truncated=np.asarray(truncated))
    conn = _connect_write(db_path)
    conn.execute(
        "INSERT INTO runs (run_id, config_id, seed, config_json, metrics_json, "
        "curves) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, "c", seed,
         json.dumps({"run_id": run_id, "seed": seed, **extra_config}),
         "{}", buf.getvalue()),
    )
    conn.commit()
    conn.close()


def test_weighted_lifetime_return_stack_pads_missing_value(tmp_path):
    db = tmp_path / "comp.db"
    # only LR=0.1 has any stored runs; LR=0.2 is entirely absent
    _insert_run(db, "r0", 0, {"LR": 0.1}, [1, 1], [0, 0], [0, 1])

    stack = weighted_lifetime_return_stack(db, "LR", [0.1, 0.2])
    assert stack.shape == (1, 2)
    # one length-2 episode of return 2 -> weighted mean is just 2
    assert stack[0, 0] == pytest.approx(2.0)
    assert np.isnan(stack[0, 1])


# --- style() axis-label overrides -------------------------------------------


def test_style_default_and_override_labels():
    fig, ax = plt.subplots()
    style(ax)
    assert ax.get_xlabel() == "Timestep"
    assert ax.get_ylabel() == "Return"
    plt.close(fig)

    fig, ax = plt.subplots()
    style(ax, xlabel="Learning rate", ylabel="Weighted lifetime return")
    assert ax.get_xlabel() == "Learning rate"
    assert ax.get_ylabel() == "Weighted lifetime return"
    plt.close(fig)
