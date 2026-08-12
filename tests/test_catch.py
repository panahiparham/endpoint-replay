"""End-to-end tests for the Catch environment integration.

Unlike Atari, ``catch_jax.Catch`` is a pure, non-stateful jax env that already
speaks this repo's six-tuple ``GymEnv`` protocol directly, so these tests drive
the real env (no fakes) through the registry and the full agent loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))  # for `import main` / repo-root modules

from environments import ENVIRONMENTS
from environments.catch import CatchConfig


def test_spaces_and_dtype():
    env, params = ENVIRONMENTS["catch"].build(CatchConfig(ROWS=10, COLUMNS=5))
    assert env.observation_space(params).shape == (10, 5)
    assert env.observation_space(params).dtype == jnp.float32
    assert env.action_space(params).n == 3


def test_reset_and_step_shapes():
    env, params = ENVIRONMENTS["catch"].build(CatchConfig())
    obs, state = env.reset(jax.random.key(0), params)
    assert obs.shape == (10, 5) and obs.dtype == jnp.float32
    obs2, state2, reward, terminated, truncated, info = env.step(
        jax.random.key(1), state, jnp.int32(1), params
    )
    assert obs2.shape == (10, 5)
    assert reward.shape == () and terminated.shape == () and truncated.shape == ()
    assert info == {}


def test_episode_cutoff_truncates_not_terminates():
    """Catch never reaches a real MDP terminal - only the configured cutoff ends
    an episode, as a truncation."""
    env, params = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=5))
    _, state = env.reset(jax.random.key(0), params)
    flags = []
    for n in range(5):
        _obs, state, _r, term, trunc, _i = env.step(
            jax.random.key(n), state, jnp.int32(1), params
        )
        flags.append((bool(term), bool(trunc)))
    assert flags == [(False, False)] * 4 + [(False, True)]


def test_registered_as_vmappable():
    assert ENVIRONMENTS["catch"].vmappable is True


def test_ddqn_agent_vmapped_e2e():
    from agents.ddqn import DDQNConfig
    from main import ExperimentConfig, build

    config = ExperimentConfig(
        AGENT="ddqn",
        ENV="catch",
        AGENT_HYPERS=DDQNConfig(TOTAL_TIMESTEPS=50, BUFFER_SIZE=64, BATCH_SIZE=4,
                                LEARNING_STARTS=50, HIDDEN_SIZE=8),
        ENV_HYPERS=CatchConfig(EPISODE_CUTOFF=10),
    )
    train = build(config)
    keys = jax.vmap(jax.random.key)(jnp.arange(3))
    out = jax.jit(jax.vmap(train))(keys)
    m = out["metrics"]
    assert m["reward"].shape == (3, 50)
    # a 10-step cutoff over 50 steps truncates exactly 5 times per seed
    assert np.asarray(m["truncated"]).sum(axis=1).tolist() == [5, 5, 5]
    assert not np.asarray(m["terminated"]).any()


def test_ddqn_agent_e2e():
    from agents.ddqn import DDQNConfig, make_train

    env, params = ENVIRONMENTS["catch"].build(CatchConfig(EPISODE_CUTOFF=10))
    train = make_train(
        DDQNConfig(TOTAL_TIMESTEPS=100, BUFFER_SIZE=200, BATCH_SIZE=8,
                   LEARNING_STARTS=10, TARGET_NETWORK_FREQUENCY=20),
        env, params,
    )
    out = jax.jit(train)(jax.random.key(0))
    m = out["metrics"]
    assert m["reward"].shape == (100,)
    assert np.isfinite(np.asarray(m["loss"])).all()
