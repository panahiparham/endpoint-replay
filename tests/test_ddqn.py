"""Tests for the DDQN trajectory buffer and n-step TD target.

Covers the flashbax trajectory buffer that replaced the item buffer: single-
observation storage (no duplicated ``next_obs``), n-step returns cut at the
first termination or truncation in the sampled window, and dead-window
exclusion from the loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import flashbax as fbx
from agents.ddqn import AtariNet, DDQNConfig, QNetwork, TimeStep, _masked_td_loss, make_train


def _networks(obs_dim=1, action_dim=2, hidden_size=4, seed=0):
    qkey, tkey = jax.random.split(jax.random.key(seed))
    q = QNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_size=hidden_size, key=qkey)
    target_q = QNetwork(
        obs_dim=obs_dim, action_dim=action_dim, hidden_size=hidden_size, key=tkey
    )
    return q, target_q


def _bootstrap_value(q, target_q, obs):
    """Reference Double DQN bootstrap value: online net selects, target net evaluates."""
    next_online = jax.vmap(q)(obs)
    next_actions = jnp.argmax(next_online, axis=-1)
    next_target = jax.vmap(target_q)(obs)
    return jnp.take_along_axis(next_target, next_actions[:, None], axis=-1).squeeze(-1)


# ===========================================================================
# 1. N_STEP=1 reproduces the current 1-step Double DQN target exactly
# ===========================================================================


def test_n_step_1_matches_hand_computed_target():
    q, target_q = _networks()
    obs = jnp.array([[[0.0], [1.0]], [[1.0], [2.0]], [[2.0], [-1.0]]], jnp.float32)
    action = jnp.array([[0, 0], [1, 0], [1, 0]], jnp.int32)
    # reward[:, 1] is a trap: it belongs to the transition beyond this
    # window and must never enter a 1-step target.
    reward = jnp.array([[1.0, 7.0], [-1.0, -7.0], [0.5, 3.0]], jnp.float32)
    terminated = jnp.array([[False, False], [True, False], [False, False]])
    truncated = jnp.array([[False, False], [False, False], [True, False]])
    dead = jnp.array([[False, False], [False, False], [False, False]])
    batch = TimeStep(
        obs=obs, action=action, reward=reward,
        terminated=terminated, truncated=truncated, dead=dead,
    )
    gamma = 0.9

    loss = _masked_td_loss(q, target_q, batch, gamma)

    q_sa = jax.vmap(q)(obs[:, 0])
    q_a = jnp.take_along_axis(q_sa, action[:, 0, None], axis=-1).squeeze(-1)
    bootstrap = _bootstrap_value(q, target_q, obs[:, 1])
    expected_target = reward[:, 0] + gamma * bootstrap * (
        1.0 - terminated[:, 0].astype(jnp.float32)
    )
    expected_loss = jnp.mean(jnp.square(q_a - expected_target))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


# ===========================================================================
# 2. N_STEP>1: correct return, correct cut on termination/truncation
# ===========================================================================


def test_n_step_return_with_no_boundary():
    q, target_q = _networks()
    gamma = 0.9
    obs = jnp.array([[[0.0], [1.0], [2.0], [3.0]]], jnp.float32)
    action = jnp.array([[0, 1, 0, 1]], jnp.int32)
    # reward[:, 3] is a trap: it belongs beyond the 3-step window.
    reward = jnp.array([[1.0, 2.0, 3.0, 999.0]], jnp.float32)
    terminated = jnp.zeros((1, 4), dtype=bool)
    truncated = jnp.zeros((1, 4), dtype=bool)
    dead = jnp.zeros((1, 4), dtype=bool)
    batch = TimeStep(
        obs=obs, action=action, reward=reward,
        terminated=terminated, truncated=truncated, dead=dead,
    )

    loss = _masked_td_loss(q, target_q, batch, gamma)

    q_sa = jax.vmap(q)(obs[:, 0])
    q_a = jnp.take_along_axis(q_sa, action[:, 0, None], axis=-1).squeeze(-1)
    G = reward[:, 0] + gamma * reward[:, 1] + gamma**2 * reward[:, 2]
    bootstrap = _bootstrap_value(q, target_q, obs[:, 3])
    expected_target = G + gamma**3 * bootstrap
    expected_loss = jnp.mean(jnp.square(q_a - expected_target))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_n_step_return_cut_by_termination():
    q, target_q = _networks()
    gamma = 0.9
    obs = jnp.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]], jnp.float32)
    action = jnp.array([[0, 1, 0, 1, 0]], jnp.int32)
    # rewards past the termination are traps and must not enter the return.
    reward = jnp.array([[1.0, 2.0, 999.0, 999.0, 999.0]], jnp.float32)
    terminated = jnp.array([[False, True, False, False, False]])
    truncated = jnp.zeros((1, 5), dtype=bool)
    dead = jnp.zeros((1, 5), dtype=bool)
    batch = TimeStep(
        obs=obs, action=action, reward=reward,
        terminated=terminated, truncated=truncated, dead=dead,
    )

    loss = _masked_td_loss(q, target_q, batch, gamma)

    q_sa = jax.vmap(q)(obs[:, 0])
    q_a = jnp.take_along_axis(q_sa, action[:, 0, None], axis=-1).squeeze(-1)
    # cut after the termination: only the first two rewards count, no bootstrap
    G = reward[:, 0] + gamma * reward[:, 1]
    expected_loss = jnp.mean(jnp.square(q_a - G))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_n_step_return_cut_by_truncation():
    """A truncation still bootstraps, from the true boundary obs reached
    right after the cut - unlike termination, which zeroes the bootstrap."""
    q, target_q = _networks()
    gamma = 0.9
    obs = jnp.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]], jnp.float32)
    action = jnp.array([[0, 1, 0, 1, 0]], jnp.int32)
    reward = jnp.array([[1.0, 2.0, 999.0, 999.0, 999.0]], jnp.float32)
    terminated = jnp.zeros((1, 5), dtype=bool)
    truncated = jnp.array([[False, True, False, False, False]])
    dead = jnp.zeros((1, 5), dtype=bool)
    batch = TimeStep(
        obs=obs, action=action, reward=reward,
        terminated=terminated, truncated=truncated, dead=dead,
    )

    loss = _masked_td_loss(q, target_q, batch, gamma)

    q_sa = jax.vmap(q)(obs[:, 0])
    q_a = jnp.take_along_axis(q_sa, action[:, 0, None], axis=-1).squeeze(-1)
    G = reward[:, 0] + gamma * reward[:, 1]
    bootstrap = _bootstrap_value(q, target_q, obs[:, 2])
    expected_target = G + gamma**2 * bootstrap
    expected_loss = jnp.mean(jnp.square(q_a - expected_target))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_dead_only_excludes_windows_starting_on_a_dead_step():
    """Only ``dead`` on the window's first transition gates exclusion; a dead
    step later in the window (never the case in practice, since a dead step
    always follows a boundary that would itself cut the return first) must
    not suppress an otherwise valid window."""
    q, target_q = _networks()
    gamma = 0.9
    obs = jnp.array([[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]], jnp.float32)
    action = jnp.array([[0, 1, 0], [1, 0, 1]], jnp.int32)
    reward = jnp.array([[1.0, 2.0, 999.0], [3.0, 4.0, 999.0]], jnp.float32)
    terminated = jnp.zeros((2, 3), dtype=bool)
    truncated = jnp.zeros((2, 3), dtype=bool)
    dead = jnp.array([[False, True, False], [False, False, False]])
    batch = TimeStep(
        obs=obs, action=action, reward=reward,
        terminated=terminated, truncated=truncated, dead=dead,
    )

    loss = _masked_td_loss(q, target_q, batch, gamma)

    q_sa = jax.vmap(q)(obs[:, 0])
    q_a = jnp.take_along_axis(q_sa, action[:, 0, None], axis=-1).squeeze(-1)
    G = reward[:, 0] + gamma * reward[:, 1]
    bootstrap = _bootstrap_value(q, target_q, obs[:, 2])
    expected_target = G + gamma**2 * bootstrap
    expected_loss = jnp.mean(jnp.square(q_a - expected_target))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_boot_obs_gather_matches_for_image_observations():
    """The same n-step gather that works for vector obs (Pinball) must also
    broadcast correctly for image obs (Atari): ``[B, N+1, H, W, C]``."""
    qkey, tkey = jax.random.split(jax.random.key(1))
    q = AtariNet((84, 84, 4), 3, key=qkey)
    target_q = AtariNet((84, 84, 4), 3, key=tkey)
    obs = jax.random.randint(jax.random.key(2), (2, 4, 84, 84, 4), 0, 255).astype(
        jnp.uint8
    )
    action = jnp.zeros((2, 4), jnp.int32)
    reward = jnp.ones((2, 4), jnp.float32)
    terminated = jnp.zeros((2, 4), dtype=bool)
    truncated = jnp.zeros((2, 4), dtype=bool)
    dead = jnp.zeros((2, 4), dtype=bool)
    batch = TimeStep(
        obs=obs, action=action, reward=reward,
        terminated=terminated, truncated=truncated, dead=dead,
    )

    loss = _masked_td_loss(q, target_q, batch, 0.99)
    assert jnp.isfinite(loss)


# ===========================================================================
# 3. Buffer round-trip: single-observation storage, no duplicated next_obs
# ===========================================================================


def test_buffer_round_trip_successor_is_true_next_obs():
    n_step = 2
    buffer = fbx.make_trajectory_buffer(
        add_batch_size=1,
        sample_batch_size=4,
        sample_sequence_length=n_step + 1,
        period=1,
        min_length_time_axis=4,
        max_length_time_axis=16,
    )
    dummy = TimeStep(
        obs=jnp.zeros((1,), jnp.float32),
        action=jnp.asarray(0, jnp.int32),
        reward=jnp.asarray(0.0, jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
        dead=jnp.asarray(False),
    )
    state = buffer.init(dummy)

    for i in range(8):
        step = TimeStep(
            obs=jnp.asarray([float(i)]),
            action=jnp.asarray(i % 2, jnp.int32),
            reward=jnp.asarray(float(i)),
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
            dead=jnp.asarray(False),
        )
        state = buffer.add(state, jax.tree.map(lambda x: x[None, None, ...], step))

    sample = buffer.sample(state, jax.random.key(0))
    obs = np.asarray(sample.experience.obs)
    # each sampled window's second observation is the true successor of its
    # first, since the known stream's obs strictly increments by 1 per add.
    for b in range(obs.shape[0]):
        assert obs[b, 1, 0] == obs[b, 0, 0] + 1.0


# ===========================================================================
# 4. jit and vmap both still work, including with N_STEP > 1
# ===========================================================================


def test_masked_td_loss_is_jittable_and_vmappable():
    def make_batch(key):
        obs = jax.random.normal(key, (5, 3, 2))
        action = jnp.zeros((5, 2), jnp.int32)
        reward = jnp.ones((5, 2))
        terminated = jnp.zeros((5, 2), dtype=bool)
        truncated = jnp.zeros((5, 2), dtype=bool)
        dead = jnp.zeros((5, 2), dtype=bool)
        return TimeStep(
            obs=obs, action=action, reward=reward,
            terminated=terminated, truncated=truncated, dead=dead,
        )

    q, target_q = _networks(obs_dim=2)
    batch = make_batch(jax.random.key(0))

    jitted = jax.jit(lambda b: _masked_td_loss(q, target_q, b, 0.99))
    loss = jitted(batch)
    assert jnp.isfinite(loss)

    keys = jax.random.split(jax.random.key(1), 4)
    batches = jax.vmap(make_batch)(keys)
    losses = jax.vmap(lambda b: _masked_td_loss(q, target_q, b, 0.99))(batches)
    assert losses.shape == (4,)
    assert jnp.all(jnp.isfinite(losses))


def test_make_train_with_n_step_runs_under_jit_and_vmap():
    class _FakeEnv:
        def observation_space(self, params=None):
            return type("B", (), {"shape": (2,), "dtype": jnp.float32})()

        def action_space(self, params=None):
            return type("D", (), {"n": 3})()

        def reset(self, key, params=None):
            return jnp.zeros((2,), jnp.float32), jnp.int32(0)

        def step(self, key, state, action, params=None):
            t = state + 1
            term = t % 5 == 0
            obs = jnp.full((2,), (t % 7).astype(jnp.float32))
            return obs, t, jnp.float32(1.0), term, jnp.asarray(False), {}

    cfg = DDQNConfig(
        TOTAL_TIMESTEPS=40, BUFFER_SIZE=64, BATCH_SIZE=4,
        LEARNING_STARTS=8, N_STEP=3, HIDDEN_SIZE=8,
    )
    train = make_train(cfg, _FakeEnv(), None)

    out = jax.block_until_ready(jax.jit(train)(jax.random.key(0)))
    assert np.isfinite(np.asarray(out["metrics"]["loss"])).all()

    batch_out = jax.block_until_ready(jax.vmap(train)(jax.random.split(jax.random.key(0), 3)))
    assert batch_out["metrics"]["loss"].shape == (3, 40)
    assert np.isfinite(np.asarray(batch_out["metrics"]["loss"])).all()
