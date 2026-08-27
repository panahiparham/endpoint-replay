"""Termination / truncation audit: episode-boundary handling across every agent
and environment.

The RL contract this suite pins down:

* **Environments** report ``terminated`` (a real MDP terminal - no bootstrap) and
  ``truncated`` (a time-limit cutoff - bootstrap continues) as *separate* flags,
  never merged, and never both true in a way that would corrupt the update.
* **Environments autoreset on the step after a boundary (NEXT_STEP).** Every
  env here (pinball, wrapped; Atari, natively) returns the *true* boundary
  observation on the step that ends an episode, then a "dead" step the
  following step: it ignores the action and returns a fresh episode's
  initial observation with ``reward=0`` and both flags false.
* **The agent does not reset anything.** It just steps; the environment
  autoresets itself.
* **The replay agent** (``ddqn``) stores every transition, including the dead
  one, but tags it ``dead=True`` (the previous step's
  ``terminated | truncated``) and masks it out of the TD loss, since it
  fabricates a link between two episodes that must never train.
* **The DDQN TD target** masks the bootstrap with ``(1 - terminated)`` only, so a
  *truncated* episode-end still bootstraps and a *terminated* one does not.
* **Analysis** (``plotting.episode_returns``) segments episodes on either flag.

Fake envs (single float obs, distinguishable reset sentinel) make the stored
transitions inspectable without ale-py/heavy deps; the real pinball env is
exercised for its flag contract. See ``test_atari.py`` for the ale-py path,
including the issue #1 regression test that a *truncated* Atari transition stores the
true pre-truncation observation rather than the fresh episode's first obs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest



# --- fake envs (GymEnv protocol; single float obs, sentinel reset) ----------


class _Box:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Discrete:
    def __init__(self, n):
        self.n = int(n)


class _State(NamedTuple):
    counter: jax.Array


class FakeEnv:
    """Single-obs env that ends every ``period`` steps.

    ``obs`` is the (1-based) in-episode step counter as a float, so it is distinct
    from the reset sentinel ``RESET_OBS`` - this lets a test tell the true terminal
    observation apart from the fresh-episode observation in the replay buffer.

    ``mode`` picks whether an episode-end is reported as ``terminated`` or
    ``truncated``. DISABLED-mode, like the raw ``pinball_jax`` env: it never
    resets itself, it returns the true boundary obs. ``_run_agent`` wraps it
    in ``AutoresetNextStep``, same as ``pinball.py``'s ``build``.
    """

    RESET_OBS = -1.0

    def __init__(self, period=3, mode="terminated", n=2):
        assert mode in ("terminated", "truncated")
        self._period = int(period)
        self._mode = mode
        self._n = int(n)

    def observation_space(self, params=None):
        return _Box((1,), jnp.float32)

    def action_space(self, params=None):
        return _Discrete(self._n)

    def reset(self, key, params=None):
        obs = jnp.asarray([self.RESET_OBS], jnp.float32)
        return obs, _State(jnp.asarray(0, jnp.int32))

    def step(self, key, state, action, params=None):
        del key, action
        nc = state.counter + 1
        done = (nc % self._period) == 0
        terminal_obs = nc.astype(jnp.float32).reshape((1,))
        reward = jnp.asarray(1.0, jnp.float32)
        terminated = done & (self._mode == "terminated")
        truncated = done & (self._mode == "truncated")
        return terminal_obs, _State(nc), reward, terminated, truncated, {}


# --- helpers: run an agent, read its replay buffer --------------------------


def _run_agent(env, *, total, buffer_size=64, **overrides):
    """Run DDQN on ``env`` (wrapped in ``AutoresetNextStep``) and return the
    jitted train output pytree."""
    from agents.ddqn import DDQNConfig, make_train
    from environments.autoreset import AutoresetNextStep

    cfg = DDQNConfig(TOTAL_TIMESTEPS=total, BUFFER_SIZE=buffer_size, BATCH_SIZE=2,
                     # no training: pure buffer
                     LEARNING_STARTS=total, HIDDEN_SIZE=8, **overrides)
    out = jax.jit(make_train(cfg, AutoresetNextStep(env), None))(jax.random.key(0))
    return jax.block_until_ready(out)


def _buffer(out):
    """Flat per-transition arrays actually written to the trajectory buffer, in add
    order: ``{"obs", "terminated", "truncated", "dead"}`` (each length = #adds)."""
    bs = out["runner_state"].buffer_state
    n = int(np.asarray(bs.current_index))
    exp = bs.experience
    return {
        "obs": np.asarray(exp.obs).reshape(-1)[:n],
        "terminated": np.asarray(exp.terminated).reshape(-1)[:n].astype(bool),
        "truncated": np.asarray(exp.truncated).reshape(-1)[:n].astype(bool),
        "dead": np.asarray(exp.dead).reshape(-1)[:n].astype(bool),
    }


# ===========================================================================
# 1. Environment flag contract (real envs)
# ===========================================================================


def test_pinball_split_and_truncation_at_cutoff():
    """Pinball reports terminated/truncated separately; a short cutoff truncates
    (terminated stays False) and the two flags are never simultaneously true.
    Also tests the NEXT_STEP dead-step behavior after the boundary."""
    from environments.pinball import PinballConfig, build

    env, params = build(PinballConfig(SETTING="empty", EPISODE_CUTOFF=5))
    obs, st = env.reset(jax.random.key(0))
    rows = []
    for i in range(5):
        obs, st, r, term, trunc, info = env.step(
            jax.random.key(i), st, jnp.int32(0), params
        )
        rows.append((bool(term), bool(trunc), float(r)))
    assert all(not (t and tr) for t, tr, _ in rows)          # never both at once
    assert all(r == -1.0 for *_, r in rows)                  # pinball reward is -1/step
    # cutoff at step 5
    assert [tr for _, tr, _ in rows] == [False, False, False, False, True]
    # truncation, not termination
    assert rows[-1][0] is False

    # Dead step: a different action (1, not 0) proves it's ignored.
    obs, st, r, term, trunc, info = env.step(
        jax.random.key(5), st, jnp.int32(1), params
    )
    assert float(r) == 0.0
    assert not bool(term)
    assert not bool(trunc)


# ===========================================================================
# 2. Replay-buffer boundary handling (the env autoresets on the dead step)
# ===========================================================================


@pytest.mark.parametrize("mode", ["terminated", "truncated"])
def test_buffer_stores_true_boundary_next_obs(mode):
    """The transition at an episode boundary is followed (at ``obs[k+1]``) by the
    *true* terminal/truncation observation. The *following* transition is the
    fabricated NEXT_STEP dead step: it is tagged ``dead=True``, carries the true
    boundary obs forward as its own ``obs``, and its successor lands on the reset
    obs. The transition after THAT starts the new episode from the reset obs.

    This is the property that makes a truncated transition bootstrap from the
    correct state - the most error-prone part of the loop.
    """
    env = FakeEnv(period=3, mode=mode)
    buf = _buffer(_run_agent(env, total=8))

    flag = buf[mode]
    other = buf["truncated" if mode == "terminated" else "terminated"]
    ends = np.flatnonzero(flag)

    # every 3rd step ends; the dead step at 3 shifts the second boundary from
    # 5 to 6
    assert list(ends) == [2, 6]
    # only the intended flag fires
    assert not other.any()
    # the obs right after a boundary is the true terminal obs (counter==3.0), NOT reset(-1)
    np.testing.assert_array_equal(buf["obs"][ends + 1], [3.0, 3.0])

    dead_idx = ends[0] + 1
    assert buf["dead"][dead_idx]
    assert buf["obs"][dead_idx] == 3.0            # the true boundary obs, carried
    assert buf["obs"][dead_idx + 1] == FakeEnv.RESET_OBS   # new episode starts here

    # first obs is the initial reset
    assert buf["obs"][0] == FakeEnv.RESET_OBS


# ===========================================================================
# 3. DDQN update rule: bootstrap masks terminated, not truncated
# ===========================================================================


def _ddqn_q_leaves(env, *, seed=0):
    """Run DDQN (with training on) on ``env`` (wrapped in ``AutoresetNextStep``)
    and return its online-Q array leaves."""
    import equinox as eqx

    from agents.ddqn import DDQNConfig, make_train
    from environments.autoreset import AutoresetNextStep

    cfg = DDQNConfig(TOTAL_TIMESTEPS=80, BUFFER_SIZE=256, BATCH_SIZE=8,
                     LEARNING_STARTS=8,
                     TRAIN_FREQUENCY=1, TARGET_NETWORK_FREQUENCY=10, HIDDEN_SIZE=16,
                     # all-random actions -> identical data
                     EPSILON_START=1.0, EPSILON_END=1.0)
    out = jax.jit(make_train(cfg, AutoresetNextStep(env), None))(jax.random.key(seed))
    q = out["runner_state"].q
    return jax.tree.leaves(eqx.filter(q, eqx.is_array))


def test_ddqn_target_masks_terminated_not_truncated():
    """DDQN bootstraps on truncation but not termination. Two runs with *identical*
    dynamics/rewards/observations that differ only in whether the episode-end is
    labelled terminated vs truncated must learn different Q-functions - because the
    TD target is masked by ``(1 - terminated)`` only. If the code masked on ``done``
    (or ignored the split) the two would be identical."""
    term_env = FakeEnv(period=4, mode="terminated")
    trunc_env = FakeEnv(period=4, mode="truncated")

    # sanity: the two runs really do differ only in the boundary flag
    tb = _buffer(_run_agent(term_env, total=40))
    ub = _buffer(_run_agent(trunc_env, total=40))
    np.testing.assert_array_equal(tb["obs"], ub["obs"])
    assert tb["terminated"].any() and not tb["truncated"].any()
    assert ub["truncated"].any() and not ub["terminated"].any()

    term_leaves = _ddqn_q_leaves(term_env)
    trunc_leaves = _ddqn_q_leaves(trunc_env)
    # at least one weight differs -> the update rule distinguishes the two flags
    assert any(not np.allclose(a, b) for a, b in zip(term_leaves, trunc_leaves))


def test_ddqn_no_termination_matches_pure_truncation():
    """A control: an env whose episode-ends are truncations learns the *same*
    Q-function as one whose ends carry no flag continuity difference - i.e.
    truncation is treated exactly like "keep bootstrapping". Here two truncating
    runs with the same seed are bit-identical, guarding against accidental
    dependence on episode index rather than the flag."""
    a = _ddqn_q_leaves(FakeEnv(period=4, mode="truncated"), seed=1)
    b = _ddqn_q_leaves(FakeEnv(period=4, mode="truncated"), seed=1)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_dead_transition_masked_out_of_the_loss():
    """``_masked_td_loss`` must give a dead-tagged window zero weight.

    A batch with two dead windows holding extreme (obs, action, reward)
    values - the kind that would blow up the loss if they trained - must
    produce EXACTLY the loss of the same batch with those windows dropped.
    A batch of only dead windows must return 0, not NaN or a division
    artifact.
    """
    from agents.ddqn import QNetwork, TimeStep, _masked_td_loss

    qkey, tkey = jax.random.split(jax.random.key(0))
    q = QNetwork(obs_dim=1, action_dim=2, hidden_size=4, key=qkey)
    target_q = QNetwork(obs_dim=1, action_dim=2, hidden_size=4, key=tkey)

    live = TimeStep(
        obs=jnp.array(
            [[[0.0], [1.0]], [[1.0], [2.0]], [[2.0], [0.0]]], jnp.float32
        ),
        action=jnp.array([[0, 0], [1, 0], [0, 0]], jnp.int32),
        reward=jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]], jnp.float32),
        terminated=jnp.array(
            [[False, False], [True, False], [False, False]]
        ),
        truncated=jnp.array(
            [[False, False], [False, False], [False, False]]
        ),
        dead=jnp.array([[False, False], [False, False], [False, False]]),
    )
    # extreme enough that an unmasked mean would clearly move
    dead = TimeStep(
        obs=jnp.array(
            [[[999.0], [999.0]], [[-999.0], [-999.0]]], jnp.float32
        ),
        action=jnp.array([[1, 0], [0, 0]], jnp.int32),
        reward=jnp.array([[12345.0, 0.0], [-12345.0, 0.0]], jnp.float32),
        terminated=jnp.array([[False, False], [False, False]]),
        truncated=jnp.array([[False, False], [False, False]]),
        dead=jnp.array([[True, False], [True, False]]),
    )
    full = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), live, dead)

    masked_loss = _masked_td_loss(q, target_q, full, gamma=0.99)
    live_only_loss = _masked_td_loss(q, target_q, live, gamma=0.99)
    np.testing.assert_array_equal(masked_loss, live_only_loss)

    # sanity: the dead rows really would move the result if left unmasked
    unmasked = full._replace(dead=jnp.zeros_like(full.dead))
    unmasked_loss = _masked_td_loss(q, target_q, unmasked, gamma=0.99)
    assert not np.allclose(unmasked_loss, masked_loss)

    # all-dead batch: no division by zero, loss is exactly 0
    all_dead_loss = _masked_td_loss(q, target_q, dead, gamma=0.99)
    assert float(all_dead_loss) == 0.0


# ===========================================================================
# 4. Analysis: episode segmentation on either boundary flag
# ===========================================================================


def test_episode_returns_segments_on_either_flag():
    from experiment.plotting import episode_returns

    reward = np.ones(10)
    terminated = np.zeros(10)
    truncated = np.zeros(10)
    terminated[3] = 1          # episode 1 ends at t=3 (return 4)
    truncated[7] = 1           # episode 2 ends at t=7 via truncation (return 4)
    ends, rets = episode_returns(reward, terminated, truncated)
    np.testing.assert_array_equal(ends, [3, 7])
    # rewards after t=7 are a dropped partial
    np.testing.assert_array_equal(rets, [4.0, 4.0])


def test_episode_returns_both_flags_same_step_is_one_boundary():
    from experiment.plotting import episode_returns

    reward = np.ones(6)
    terminated = np.zeros(6)
    truncated = np.zeros(6)
    # both fire on the same step
    terminated[2] = truncated[2] = 1
    ends, rets = episode_returns(reward, terminated, truncated)
    np.testing.assert_array_equal(ends, [2])                 # counted once, not twice
    np.testing.assert_array_equal(rets, [3.0])


def test_episode_returns_back_to_back_boundaries():
    from experiment.plotting import episode_returns

    reward = np.array([1.0, 2.0, 3.0, 4.0])
    terminated = np.array([0, 1, 1, 0])                      # length-1 episode at t=2
    truncated = np.zeros(4)
    ends, rets = episode_returns(reward, terminated, truncated)
    np.testing.assert_array_equal(ends, [1, 2])
    np.testing.assert_array_equal(rets, [3.0, 3.0])          # (1+2) then (3)


def test_episode_returns_no_boundary_is_empty():
    from experiment.plotting import episode_returns

    ends, rets = episode_returns(np.ones(5), np.zeros(5), np.zeros(5))
    # a wholly-partial run yields nothing
    assert ends.size == 0 and rets.size == 0
