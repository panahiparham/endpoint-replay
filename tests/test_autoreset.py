"""Tests for AutoresetNextStep wrapper: DISABLED->NEXT_STEP mode conversion.

The wrapper must:
- Track pending reset state from the previous step's boundary flags.
- On a dead step (when pending=True), ignore the action and return a fresh
  reset obs with reward=0.0, terminated=False, truncated=False.
- This contract must hold under plain Python, jax.jit, and jax.vmap.
- A fresh reset() must never start pending.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from environments.autoreset import AutoresetNextStep


# --- fake env (GymEnv protocol; single float obs, sentinel reset) -----------


class _Box:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Discrete:
    def __init__(self, n):
        self.n = int(n)


class _State(NamedTuple):
    counter: jax.Array


class SimpleEnv:
    """Single-obs env that terminates every ``period`` steps.

    Obs is the (1-based) in-episode step counter as a float, distinct from
    the reset sentinel. Mode picks terminated vs truncated for boundaries.
    Like DISABLED-mode envs in this repo, it never resets itself - the caller
    must call reset explicitly.
    """

    RESET_OBS = -1.0

    def __init__(self, period: int = 3, mode: str = "terminated"):
        assert mode in ("terminated", "truncated")
        self._period = int(period)
        self._mode = mode

    def observation_space(self, params=None):
        return _Box((1,), jnp.float32)

    def action_space(self, params=None):
        return _Discrete(10)

    def reset(self, key, params=None):
        obs = jnp.asarray([self.RESET_OBS], jnp.float32)
        return obs, _State(jnp.asarray(0, jnp.int32))

    def step(self, key, state, action, params=None):
        del key, action, params
        nc = state.counter + 1
        done = (nc % self._period) == 0
        obs = nc.astype(jnp.float32).reshape((1,))
        reward = jnp.asarray(1.0, jnp.float32)
        terminated = done & (self._mode == "terminated")
        truncated = done & (self._mode == "truncated")
        return obs, _State(nc), reward, terminated, truncated, {}


# --- helpers: rollout and state inspection -----------------------------------


def _step_n_times(
    env: AutoresetNextStep, n: int, key: jax.Array, action: int | jax.Array = 0
) -> tuple[list, list, list, list, list]:
    """Rollout n steps from reset, return [obs, rewards, terminated, truncated, states]."""
    obs, state = env.reset(key)
    obs_list = [obs]
    r_list = []
    term_list = []
    trunc_list = []
    state_list = [state]

    for i in range(n):
        key = jax.random.fold_in(key, i)
        obs, state, r, term, trunc, _ = env.step(key, state, action)
        obs_list.append(obs)
        r_list.append(r)
        term_list.append(term)
        trunc_list.append(trunc)
        state_list.append(state)

    return obs_list, r_list, term_list, trunc_list, state_list


# ===========================================================================
# 1. Plain Python: boundary + dead-step contract
# ===========================================================================


def test_boundary_returns_true_final_obs_and_flag():
    """At a boundary step, the wrapper returns the true final obs with the
    appropriate flag set (not a reset sentinel)."""
    inner = SimpleEnv(period=3, mode="terminated")
    wrapped = AutoresetNextStep(inner)

    obs_list, _, term_list, trunc_list, _ = _step_n_times(
        wrapped, n=5, key=jax.random.key(0)
    )

    # period=3, so boundaries at steps 3, 6, ... (1-indexed) = indices 3, 6 in obs_list
    # step 1,2,3 -> obs=[1, 2, 3], step 4,5,6 -> obs=[1, 2, 3]
    assert float(obs_list[3][0]) == 3.0  # true boundary obs, step 3
    assert bool(term_list[2]) is True     # terminated at step 3 (index 2 in term_list)
    assert bool(trunc_list[2]) is False   # not truncated


def test_dead_step_ignores_action():
    """The step immediately after a boundary ignores the provided action and
    returns a fresh reset obs with reward=0 and both flags false, regardless
    of the action value."""
    inner = SimpleEnv(period=2, mode="terminated")
    wrapped = AutoresetNextStep(inner)

    # Rollout 1: action=0 on the dead step
    obs1, state1 = wrapped.reset(jax.random.key(0))
    # Step 1: counter 0->1, obs=1.0, not done
    obs1, state1, r1, term1, trunc1, _ = wrapped.step(
        jax.random.key(1), state1, action=jnp.int32(0)
    )
    # Step 2: counter 1->2, obs=2.0, done (boundary)
    obs1_boundary, state1_boundary, r1_b, term1_b, trunc1_b, _ = wrapped.step(
        jax.random.key(2), state1, action=jnp.int32(0)
    )
    # Step 3: dead step (pending=True from boundary)
    obs1_dead, state1_dead, r1_dead, term1_dead, trunc1_dead, _ = wrapped.step(
        jax.random.key(3), state1_boundary, action=jnp.int32(0)
    )

    # Rollout 2: action=99 on the dead step (clearly different)
    obs2, state2 = wrapped.reset(jax.random.key(0))
    # Step 1: counter 0->1, obs=1.0, not done
    obs2, state2, r2, term2, trunc2, _ = wrapped.step(
        jax.random.key(1), state2, action=jnp.int32(99)
    )
    # Step 2: counter 1->2, obs=2.0, done (boundary)
    obs2_boundary, state2_boundary, r2_b, term2_b, trunc2_b, _ = wrapped.step(
        jax.random.key(2), state2, action=jnp.int32(99)
    )
    # Step 3: dead step (pending=True from boundary)
    obs2_dead, state2_dead, r2_dead, term2_dead, trunc2_dead, _ = wrapped.step(
        jax.random.key(3), state2_boundary, action=jnp.int32(99)
    )

    # Same starting state (after boundary) -> same dead-step results regardless of action
    np.testing.assert_array_equal(obs1_dead, obs2_dead)
    assert float(r1_dead) == float(r2_dead)
    assert bool(term1_dead) == bool(term2_dead)
    assert bool(trunc1_dead) == bool(trunc2_dead)

    # Dead step must return reset obs with reward 0 and both flags false
    np.testing.assert_array_equal(obs1_dead, [SimpleEnv.RESET_OBS])
    assert float(r1_dead) == 0.0
    assert bool(term1_dead) is False
    assert bool(trunc1_dead) is False


# ===========================================================================
# 2. JAX JIT: boundary contract across a scanned rollout
# ===========================================================================


def test_jit_rollout_past_two_boundaries():
    """Wrap a reset + lax.scan-over-step rollout in jax.jit, run past 2
    boundaries, and assert the contract holds: boundary step returns true final
    obs + flag, next step returns fresh obs + reward 0 + both flags false."""
    inner = SimpleEnv(period=3, mode="terminated")
    wrapped = AutoresetNextStep(inner)

    def rollout(key):
        """Reset and step 8 times via lax.scan."""
        obs, state = wrapped.reset(key)

        def step_fn(carry, i):
            obs, state, key = carry
            key = jax.random.fold_in(key, i)
            obs, state, r, term, trunc, _ = wrapped.step(key, state, jnp.int32(0))
            return (obs, state, key), (obs, r, term, trunc)

        (obs, state, key), (obs_seq, r_seq, term_seq, trunc_seq) = lax.scan(
            step_fn, (obs, state, key), lax.iota(jnp.int32, 8)
        )
        return obs_seq, r_seq, term_seq, trunc_seq

    jitted_rollout = jax.jit(rollout)
    obs_seq, r_seq, term_seq, trunc_seq = jitted_rollout(jax.random.key(0))

    # Convert to numpy for inspection
    obs_seq = np.asarray(obs_seq)
    r_seq = np.asarray(r_seq)
    term_seq = np.asarray(term_seq)
    trunc_seq = np.asarray(trunc_seq)

    # With period=3:
    # Step 1: counter 0->1, obs=1.0
    # Step 2: counter 1->2, obs=2.0
    # Step 3: counter 2->3, obs=3.0, boundary (index 2)
    # Step 4: dead step, reset (index 3)
    # Step 5: counter 0->1, obs=1.0
    # Step 6: counter 1->2, obs=2.0
    # Step 7: counter 2->3, obs=3.0, boundary (index 6)
    # Step 8: dead step, reset (index 7)

    # First boundary at index 2 (step 3): obs should be 3.0
    assert float(obs_seq[2, 0]) == 3.0
    assert bool(term_seq[2]) is True
    assert bool(trunc_seq[2]) is False

    # Dead step after first boundary at index 3:
    # obs should be reset (-1.0), reward 0, both flags false
    assert float(obs_seq[3, 0]) == SimpleEnv.RESET_OBS
    assert float(r_seq[3]) == 0.0
    assert bool(term_seq[3]) is False
    assert bool(trunc_seq[3]) is False

    # Second boundary at index 6 (step 7): obs should be 3.0
    assert float(obs_seq[6, 0]) == 3.0
    assert bool(term_seq[6]) is True
    assert bool(trunc_seq[6]) is False

    # Dead step after second boundary at index 7:
    assert float(obs_seq[7, 0]) == SimpleEnv.RESET_OBS
    assert float(r_seq[7]) == 0.0
    assert bool(term_seq[7]) is False
    assert bool(trunc_seq[7]) is False


# ===========================================================================
# 3. JAX vmap: contract across vmapped batch
# ===========================================================================


def test_vmap_batch_contract_at_boundaries():
    """vmap the wrapper's reset and step over a batch of PRNG seeds.
    All seeds share the same env period so boundaries are deterministic.
    Run past a boundary and assert the contract holds for every seed in the
    batch: boundary step returns true final obs + flag, next step returns fresh
    obs + reward 0 + both flags false."""
    batch_size = 4
    inner = SimpleEnv(period=3, mode="terminated")
    wrapped = AutoresetNextStep(inner)

    def single_rollout(key):
        """Rollout from reset, taking n_steps via lax.scan."""
        obs, state = wrapped.reset(key)

        def step_fn(carry, i):
            obs, state, key = carry
            key = jax.random.fold_in(key, i)
            obs, state, r, term, trunc, _ = wrapped.step(key, state, jnp.int32(0))
            return (obs, state, key), (obs, r, term, trunc)

        (obs, state, key), (obs_seq, r_seq, term_seq, trunc_seq) = lax.scan(
            step_fn, (obs, state, key), jnp.arange(5)
        )
        return obs_seq, r_seq, term_seq, trunc_seq

    # Vmap over batch of keys
    batch_keys = jax.random.split(jax.random.key(0), batch_size)
    vmapped_rollout = jax.vmap(single_rollout)
    obs_batch, r_batch, term_batch, trunc_batch = vmapped_rollout(batch_keys)

    # Convert to numpy
    obs_batch = np.asarray(obs_batch)       # (batch, n_steps, 1)
    r_batch = np.asarray(r_batch)           # (batch, n_steps)
    term_batch = np.asarray(term_batch)     # (batch, n_steps)
    trunc_batch = np.asarray(trunc_batch)   # (batch, n_steps)

    # For all seeds: boundary at index 2 (step 3)
    for seed_idx in range(batch_size):
        assert float(obs_batch[seed_idx, 2, 0]) == 3.0
        assert bool(term_batch[seed_idx, 2]) is True
        assert bool(trunc_batch[seed_idx, 2]) is False

        # Dead step at index 3 (step 4)
        assert float(obs_batch[seed_idx, 3, 0]) == SimpleEnv.RESET_OBS
        assert float(r_batch[seed_idx, 3]) == 0.0
        assert bool(term_batch[seed_idx, 3]) is False
        assert bool(trunc_batch[seed_idx, 3]) is False


# ===========================================================================
# 4. Reset never starts pending
# ===========================================================================


def test_reset_never_starts_pending():
    """After env.reset(...), state.pending must be False so a fresh run's
    first step is never mistaken for a dead step."""
    inner = SimpleEnv(period=5, mode="terminated")
    wrapped = AutoresetNextStep(inner)

    obs, state = wrapped.reset(jax.random.key(0))
    assert state.pending is not None
    assert bool(state.pending) is False

    # Test with multiple resets (they should all start non-pending)
    for i in range(3):
        obs, state = wrapped.reset(jax.random.fold_in(jax.random.key(0), i))
        assert bool(state.pending) is False
