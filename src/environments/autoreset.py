"""Generic NEXT_STEP wrapper for DISABLED-mode gym environments.

The wrapper converts a DISABLED-mode environment (where step returns the true
final observation on episode boundaries and expects the caller to reset
explicitly) into a NEXT_STEP-mode environment (where the immediate step
following a boundary automatically returns a fresh episode's initial
observation with reward=0.0 and both flags false).

Why jnp.where instead of lax.cond: Under vmap, a lax.cond with a per-example
traced predicate compiles to a select anyway (no actual branching), but reads
as if it might branch. Using jnp.where is honest about the select and makes
the vmap-safety obvious: each example's (obs, state, reward, flags) are
computed unconditionally for both branches, then selected. This is correct
regardless of whether the predicate is traced (under vmap) or concrete.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from environments.gym_env import GymEnv


class AutoresetState(NamedTuple):
    """Wrapper state: inner env state plus pending reset flag."""

    inner: Any
    pending: jax.Array


class AutoresetNextStep[ActionSpaceT]:
    """Wraps a DISABLED-mode env into NEXT_STEP-mode.

    A DISABLED-mode env returns the true final observation when the episode
    ends, but does not reset itself: the caller must call ``reset`` explicitly.
    This wrapper tracks whether the episode ended on the previous step, and on
    the very next step ignores the provided action and returns a fresh
    episode's initial observation with reward=0.0 and both flags false.
    """

    def __init__(self, env: GymEnv[ActionSpaceT]):
        """Initialize the wrapper with an inner environment.

        Args:
            env: A DISABLED-mode GymEnv to wrap.
        """
        self._env = env

    def observation_space(self, params: object | None = None):
        """Return the observation space, unchanged from the inner env."""
        return self._env.observation_space(params)

    def action_space(self, params: object | None = None) -> ActionSpaceT:
        """Return the action space, unchanged from the inner env."""
        return self._env.action_space(params)

    def reset(
        self, key: jax.Array, params: object | None = None
    ) -> tuple[jax.Array, AutoresetState]:
        """Start a new episode.

        Args:
            key: PRNG key for the initial state.
            params: Environment parameters, or ``None`` for the env's own.

        Returns:
            An ``(obs, state)`` pair; a fresh run never starts pending.
        """
        obs, inner_state = self._env.reset(key, params)
        return obs, AutoresetState(inner=inner_state, pending=jnp.asarray(False))

    def step(
        self,
        key: jax.Array,
        state: AutoresetState,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[
        jax.Array, AutoresetState, jax.Array, jax.Array, jax.Array,
        dict[str, jax.Array],
    ]:
        """Advance one timestep, autoresetting on the step after a boundary.

        Args:
            key: PRNG key for any stochastic transition or reset.
            state: The current wrapper state (inner state + pending flag).
            action: The action to take (ignored on a dead step).
            params: Environment parameters, or ``None`` for the env's own.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``, matching
            :meth:`environments.gym_env.GymEnv.step`'s NEXT_STEP contract.
        """
        reset_key, step_key = jax.random.split(key)
        obs_re, state_re = self._env.reset(reset_key, params)
        obs_st, state_st, r, term, trunc, info = self._env.step(
            step_key, state.inner, action, params
        )

        # `p`: whether the *previous* step ended, i.e. this step is the dead
        # one. Selected with jnp.where rather than lax.cond: under vmap a
        # per-example predicate lowers lax.cond to a select anyway, so
        # jnp.where is both correct and honest about the cost.
        p = state.pending
        obs = jnp.where(p, obs_re, obs_st)
        inner_state = jax.tree.map(lambda a, b: jnp.where(p, a, b), state_re, state_st)
        term, trunc = term & ~p, trunc & ~p
        reward = jnp.where(p, 0.0, r)

        return (
            obs,
            AutoresetState(inner=inner_state, pending=term | trunc),
            reward,
            term,
            trunc,
            info,
        )
