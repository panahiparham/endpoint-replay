"""Gymnasium-style environment Protocol for RL agent typing.

Agents accept the tuple-returning interface described here.
Single-env (non-vectorized). Observation/action spaces and
env params stay loosely typed until a concrete env needs more
structure.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp


class ObservationSpace(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> jnp.dtype: ...


class DiscreteActionSpace(Protocol):
    @property
    def n(self) -> int: ...


class ContinuousActionSpace(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


class GymEnv[ActionSpaceT](Protocol):
    def observation_space(self, params: object | None = None) -> ObservationSpace:
        """Return the observation space, whose ``shape`` and ``dtype`` are static."""
        ...

    def action_space(self, params: object | None = None) -> ActionSpaceT:
        """Return the action space."""
        ...

    def reset(
        self, key: jax.Array, params: object | None = None
    ) -> tuple[jax.Array, object]:
        """Start a new episode.

        Args:
            key: PRNG key for the initial state.
            params: Environment parameters, or ``None`` for the env's own.

        Returns:
            An ``(obs, state)`` pair.
        """
        ...

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[
        jax.Array, object, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]
    ]:
        """Advance one timestep with automatic episode resets.

        On an episode boundary (when ``terminated`` or ``truncated`` is true),
        this step returns the true final observation of that episode. The
        immediately following step is a "dead" step: it ignores the provided
        ``action``, returns the initial observation of the new episode, and
        reports ``reward=0.0`` with both flags false. The environment resets
        itself on the dead step; explicit ``reset`` calls are needed only at
        the very start of a run.

        Args:
            key: PRNG key for any stochastic transition.
            state: The current environment state.
            action: The action to take (ignored on dead steps).
            params: Environment parameters, or ``None`` for the env's own.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. On a
            boundary step, ``obs`` is the true final observation of the
            episode. On the following dead step, ``obs`` is the new episode's
            initial observation, ``reward`` is 0.0, and both flags are false.
        """
        ...
