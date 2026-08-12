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
        """Advance one timestep, without resetting on a boundary.

        Args:
            key: PRNG key for any stochastic transition.
            state: The current environment state.
            action: The action to take.
            params: Environment parameters, or ``None`` for the env's own.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``. ``obs`` is the
            true boundary observation on a terminal step; the caller resets.
        """
        ...
