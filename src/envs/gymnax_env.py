"""Thin adapter from a ``gymnax`` env to this repo's :class:`~envs.gym_env.GymEnv`
protocol.

``gymnax`` envs are pure-JAX (jittable/vmappable) but speak a slightly different
interface: their step core ``step_env`` returns a single ``done`` (which folds in the
time limit) and a five-tuple ``(obs, state, reward, done, info)``, whereas this repo's
agents expect a six-tuple ``(obs, state, reward, terminated, truncated, info)`` with
*terminated* (a real environment terminal) and *truncated* (the episode-length cutoff)
split apart - agents bootstrap on truncation but not on termination.

:class:`GymnaxEnv` wraps ``step_env`` (the non-auto-resetting core, matching pinball and
Atari - every env in this repo leaves the boundary reset to the agent) and recovers the
split from the step counter: ``truncated = next_state.time >= max_steps_in_episode`` and
``terminated = done & ~truncated``. (A step that hits the goal on the very last allowed
timestep is labeled truncated rather than terminated - a negligible edge case.)

The observation/action spaces are gymnax's own (they already expose
``.shape``/``.dtype`` and ``.n``), and ``info`` is emptied to match the other envs
(whose agents merge ``info`` into the per-step metrics).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


class GymnaxEnv:
    """Adapts a constructed gymnax ``(env, env_params)`` pair to the repo protocol.

    Construct host-side via :meth:`make` (which resolves the gymnax name and applies
    the episode cutoff); the returned object's ``env_params`` is passed alongside it,
    exactly like the other envs' ``build`` functions.
    """

    def __init__(self, env: Any, env_params: Any) -> None:
        self._env = env
        self.env_params = env_params

    @classmethod
    def make(cls, name: str, episode_cutoff: int) -> tuple["GymnaxEnv", Any]:
        """Build a gymnax env by name and adapt it to the repo protocol.

        Args:
            name: A gymnax env id, e.g. ``"Acrobot-v1"``.
            episode_cutoff: Steps per episode, set as ``max_steps_in_episode``.

        Returns:
            The ``(env, env_params)`` pair the harness and agents expect.
        """
        import gymnax

        env, env_params = gymnax.make(name)
        env_params = env_params.replace(max_steps_in_episode=int(episode_cutoff))
        return cls(env, env_params), env_params

    def observation_space(self, params: object | None = None):
        """Return gymnax's own observation space."""
        return self._env.observation_space(
            self.env_params if params is None else params
        )

    def action_space(self, params: object | None = None):
        """Return gymnax's own action space."""
        return self._env.action_space(self.env_params if params is None else params)

    def reset(self, key: jax.Array, params: object | None = None):
        """Start a new episode and return ``(obs, state)``."""
        return self._env.reset(key, self.env_params if params is None else params)

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        """Advance one timestep, splitting gymnax's ``done`` into the two flags.

        Args:
            key: PRNG key for the transition.
            state: The current gymnax state.
            action: The action to take.
            params: Environment parameters, or ``None`` for the env's own.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``, with ``info``
            emptied to match the other envs.
        """
        params = self.env_params if params is None else params
        obs, next_state, reward, done, _info = self._env.step_env(
            key, state, action, params
        )
        truncated = next_state.time >= params.max_steps_in_episode
        terminated = jnp.logical_and(done, jnp.logical_not(truncated))
        return obs, next_state, reward, terminated, truncated, {}
