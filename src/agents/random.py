"""Uniform-random agent (no buffer, no learning)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, TypedDict

import jax
import jax.numpy as jnp

from envs.gym_env import DiscreteActionSpace, GymEnv


@dataclass(frozen=True, kw_only=True)
class RandomConfig:
    TOTAL_TIMESTEPS: int = 100_000
    SEED: int = 0


class RunnerState(NamedTuple):
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class AgentTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def make_train(
    config: RandomConfig,
    env: GymEnv[DiscreteActionSpace],
    env_params: object | None = None,
) -> Callable[[jax.Array], AgentTrainOutput]:
    """Build the pure training function for the uniform-random agent.

    Args:
        config: Agent hyperparameters.
        env: The environment to act in, with a discrete action space.
        env_params: Environment parameters, passed through to ``env``.

    Returns:
        A callable mapping a PRNG key to the run's ``AgentTrainOutput``.
    """
    action_dim = env.action_space(env_params).n

    def train(rng: jax.Array) -> AgentTrainOutput:
        """Run one seed of the random agent and return its state and metrics."""
        rng, reset_key = jax.random.split(rng)
        obsv, env_state = env.reset(reset_key, env_params)

        def _update_step(
            runner_state: RunnerState, _: jax.Array
        ) -> tuple[RunnerState, dict[str, jax.Array]]:
            env_state, last_obs, rng = runner_state
            rng, action_key, step_key, reset_key = jax.random.split(rng, 4)
            action = jax.random.randint(action_key, (), 0, action_dim, dtype=jnp.int32)

            obsv, env_state, reward, terminated, truncated, info = env.step(
                step_key, env_state, action, env_params
            )

            # Conditional, not masked: a stateful env (ale-py Atari) cannot have reset
            # called on non-boundary steps, since it mutates the emulator and no
            # jnp.where can undo that.
            obsv, env_state = jax.lax.cond(
                terminated | truncated,
                lambda: env.reset(reset_key, env_params),
                lambda: (obsv, env_state),
            )

            return (
                RunnerState(env_state=env_state, last_obs=obsv, rng=rng),
                {"reward": reward, "terminated": terminated,
                 "truncated": truncated, **info},
            )

        runner_state, metrics = jax.lax.scan(
            _update_step,
            RunnerState(env_state=env_state, last_obs=obsv, rng=rng),
            jnp.arange(config.TOTAL_TIMESTEPS),
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
