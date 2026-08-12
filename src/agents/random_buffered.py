"""Uniform-random agent that stores transitions in a Flashbax item buffer.

Each buffer entry is a full transition ``(obs, action, reward, next_obs,
terminated, truncated)``. On ``terminated | truncated``, the agent calls
``env.reset`` for the next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, TypedDict

import flashbax as fbx
import jax
import jax.numpy as jnp

from envs.gym_env import DiscreteActionSpace, GymEnv


@dataclass(frozen=True, kw_only=True)
class RandomBufferConfig:
    TOTAL_TIMESTEPS: int = 100_000
    BUFFER_SIZE: int = 10_000
    BATCH_SIZE: int = 32
    SEED: int = 0


class TimeStep(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class RunnerState(NamedTuple):
    buffer_state: object
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class AgentTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]
    can_sample: jax.Array


def make_train(
    config: RandomBufferConfig,
    env: GymEnv[DiscreteActionSpace],
    env_params: object | None = None,
) -> Callable[[jax.Array], AgentTrainOutput]:
    """Build the pure training function for the buffered random agent.

    Acts uniformly at random and fills a replay buffer without learning from it,
    which exercises the buffer path on its own.

    Args:
        config: Agent hyperparameters.
        env: The environment to act in, with a discrete action space.
        env_params: Environment parameters, passed through to ``env``.

    Returns:
        A callable mapping a PRNG key to the run's ``AgentTrainOutput``.
    """
    action_dim = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    obs_dtype = env.observation_space(env_params).dtype

    buffer = fbx.make_item_buffer(
        max_length=config.BUFFER_SIZE,
        min_length=config.BATCH_SIZE,
        sample_batch_size=config.BATCH_SIZE,
        add_sequences=False,
        add_batches=False,
    )

    def train(rng: jax.Array) -> AgentTrainOutput:
        """Run one seed of the buffered random agent and return state + metrics."""
        zeros = jnp.zeros(obs_shape, dtype=obs_dtype)
        dummy_timestep = TimeStep(
            obs=zeros,
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            next_obs=zeros,
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
        )
        buffer_state = buffer.init(dummy_timestep)

        rng, reset_key = jax.random.split(rng)
        obsv, env_state = env.reset(reset_key, env_params)

        def _update_step(
            runner_state: RunnerState, t: jax.Array
        ) -> tuple[RunnerState, dict[str, jax.Array]]:
            del t
            buffer_state, env_state, last_obs, rng = runner_state

            rng, action_key, step_key, reset_key = jax.random.split(rng, 4)
            action = jax.random.randint(action_key, (), 0, action_dim, dtype=jnp.int32)

            obsv, env_state, reward, terminated, truncated, info = env.step(
                step_key, env_state, action, env_params
            )

            # Store the transition before any auto-reset replaces next_obs.
            buffer_state = buffer.add(
                buffer_state,
                TimeStep(
                    obs=last_obs,
                    action=action,
                    reward=reward,
                    next_obs=obsv,
                    terminated=terminated,
                    truncated=truncated,
                ),
            )

            # Reset for the next step, *after* the transition above captured the
            # true boundary obs as next_obs. Conditional, not masked: a stateful
            # env (ale-py Atari) cannot have reset called on non-boundary steps,
            # since it mutates the emulator and no jnp.where can undo that.
            obsv, env_state = jax.lax.cond(
                terminated | truncated,
                lambda: env.reset(reset_key, env_params),
                lambda: (obsv, env_state),
            )

            metrics = {
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                **info,
            }
            return (
                RunnerState(
                    buffer_state=buffer_state,
                    env_state=env_state,
                    last_obs=obsv,
                    rng=rng,
                ),
                metrics,
            )

        runner_state = RunnerState(
            buffer_state=buffer_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS)
        )
        can_sample = buffer.can_sample(runner_state.buffer_state)
        return {
            "runner_state": runner_state,
            "metrics": metrics,
            "can_sample": can_sample,
        }

    return train
