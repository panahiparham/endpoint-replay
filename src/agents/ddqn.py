"""
Double DQN implemented with Equinox Q-network and Flashbax replay buffer.


    a* = argmax_a Q_online(s', a)
    y  = r + γ Q_target(s', a*)   (masked by ``(1 - terminated)``)

Truncated episodes still bootstrap; on ``terminated | truncated`` the agent
resets for the next step after storing the true boundary ``next_obs``.

The Q-network presets (``mlp`` / ``atarinet``), the buffer's transition type,
and the scan carry live here alongside the agent, since this is currently the
only value-based agent in the package. If a second one is added and needs the
same pieces, split them back out into a shared module.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, TypedDict

import equinox as eqx
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax

from environments.gym_env import DiscreteActionSpace, GymEnv


class TimeStep(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class QNetwork(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_size: int, key: jax.Array
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Linear(obs_dim, hidden_size, key=k1)
        self.layer2 = eqx.nn.Linear(hidden_size, hidden_size, key=k2)
        self.layer3 = eqx.nn.Linear(hidden_size, action_dim, key=k3)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        return self.layer3(x)


def _atarinet_flat_dim(obs_shape: tuple[int, ...]) -> int:
    """Flattened size after the Nature-DQN conv stack."""
    h, w = obs_shape[0], obs_shape[1]
    for kernel, stride in ((8, 4), (4, 2), (3, 1)):
        h = (h - kernel) // stride + 1
        w = (w - kernel) // stride + 1
    return 64 * h * w


class AtariNet(eqx.Module):
    """Nature-DQN conv net + linear head for q-values."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    head: eqx.nn.Linear
    out: eqx.nn.Linear

    def __init__(
        self, obs_shape: tuple[int, ...], action_dim: int, key: jax.Array
    ) -> None:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        channels = obs_shape[-1]
        self.conv1 = eqx.nn.Conv2d(channels, 32, kernel_size=8, stride=4, key=k1)
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=4, stride=2, key=k2)
        self.conv3 = eqx.nn.Conv2d(64, 64, kernel_size=3, stride=1, key=k3)
        self.head = eqx.nn.Linear(_atarinet_flat_dim(obs_shape), 512, key=k4)
        self.out = eqx.nn.Linear(512, action_dim, key=k5)

    def __call__(self, x: jax.Array) -> jax.Array:
        # (H,W,C) uint8 -> (C,H,W) float in [0,1]
        x = jnp.transpose(x, (2, 0, 1)).astype(jnp.float32) / 255.0
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = jax.nn.relu(self.head(jnp.ravel(x)))
        return self.out(x)


class RunnerState(NamedTuple):
    q: eqx.Module
    target_q: eqx.Module
    opt_state: Any
    buffer_state: object
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class AgentTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


@dataclass(frozen=True, kw_only=True)
class DDQNConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: int = 1
    TARGET_NETWORK_FREQUENCY: int = 1_000
    GAMMA: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_FRACTION: float = 0.5
    HIDDEN_SIZE: int = 64
    NETWORK_PRESET: str = "mlp"
    ADAM_EPS: float = 1e-8
    SEED: int = 42


def make_train(
    config: DDQNConfig,
    env: GymEnv[DiscreteActionSpace],
    env_params: object | None = None,
) -> Callable[[jax.Array], AgentTrainOutput]:
    action_dim = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    obs_dtype = env.observation_space(env_params).dtype
    obs_dim = math.prod(obs_shape)

    def _build_q(key: jax.Array) -> eqx.Module:
        match config.NETWORK_PRESET:
            case 'atarinet': return AtariNet(obs_shape, action_dim, key)
            case 'mlp': return QNetwork(obs_dim, action_dim, config.HIDDEN_SIZE, key)
            case _: raise ValueError(f"unknown NETWORK_PRESET {config.NETWORK_PRESET!r}")

    buffer = fbx.make_item_buffer(
        max_length=config.BUFFER_SIZE,
        min_length=config.BATCH_SIZE,
        sample_batch_size=config.BATCH_SIZE,
        add_sequences=False,
        add_batches=False,
    )
    optimizer = optax.adam(config.LR, eps=config.ADAM_EPS)

    def train(rng: jax.Array) -> AgentTrainOutput:
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

        rng, q_key, reset_key = jax.random.split(rng, 3)
        q = _build_q(q_key)
        target_q = q
        opt_state = optimizer.init(eqx.filter(q, eqx.is_array))

        obsv, env_state = env.reset(reset_key, env_params)

        def _update_step(
            runner_state: RunnerState, t: jax.Array
        ) -> tuple[RunnerState, dict[str, jax.Array]]:
            (q, target_q, opt_state, buffer_state,
             env_state, last_obs, rng) = runner_state

            epsilon = jnp.maximum(
                config.EPSILON_END,
                config.EPSILON_START
                - (config.EPSILON_START - config.EPSILON_END)
                * (t / (config.TOTAL_TIMESTEPS * config.EPSILON_FRACTION)),
            )

            (rng, action_key, explore_key,
             step_key, reset_key, train_key) = jax.random.split(rng, 6)

            q_values = q(last_obs)
            greedy_action = jnp.argmax(q_values).astype(jnp.int32) # TODO: add random tie breaking
            random_action = jax.random.randint(
                action_key, (), 0, action_dim, dtype=jnp.int32
            )
            explore = jax.random.uniform(explore_key, ()) < epsilon # TODO: maybe build the e-greedy policy and directly sample from it
            action = jnp.where(explore, random_action, greedy_action)

            obsv, env_state, reward, terminated, truncated, info = env.step(
                step_key, env_state, action, env_params
            )

            buffer_state = buffer.add( # TODO: this buffer is storing each obs twice,
                                        # we should add custom Flashbax flat buffers that store each obs once,
                                        # do not store duplicated frames, support n-step updates, and this would be where we build endpoint replay into DDQN
                                        # note endpoint will requiring storing or accessing next action as well as next state.
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
            obsv, env_state = jax.lax.cond( # TODO: this may be slowing us down. We should reconsider the env interface and find a way to avoid this cond.
                                            # One option is to match the interface of all envs to be
                terminated | truncated,
                lambda: env.reset(reset_key, env_params),
                lambda: (obsv, env_state),
            )

            def _do_train(
                q: QNetwork,
                target_q: QNetwork,
                opt_state: Any,
                buffer_state: object,
                key: jax.Array,
            ) -> tuple[QNetwork, Any, jax.Array]:
                batch = buffer.sample(buffer_state, key).experience

                def loss_fn(q: QNetwork) -> jax.Array:
                    q_sa = jax.vmap(q)(batch.obs)
                    q_a = jnp.take_along_axis(
                        q_sa, batch.action[:, None], axis=-1
                    ).squeeze(-1)
                    # Double DQN: online net selects, target net evaluates.
                    next_online = jax.vmap(q)(batch.next_obs)
                    next_actions = jnp.argmax(next_online, axis=-1)
                    next_target = jax.vmap(target_q)(batch.next_obs)
                    next_q = jnp.take_along_axis(
                        next_target, next_actions[:, None], axis=-1
                    ).squeeze(-1)
                    target = batch.reward + config.GAMMA * next_q * (
                        1.0 - batch.terminated.astype(jnp.float32)
                    )
                    return jnp.mean(jnp.square(q_a - jax.lax.stop_gradient(target)))

                loss, grads = eqx.filter_value_and_grad(loss_fn)(q)
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(q, eqx.is_array)
                )
                q = eqx.apply_updates(q, updates)
                return q, opt_state, loss

            # Keep free of per-seed data: a batched predicate makes vmap run this
            # on every step.
            can_train = (
                buffer.can_sample(buffer_state)
                & (t >= config.LEARNING_STARTS)
                & (t % config.TRAIN_FREQUENCY == 0)
            )

            q, opt_state, loss = jax.lax.cond(
                can_train,
                lambda: _do_train(q, target_q, opt_state, buffer_state, train_key),
                lambda: (q, opt_state, jnp.asarray(0.0)),
            )

            target_q = jax.lax.cond(
                t % config.TARGET_NETWORK_FREQUENCY == 0,
                lambda: q,
                lambda: target_q,
            )

            metrics = {
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "loss": loss,
                "epsilon": epsilon,
                **info,
            }
            return (
                RunnerState(
                    q=q,
                    target_q=target_q,
                    opt_state=opt_state,
                    buffer_state=buffer_state,
                    env_state=env_state,
                    last_obs=obsv,
                    rng=rng,
                ),
                metrics,
            )

        runner_state = RunnerState(
            q=q,
            target_q=target_q,
            opt_state=opt_state,
            buffer_state=buffer_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
