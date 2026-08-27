"""
Double DQN implemented with Equinox Q-network and Flashbax replay buffer.


    a* = argmax_a Q_online(s', a)
    y  = r + γ Q_target(s', a*)   (masked by ``(1 - terminated)``)

Truncated episodes still bootstrap; on ``terminated | truncated`` the
environment autoresets on the following step (NEXT_STEP), fabricating a
"dead" transition that links two episodes. That transition is stored with
``TimeStep.dead=True`` and masked out of the loss, never the reset itself -
the agent does not reset anything.

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
    terminated: jax.Array
    truncated: jax.Array
    dead: jax.Array


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


def _epsilon_greedy_action(
    q_values: jax.Array, epsilon: jax.Array, action_dim: int, key: jax.Array
) -> jax.Array:
    """Sample one action from the epsilon-greedy policy over ``q_values``.

    Args:
        q_values: Action values for a single observation.
        epsilon: Probability of acting uniformly at random.
        action_dim: Number of discrete actions.
        key: PRNG key for the draw.

    Returns:
        The sampled action. Greedy mass is split evenly across every action
        tied for the maximum, so ties break uniformly at random rather than
        toward the lowest index.
    """
    is_max = q_values == jnp.max(q_values)
    greedy_probs = is_max / jnp.sum(is_max)
    probs = epsilon / action_dim + (1.0 - epsilon) * greedy_probs
    return jax.random.categorical(key, jnp.log(probs)).astype(jnp.int32)


def _masked_td_loss(
    q: QNetwork, target_q: QNetwork, batch: TimeStep, gamma: float
) -> jax.Array:
    """Double DQN n-step TD loss over a sampled batch, excluding dead windows.

    Args:
        q: The online network; ``q(obs)`` selects the bootstrap action.
        target_q: The target network; evaluates the selected action's value.
        batch: A batch of ``[B, N+1, ...]`` ``TimeStep`` windows, e.g. from
            ``buffer.sample(...).experience``.
        gamma: Discount factor.

    Returns:
        The mean squared TD error over ``batch``, excluding any window whose
        first transition has ``dead=True`` (a fabricated NEXT_STEP transition
        linking two episodes) from both the numerator and the averaging
        denominator. Zero if every window in ``batch`` is dead.
    """
    n_step = batch.reward.shape[1] - 1
    q_sa = jax.vmap(q)(batch.obs[:, 0])
    q_a = jnp.take_along_axis(q_sa, batch.action[:, 0, None], axis=-1).squeeze(-1)

    done = (batch.terminated | batch.truncated)[:, :-1]
    alive = jnp.cumprod(1.0 - done.astype(jnp.float32), axis=1)
    w = jnp.concatenate([jnp.ones_like(alive[:, :1]), alive[:, :-1]], axis=1)

    discount = gamma ** jnp.arange(n_step)
    G = jnp.sum(w * discount * batch.reward[:, :n_step], axis=1)
    n = jnp.sum(w, axis=1).astype(jnp.int32)

    n_bcast = n.reshape((-1,) + (1,) * (batch.obs.ndim - 1))
    boot_obs = jnp.take_along_axis(batch.obs, n_bcast, axis=1)[:, 0]
    term_cut = jnp.take_along_axis(
        batch.terminated[:, :n_step], (n - 1)[:, None], axis=1
    ).squeeze(-1)

    # Double DQN: online net selects, target net evaluates.
    next_online = jax.vmap(q)(boot_obs)
    next_actions = jnp.argmax(next_online, axis=-1)
    next_target = jax.vmap(target_q)(boot_obs)
    next_q = jnp.take_along_axis(
        next_target, next_actions[:, None], axis=-1
    ).squeeze(-1)
    target = G + (gamma**n) * next_q * (1.0 - term_cut.astype(jnp.float32))
    target = jax.lax.stop_gradient(target)
    # A dead window fabricates a link between two episodes and must never
    # train: mask it out rather than skip the add, since a per-sample
    # lax.cond on `dead` would also lower to a select and add it anyway.
    valid = ~batch.dead[:, 0]
    return jnp.sum(valid * jnp.square(q_a - target)) / jnp.maximum(
        jnp.sum(valid), 1
    )


class RunnerState(NamedTuple):
    q: eqx.Module
    target_q: eqx.Module
    opt_state: Any
    buffer_state: object
    env_state: object
    last_obs: jax.Array
    prev_done: jax.Array
    rng: jax.Array


class AgentTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


@dataclass(frozen=True, kw_only=True)
class DDQNConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    N_STEP: int = 1
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

    buffer = fbx.make_trajectory_buffer(
        add_batch_size=1,
        sample_batch_size=config.BATCH_SIZE,
        sample_sequence_length=config.N_STEP + 1,
        period=1,
        min_length_time_axis=max(config.BATCH_SIZE, config.N_STEP + 1),
        max_length_time_axis=config.BUFFER_SIZE,
    )
    optimizer = optax.adam(config.LR, eps=config.ADAM_EPS)

    def train(rng: jax.Array) -> AgentTrainOutput:
        zeros = jnp.zeros(obs_shape, dtype=obs_dtype)
        dummy_timestep = TimeStep(
            obs=zeros,
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
            dead=jnp.asarray(False),
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
             env_state, last_obs, prev_done, rng) = runner_state

            epsilon = jnp.maximum(
                config.EPSILON_END,
                config.EPSILON_START
                - (config.EPSILON_START - config.EPSILON_END)
                * (t / (config.TOTAL_TIMESTEPS * config.EPSILON_FRACTION)),
            )

            rng, action_key, step_key, train_key = jax.random.split(rng, 4)

            q_values = q(last_obs)
            action = _epsilon_greedy_action(
                q_values, epsilon, action_dim, action_key
            )

            obsv, env_state, reward, terminated, truncated, info = env.step(
                step_key, env_state, action, env_params
            )

            # The env autoresets on NEXT_STEP (see environments.gym_env): when
            # prev_done is True, this transition is the fabricated dead step
            # linking two episodes. It is stored (dead=True) so the buffer
            # stays a plain, uniform stream, then masked out of the loss below.
            buffer_state = buffer.add(
                buffer_state,
                jax.tree.map(
                    lambda x: x[None, None, ...],
                    TimeStep(
                        obs=last_obs,
                        action=action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        dead=prev_done,
                    ),
                ),
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
                    return _masked_td_loss(q, target_q, batch, config.GAMMA)

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
                "dead": prev_done,
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
                    prev_done=terminated | truncated,
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
            prev_done=jnp.asarray(False),
            rng=rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
