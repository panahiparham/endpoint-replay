"""Q-networks, replay transition type, and runner state shared by the agents.

The value-based agents in this package differ only in their TD target, so the
Equinox Q-network presets (``mlp`` / ``nature_cnn``), the buffer's transition
type and the scan carry all live here rather than in any one agent.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp


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


def _nature_flat_dim(obs_shape: tuple[int, ...]) -> int:
    """Flattened size after the Nature-DQN conv stack, computed host-side."""
    h, w = obs_shape[0], obs_shape[1]
    for kernel, stride in ((8, 4), (4, 2), (3, 1)):
        h = (h - kernel) // stride + 1
        w = (w - kernel) // stride + 1
    return 64 * h * w


class NatureCNN(eqx.Module):
    """Nature-DQN conv torso + 512 head for channel-last image observations."""

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
        self.head = eqx.nn.Linear(_nature_flat_dim(obs_shape), 512, key=k4)
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


def polyak_update(target: QNetwork, online: QNetwork, tau: float) -> QNetwork:
    """Blend the online network into the target network.

    Args:
        target: The target network to update.
        online: The online network to blend in.
        tau: Blend weight; ``1.0`` is a hard copy of ``online``.

    Returns:
        The updated target network.
    """
    online_arr, online_other = eqx.partition(online, eqx.is_array)
    target_arr, _ = eqx.partition(target, eqx.is_array)
    new_arr = jax.tree.map(
        lambda t, o: tau * o + (1.0 - tau) * t, target_arr, online_arr
    )
    return eqx.combine(new_arr, online_other)


__all__ = [
    "TimeStep",
    "QNetwork",
    "NatureCNN",
    "RunnerState",
    "AgentTrainOutput",
    "polyak_update",
]
