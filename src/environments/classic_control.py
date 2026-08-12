"""Classic-control environment integrations: config + builder for the three
``gymnax`` tasks used by the ``classic_control`` experiment.

Each config wraps a single ``gymnax`` env behind the project-wide
``CAPITALIZED_WITH_UNDERSCORE`` hyper naming and this repo's env protocol (via
:class:`envs.gymnax_env.GymnaxEnv`). The only hyper is ``EPISODE_CUTOFF`` (the
per-episode step limit → gymnax ``max_steps_in_episode``); the physics is fixed.
Defaults match the reference DQN experiments (MountainCar 1000, Cartpole/Acrobot 500).
"""

from __future__ import annotations

from dataclasses import dataclass

from envs.gymnax_env import GymnaxEnv


@dataclass(frozen=True)
class MountainCarConfig:
    """Settings for MountainCar-v0 (3 actions, 2-d obs, reward -1/step)."""

    EPISODE_CUTOFF: int = 1000     # max steps per episode (truncation)


@dataclass(frozen=True)
class CartpoleConfig:
    """Settings for the CartPole-v1 environment (2 actions, 4-d obs, reward +1/step)."""

    EPISODE_CUTOFF: int = 500      # max steps per episode (truncation)


@dataclass(frozen=True)
class AcrobotConfig:
    """Settings for the Acrobot-v1 environment (3 actions, 6-d obs, reward -1/step)."""

    EPISODE_CUTOFF: int = 500      # max steps per episode (truncation)


def build_mountaincar(config: MountainCarConfig):
    """Construct the MountainCar-v0 environment for a config.

    Args:
        config: The environment settings, currently the episode cutoff.

    Returns:
        An ``(env, env_params)`` pair for the repo env protocol.
    """
    return GymnaxEnv.make("MountainCar-v0", config.EPISODE_CUTOFF)


def build_cartpole(config: CartpoleConfig):
    """Construct the CartPole-v1 environment for a config.

    Args:
        config: The environment settings, currently the episode cutoff.

    Returns:
        An ``(env, env_params)`` pair for the repo env protocol.
    """
    return GymnaxEnv.make("CartPole-v1", config.EPISODE_CUTOFF)


def build_acrobot(config: AcrobotConfig):
    """Construct the Acrobot-v1 environment for a config.

    Args:
        config: The environment settings, currently the episode cutoff.

    Returns:
        An ``(env, env_params)`` pair for the repo env protocol.
    """
    return GymnaxEnv.make("Acrobot-v1", config.EPISODE_CUTOFF)
