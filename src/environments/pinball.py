"""Pinball environment integration: config + builder.

Wraps ``pinball_jax`` behind a ``PinballConfig`` whose fields follow the
project-wide ``CAPITALIZED_WITH_UNDERSCORE`` hyper naming.
"""

from __future__ import annotations

from dataclasses import dataclass

from pinball_jax import Pinball, PinballParams

from environments.autoreset import AutoresetNextStep


@dataclass(frozen=True)
class PinballConfig:
    """Settings for the Pinball environment."""

    SETTING: str = "easy"          # bundled config: box/empty/easy/medium/hard
    EPISODE_CUTOFF: int = 1000     # max steps per episode (truncation)


def build(config: PinballConfig):
    """Construct the Pinball environment for a config.

    Args:
        config: The maze setting and episode cutoff.

    Returns:
        An ``(env, env_params)`` pair for the repo env protocol.
    """
    env = Pinball(config.SETTING)
    env = AutoresetNextStep(env)
    env_params = PinballParams(max_steps_in_episode=config.EPISODE_CUTOFF)
    return env, env_params
