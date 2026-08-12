"""Catch environment integration: config + builder.

Wraps ``catch_jax`` behind a ``CatchConfig`` whose fields follow the
project-wide ``CAPITALIZED_WITH_UNDERSCORE`` hyper naming. ``catch_jax.Catch``
already speaks this repo's six-tuple ``GymEnv`` step protocol (separate
``terminated``/``truncated``), so no adapter is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from catch_jax import Catch, CatchParams


@dataclass(frozen=True)
class CatchConfig:
    """Settings for the Catch environment.

    Catch is a continuing task - the ball keeps spawning and there is no
    terminal state - so ``EPISODE_CUTOFF`` defaults high enough that no
    run of reasonable length ever truncates. Lower it explicitly (as the
    env tests do) to exercise the truncation mechanism itself.
    """

    ROWS: int = 10                 # grid height
    COLUMNS: int = 5               # grid width
    SPAWN_PROBABILITY: float = 0.1  # chance a new ball spawns each step
    EPISODE_CUTOFF: int = 1_000_000_000  # effectively unbounded (truncation)


def build(config: CatchConfig):
    """Construct the Catch environment for a config.

    Args:
        config: The grid size, spawn rate, and episode cutoff.

    Returns:
        An ``(env, env_params)`` pair for the repo env protocol.
    """
    env = Catch(rows=config.ROWS, columns=config.COLUMNS)
    env_params = CatchParams(
        spawn_probability=config.SPAWN_PROBABILITY,
        max_steps_in_episode=config.EPISODE_CUTOFF,
    )
    return env, env_params
