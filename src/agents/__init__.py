"""Agent registry: map a name string to its config class + ``make_train``.

``main`` selects the agent by ``ExperimentConfig.AGENT`` and looks it up here.
Register a new agent by adding its ``(config_cls, make_train)`` under a name.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from agents.ddqn import DDQNConfig, make_train as ddqn_make_train
from agents.dqn import DQNConfig, make_train as dqn_make_train
from agents.random import RandomConfig, make_train as random_make_train
from agents.random_buffered import (
    RandomBufferConfig,
    make_train as random_buffered_make_train,
)


class AgentSpec(NamedTuple):
    config_cls: type          # the agent's config dataclass
    make_train: Callable      # (config, env, env_params) -> (rng -> output pytree)


AGENTS: dict[str, AgentSpec] = {
    "random": AgentSpec(RandomConfig, random_make_train),
    "random_buffered": AgentSpec(RandomBufferConfig, random_buffered_make_train),
    "dqn": AgentSpec(DQNConfig, dqn_make_train),
    "ddqn": AgentSpec(DDQNConfig, ddqn_make_train),
}


def get_config(name: str):
    """Build the default config object for a registered agent.

    Args:
        name: A key of :data:`AGENTS`, e.g. ``"dqn"``.

    Returns:
        A new instance of that agent's config dataclass, with defaults.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; registered: {sorted(AGENTS)}")
    return AGENTS[name].config_cls()


__all__ = [
    "AgentSpec",
    "AGENTS",
    "get_config",
    "RandomConfig",
    "RandomBufferConfig",
    "DQNConfig",
    "DDQNConfig",
]
