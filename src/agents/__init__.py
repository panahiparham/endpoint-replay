from collections.abc import Callable
from typing import NamedTuple

from agents.ddqn import DDQNConfig
from agents.ddqn import make_train as ddqn_make_train


class AgentSpec(NamedTuple):
    config_cls: type  # the agent's config dataclass
    make_train: Callable  # (config, env, env_params) -> (rng -> output pytree)


AGENTS: dict[str, AgentSpec] = {
    "ddqn": AgentSpec(DDQNConfig, ddqn_make_train),
}


def get_config(name: str):
    """Build the default config object for a registered agent."""
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; registered: {sorted(AGENTS)}")
    return AGENTS[name].config_cls()


__all__ = [
    "AGENTS",
    "AgentSpec",
    "DDQNConfig",
    "get_config",
]
