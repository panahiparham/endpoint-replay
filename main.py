"""The single agent<->environment interaction loop for the whole project.

Every experiment runs its agent by calling :func:`main` - nothing else runs an
agent in an environment. ``main`` takes one :class:`ExperimentConfig` - the agent and
environment *names* plus their hyperparameters - and a PRNG key, and returns the
raw result pytree of one run. The agent/env names are resolved through the
registries in ``agents`` (:data:`agents.AGENTS`) and ``environments``
(:data:`environments.ENVIRONMENTS`). The experiment harness (``src/experiment``)
vmaps :func:`main` over seeds; run this file directly for a single tyro run.

Adding an agent or environment
------------------------------
Register its ``(config_cls, make_train)`` / ``(config_cls, build)`` under a name
in ``agents.AGENTS`` / ``environments.ENVIRONMENTS``. It then becomes selectable
by name here automatically (the ``*_HYPERS`` union types below are built from the
registries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import jax

from agents import AGENTS, get_config as get_agent_config
from environments import ENVIRONMENTS, get_config as get_env_config

_DEFAULT_AGENT = "ddqn"
_DEFAULT_ENV = "pinball"

# Union of every registered agent/env config type, so tyro can parse whichever
# one the config carries. Stays in sync with the registries automatically.
_AGENT_TYPES = tuple(spec.config_cls for spec in AGENTS.values())
_ENV_TYPES = tuple(spec.config_cls for spec in ENVIRONMENTS.values())
AgentHypers = Union[_AGENT_TYPES] if len(_AGENT_TYPES) > 1 else _AGENT_TYPES[0]
EnvHypers = Union[_ENV_TYPES] if len(_ENV_TYPES) > 1 else _ENV_TYPES[0]


@dataclass(frozen=True)
class ExperimentConfig:
    """One fully-specified run: agent + env selected by name, plus their hypers.

    ``AGENT`` / ``ENV`` are registry keys; ``AGENT_HYPERS`` / ``ENV_HYPERS`` hold
    the hyperparameters for the selected agent / env (keep them consistent with
    the chosen names).
    """

    AGENT: str = _DEFAULT_AGENT
    ENV: str = _DEFAULT_ENV
    AGENT_HYPERS: AgentHypers = field(
        default_factory=lambda: get_agent_config(_DEFAULT_AGENT)
    )
    ENV_HYPERS: EnvHypers = field(default_factory=lambda: get_env_config(_DEFAULT_ENV))


def build(config: ExperimentConfig) -> Callable[[jax.Array], dict]:
    """Construct the environment and agent named by a config.

    Args:
        config: The agent and environment names plus their hyperparameters.

    Returns:
        The pure training function, mapping a PRNG key to one run's result
        pytree. The harness jits and vmaps it (see ``experiment.core``).

    Raises:
        KeyError: If ``config.AGENT`` or ``config.ENV`` is not registered.
    """
    # Host-side, outside any jax.jit trace: ale-py's Atari env segfaults if its
    # AtariVectorEnv is constructed during tracing, and building here also costs
    # one construction per config rather than one per trace.
    env, env_params = ENVIRONMENTS[config.ENV].build(config.ENV_HYPERS)
    return AGENTS[config.AGENT].make_train(config.AGENT_HYPERS, env, env_params)


def main(config: ExperimentConfig, rng: jax.Array) -> dict:
    """Run one agent-environment interaction.

    Args:
        config: The agent and environment names plus their hyperparameters.
        rng: PRNG key seeding the run.

    Returns:
        The raw result pytree, with ``runner_state`` and ``metrics`` keys.
    """
    return build(config)(rng)


if __name__ == "__main__":
    import tyro

    config = tyro.cli(ExperimentConfig)
    out = jax.jit(build(config))(jax.random.key(0))
    metrics = out["metrics"]
    print(f"AGENT={config.AGENT} ENV={config.ENV} ({config.ENV_HYPERS})")
    print(f"mean reward:      {float(metrics['reward'].mean()):.3f}")
    print(f"terminated steps: {int(metrics['terminated'].sum())}")
    print(f"truncated steps:  {int(metrics['truncated'].sum())}")
    if "loss" in metrics:
        print(f"mean loss:        {float(metrics['loss'].mean()):.4f}")
    if "can_sample" in out:
        print(f"buffer ready:     {bool(out['can_sample'])}")
