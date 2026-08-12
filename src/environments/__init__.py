"""Environment registry: map a name string to its config class + builder.

``main`` selects the environment by ``ExperimentConfig.ENV`` and looks it up here.
Register a new environment by adding its ``(config_cls, build)`` under a name.

Each spec also records whether the environment survives ``jax.vmap``. A stateful
FFI env (ale-py Atari) does not: under ``vmap`` the agent's boundary-reset
``lax.cond`` gets a per-seed predicate and lowers to a ``select``, which runs the
reset branch on *every* step. ``experiment.Component`` reads this flag and rejects
a ``vmappable=True`` component on such an env.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from environments.atari import AtariConfig, build as build_atari
from environments.pinball import PinballConfig, build as build_pinball


class EnvSpec(NamedTuple):
    config_cls: type          # the environment's config dataclass
    build: Callable           # (config) -> (env, env_params)
    vmappable: bool = True    # False for a stateful env that cannot run under jax.vmap


ENVIRONMENTS: dict[str, EnvSpec] = {
    "pinball": EnvSpec(PinballConfig, build_pinball),
    "atari": EnvSpec(AtariConfig, build_atari, vmappable=False),   # stateful ale-py FFI
}


def get_config(name: str):
    """Build the default config object for a registered environment.

    Args:
        name: A key of :data:`ENVIRONMENTS`, e.g. ``"pinball"``.

    Returns:
        A new instance of that environment's config dataclass, with defaults.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in ENVIRONMENTS:
        raise ValueError(
            f"unknown environment {name!r}; registered: {sorted(ENVIRONMENTS)}"
        )
    return ENVIRONMENTS[name].config_cls()


__all__ = [
    "EnvSpec",
    "ENVIRONMENTS",
    "get_config",
    "PinballConfig",
    "AtariConfig",
]
