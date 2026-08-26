"""Atari environment integration: config, builder, and env wrapper, backed by
ale-py's XLA vector env.

Fields on :class:`AtariConfig` follow the project-wide
``CAPITALIZED_WITH_UNDERSCORE`` hyper naming.

``ale_py.AtariVectorEnv(...).xla()`` is a *stateful, vectorised* FFI: it returns
``(init_handle, reset_fn, step_fn)`` where the emulator lives in C++ and ``handle``
(a ``(8,)`` uint8 array) is threaded through as the environment state.
:class:`AtariEnvLike` wraps it as a single-env (``num_envs == 1``), pure,
``jax.jit``/``lax.scan``-compatible ``reset``/``step`` matching
:mod:`environments.gym_env`.

ale-py's XLA path only supports ``AutoresetMode.NEXT_STEP``: on a terminal step
it returns the true final observation, and the *next* step ignores its action
and returns a fresh episode's observation with ``reward=0``, both flags
``False``. That is exactly this repo's NEXT_STEP contract (see
:mod:`environments.gym_env`), so :meth:`AtariEnvLike.step` is a straight
pass-through of ale's own tuple: no wrapper-owned truncation counting, no
``lax.cond``, no in-step reset call. :func:`build` hands ale the exact episode
cutoff (``EPISODE_CUTOFF * FRAMESKIP`` raw frames), so ale's own truncation is
exact in agent steps, and its dead step is ale's, not a wrapper fabrication -
both verified against ale-py 0.12.0 (PR #707) in ``tests/test_atari.py``.

**Layout.** ale hands back ``(num_envs=1, frames, H, W)`` uint8; we present it
channel-last ``(H, W, frames)`` (stacked frames become channels), which the
``atarinet`` Q-network normalises (``/255``) and consumes.

Not vmap-able. ``jax.vmap`` traces and compiles over the FFI custom call without
complaint, then fails inside ale at run time (``INTERNAL: Incorrect handle buffer
size in reset``). Run one env per process: ``experiment.Component`` requires
``vmappable=False`` for this env and rejects anything else.

Platform note: on macOS-CPU ale-py's XLA FFI could intermittently *segfault*
under a heavy graph like DQN, historically when the agent made a second
stateful call (a boundary-reset) downstream of a step in the same graph. The
agent no longer makes that call at all now that Atari's own NEXT_STEP dead
step supplies the reset, which should help rather than hurt. The real ale-py
DQN smoke test still isolates itself in a subprocess and skips on a crash,
since the underlying FFI fragility is not proven to be gone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class _Box:
    """Minimal image/box observation space (``.shape`` / ``.dtype``)."""

    def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
        self.shape = shape
        self.dtype = dtype


class _Discrete:
    """Minimal discrete action space (``.n``)."""

    def __init__(self, n: int) -> None:
        self.n = int(n)


class AtariState(NamedTuple):
    """Environment state: ale-py's opaque ``(8,)`` uint8 emulator handle."""

    handle: jax.Array


class AtariEnvLike:
    """Wrap an ``ale_py.AtariVectorEnv`` (num_envs == 1) as a ``GymEnv``.

    ale itself owns both the episode cutoff (given to it in :func:`build`) and
    the NEXT_STEP dead step; this class does no bookkeeping of its own.
    """

    def __init__(self, vector_env) -> None:
        if vector_env.num_envs != 1:
            raise ValueError(
                "AtariEnvLike supports a single environment, got "
                f"num_envs={vector_env.num_envs}."
            )
        self._num_envs = 1
        self._init_handle, self._reset_fn, self._step_fn = vector_env.xla()
        frames, height, width = vector_env.single_observation_space.shape
        self._obs_shape = (height, width, frames)  # channel-last
        self._n_actions = int(vector_env.single_action_space.n)

    def observation_space(self, params: object | None = None) -> _Box:
        """Return the channel-last ``(H, W, frames)`` uint8 observation space."""
        return _Box(self._obs_shape, jnp.uint8)

    def action_space(self, params: object | None = None) -> _Discrete:
        """Return the game's discrete action space."""
        return _Discrete(self._n_actions)

    def _to_hwc(self, obs: jax.Array) -> jax.Array:
        return jnp.transpose(obs[0], (1, 2, 0))  # (1, frames, H, W) -> (H, W, frames)

    def reset(
        self, key: jax.Array, params: object | None = None
    ) -> tuple[jax.Array, AtariState]:
        """Start a new episode, reseeding the emulator.

        Args:
            key: PRNG key, used only to draw ale's seed.
            params: Unused; present for the env protocol.

        Returns:
            An ``(obs, state)`` pair for the emulator's first observation.
        """
        seed = jax.random.randint(
            key, (1,), 0, jnp.iinfo(jnp.int32).max
        ).astype(jnp.int32)
        handle, (obs, _info) = self._reset_fn(self._init_handle, seed)
        return self._to_hwc(obs), AtariState(handle=handle)

    def step(
        self,
        key: jax.Array,
        state: AtariState,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, AtariState, jax.Array, jax.Array, jax.Array, dict]:
        """Advance one agent step, per ale's native NEXT_STEP contract.

        Args:
            key: Unused; ale keeps its own RNG, seeded at reset.
            state: The current state, holding ale's emulator handle.
            action: The action to take; ignored by ale on a dead step.
            params: Unused; present for the env protocol.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``, a
            pass-through of ale's own tuple.
        """
        del key  # ale keeps its own RNG; sticky actions add the stochasticity
        actions = jnp.asarray(action, dtype=jnp.int32).reshape((1,))
        handle, (obs, rewards, terminations, truncations, _info) = self._step_fn(
            state.handle, actions
        )
        reward = rewards[0].astype(jnp.float32)
        terminated = terminations[0]
        truncated = truncations[0]
        state = AtariState(handle=handle)
        return self._to_hwc(obs), state, reward, terminated, truncated, {}


def require_ale_xla(game: str) -> None:
    """Check that ale-py's XLA FFI and the game ROM are installed.

    Fails here, with a pointer to the install script, rather than deep inside
    ale-py later.

    Args:
        game: The ale-py game key, e.g. ``"pong"``.

    Raises:
        RuntimeError: If the XLA FFI build, its CUDA targets, or the ROM is
            missing.
    """
    import ale_py._ale_py as _c
    import ale_py.roms

    script = "scripts/install_ale_wheel.sh"
    if not hasattr(_c, "VectorXLAReset"):
        raise RuntimeError(
            "ale-py was installed without the XLA vector-env FFI (no VectorXLAReset). "
            f"Run `{script}` to install a build with XLA "
            "(see README 'Atari (XLA) setup')."
        )
    if jax.default_backend() != "cpu" and not hasattr(_c, "VectorXLAResetGPU"):
        raise RuntimeError(
            "jax is on GPU but ale-py has no CUDA XLA FFI targets "
            "(no VectorXLAResetGPU) - this is the PyPI build, not the PR #707 "
            f"one. Run `{script}` (and `uv sync --extra cuda`)."
        )
    try:
        rom = ale_py.roms.get_rom_path(game)
    except Exception:
        rom = None
    if rom is None or not os.path.exists(str(rom)):
        raise RuntimeError(
            f"Atari ROM '{game}' is not installed. The PR #707 wheels bundle all ROMs; "
            f"run `{script}`, or `AutoROM --accept-license` to fetch ROMs."
        )


@dataclass(frozen=True)
class AtariConfig:
    """Settings for an Atari game (ale-py XLA vector env, num_envs == 1)."""

    GAME: str = "pong"             # ale-py game key, e.g. "pong", "breakout"
    FRAMESKIP: int = 4             # emulator frames per agent step
    STICKY_ACTIONS: float = 0.25   # stochasticity -> ale repeat_action_probability
    EPISODE_CUTOFF: int = 27000    # max agent steps/episode (x FRAMESKIP = frames)


def build(config: AtariConfig):
    """Construct the Atari environment for a config.

    Args:
        config: The game and its emulator settings.

    Returns:
        An ``(env, env_params)`` pair; ``env_params`` is always ``None``.

    Raises:
        RuntimeError: If ale-py lacks the XLA FFI build or the game ROM.
    """
    import ale_py  # lazy: the base project installs without the atari extra

    require_ale_xla(config.GAME)

    kwargs = dict(
        game=config.GAME,
        num_envs=1,
        frameskip=int(config.FRAMESKIP),
        repeat_action_probability=float(config.STICKY_ACTIONS),
    )
    if config.EPISODE_CUTOFF and config.EPISODE_CUTOFF > 0:
        # EPISODE_CUTOFF is in agent steps; ale truncates on raw frames, so
        # scale by FRAMESKIP. ale owns both the cutoff and the NEXT_STEP dead
        # step that follows it - no wrapper-side bookkeeping needed.
        kwargs["max_num_frames_per_episode"] = (
            int(config.EPISODE_CUTOFF) * int(config.FRAMESKIP)
        )

    return AtariEnvLike(ale_py.AtariVectorEnv(**kwargs)), None
