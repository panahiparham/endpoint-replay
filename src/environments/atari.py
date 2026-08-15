"""Atari environment integration: config, builder, and env wrapper, backed by
ale-py's XLA vector env.

Fields on :class:`AtariConfig` follow the project-wide
``CAPITALIZED_WITH_UNDERSCORE`` hyper naming.

``ale_py.AtariVectorEnv(...).xla()`` is a *stateful, vectorised* FFI: it returns
``(init_handle, reset_fn, step_fn)`` where the emulator lives in C++ and ``handle``
(a ``(8,)`` uint8 array) is threaded through as the environment state. :class:`AtariEnvLike`
wraps it as a single-env (``num_envs == 1``), pure, ``jax.jit``/``lax.scan``-compatible
``reset``/``step`` matching :mod:`environments.gym_env`.

Two adaptations matter:

* **No auto-reset.** ale-py's XLA path only supports ``AutoresetMode.NEXT_STEP`` - on
  a terminal step it returns the terminal obs, then resets on the *next* step (a "dead"
  step whose action is ignored). This wrapper does **not** absorb that reset. It returns
  the true boundary observation and leaves the reset to the agent, exactly like pinball
  as pinball, so a truncated transition's stored ``next_obs`` is the state the agent
  actually reached (issue #1). :meth:`AtariEnvLike.reset` clears any pending
  ``NEXT_STEP`` autoreset and works mid-episode - both verified against ale-py 0.12.0
  (PR #707) - so the dead step never reaches the agent. Agents must therefore reset on
  ``terminated | truncated``, and must do so *conditionally*: calling
  :meth:`AtariEnvLike.reset` on a non-boundary step mutates the emulator, and no
  ``jnp.where`` can undo that.
* **Episode cutoff.** The *wrapper* owns truncation, not ale: it counts agent steps in
  :class:`AtariState` and reports ``truncated`` at ``episode_cutoff``, while
  :func:`build` gives ale a limit one step *later*. So ale never sets ``done`` on a
  truncation, which is what would leave a pending autoreset mid-episode. ale's own
  ``truncations`` flag is OR-ed in as a backstop. (ale's limit is already exact in
  agent steps - it counts ``max_num_frames_per_episode / frameskip`` steps, excluding
  the random no-op start - so this buys the *ordering*, not accuracy.)
* **Layout.** ale hands back ``(num_envs=1, frames, H, W)`` uint8; we present it
  channel-last ``(H, W, frames)`` (stacked frames become channels), which the
  ``nature_cnn`` Q-network normalises (``/255``) and consumes.

Not vmap-able. ``jax.vmap`` traces and compiles over the FFI custom call without
complaint, then fails inside ale at run time (``INTERNAL: Incorrect handle buffer
size in reset``). Under ``vmap`` the agent's boundary-reset ``lax.cond`` also gets
a per-seed predicate and lowers to a ``select``, which would reset the emulator on
every step. Run one env per process: ``experiment.Component`` requires
``vmappable=False`` for this env and rejects anything else.

Platform note: on macOS-CPU ale-py's XLA FFI could intermittently *segfault* when
the in-step episode-boundary reset ran under a heavy graph like DQN. :meth:`AtariEnvLike.step`
no longer makes that second stateful call - the boundary reset moved to the agent,
so a step is one FFI call - and the full ale-py suite (including the DQN-Atari
smoke) now passes repeatedly on macOS-CPU. The smoke test still isolates itself in
a subprocess and skips on a crash, since the underlying FFI fragility is unproven
to be gone rather than merely less likely.
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
    """Environment state: ale-py's opaque ``(8,)`` uint8 emulator handle, plus the
    in-episode agent-step counter the wrapper truncates on."""

    handle: jax.Array
    t: jax.Array


class AtariEnvLike:
    """Wrap an ``ale_py.AtariVectorEnv`` (num_envs == 1) as a ``GymEnv``.

    ``episode_cutoff`` is the truncation limit in *agent steps*, enforced here rather
    than by ale (see the module docstring). ``None`` leaves truncation entirely to
    ale's own ``max_num_frames_per_episode``.
    """

    def __init__(self, vector_env, episode_cutoff: int | None = None) -> None:
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
        self._episode_cutoff = int(episode_cutoff) if episode_cutoff else None

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
            An ``(obs, state)`` pair with the in-episode step count reset to 0.
        """
        seed = jax.random.randint(
            key, (1,), 0, jnp.iinfo(jnp.int32).max
        ).astype(jnp.int32)
        handle, (obs, _info) = self._reset_fn(self._init_handle, seed)
        return self._to_hwc(obs), AtariState(handle=handle, t=jnp.asarray(0, jnp.int32))

    def step(
        self,
        key: jax.Array,
        state: AtariState,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, AtariState, jax.Array, jax.Array, jax.Array, dict]:
        """Advance one agent step, without resetting on a boundary.

        Args:
            key: Unused; ale keeps its own RNG, seeded at reset.
            state: The current state, holding ale's handle and the step count.
            action: The action to take.
            params: Unused; present for the env protocol.

        Returns:
            ``(obs, state, reward, terminated, truncated, info)``, where ``obs``
            is the true boundary observation on a terminal step.
        """
        del key  # ale keeps its own RNG; sticky actions add the stochasticity
        actions = jnp.asarray(action, dtype=jnp.int32).reshape((1,))
        t = state.t + jnp.asarray(1, jnp.int32)  # in-episode index, 1-based
        handle, (obs, rewards, terminations, truncations, _info) = self._step_fn(
            state.handle, actions
        )
        # Retained conservatively as the FFI-boundary guard: this used to separate two
        # stateful custom calls per step (step + reset-consume), a suspected cause of a
        # rare macOS-CPU segfault. Only one call per step remains, but the agent's
        # boundary reset is still a stateful call downstream of this handle.
        handle = jax.lax.optimization_barrier(handle)

        reward = rewards[0].astype(jnp.float32)
        terminated = terminations[0]
        truncated = truncations[0]
        if self._episode_cutoff is not None:
            # Fires one step *before* ale's own limit, so ale never sets done on a
            # truncation. ale's flag stays OR-ed in as a backstop.
            truncated = truncated | (t >= self._episode_cutoff)

        # No in-step reset: `obs` is the true boundary observation. The agent resets.
        state = AtariState(handle=handle, t=t)
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
    cutoff = (
        int(config.EPISODE_CUTOFF)
        if config.EPISODE_CUTOFF and config.EPISODE_CUTOFF > 0
        else None
    )
    if cutoff is not None:
        # EPISODE_CUTOFF is in agent steps; ale truncates on raw frames (exactly
        # max_num_frames_per_episode / FRAMESKIP agent steps, excluding the
        # no-op start).
        # Give ale one step MORE than the cutoff so the wrapper's own truncation always
        # fires first and ale never autoresets on a truncation -- that autoreset is what
        # would discard the true pre-truncation observation. ale stays as a backstop.
        kwargs["max_num_frames_per_episode"] = (cutoff + 1) * int(config.FRAMESKIP)

    return AtariEnvLike(ale_py.AtariVectorEnv(**kwargs), episode_cutoff=cutoff), None
