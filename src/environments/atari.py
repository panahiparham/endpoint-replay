"""Atari environment integration: config + builder over ale-py's XLA env.

Fields follow the project-wide ``CAPITALIZED_WITH_UNDERSCORE`` hyper naming. The
env wrapper (``envs.atari.AtariEnvLike``) is a single, stateful ale-py vector env
and is **not** ``jax.vmap``-able, so experiments using it must set
``VMAPPABLE = False`` (one env per process).
"""

from __future__ import annotations

from dataclasses import dataclass


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

    from envs.atari import AtariEnvLike, require_ale_xla

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
