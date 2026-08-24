"""Tests for the Atari env wrapper and DDQN-on-image-obs.

Logic tests drive ``AtariEnvLike`` against a fake vector env reproducing ale-py's
``.xla()`` FFI contract (opaque ``(8,)`` handle, NEXT_STEP autoreset), and run the
``atarinet`` DDQN on a fake image env - all without ale-py, so they run anywhere.
The real ale-py tests self-skip when the XLA build is not installed.

Note: on macOS-CPU, ale-py's XLA FFI can intermittently *segfault* when the
episode-boundary reset consume runs under the DDQN graph (it is solid on
Linux-CUDA, where Atari training actually happens). The real DDQN smoke therefore
runs in a subprocess and skips on such a crash rather than taking down pytest.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))  # for `import main` / repo-root modules

from environments.atari import AtariEnvLike


class _FakeVectorEnv:
    """Mimics ``ale_py.AtariVectorEnv`` + ``.xla()`` (jittable, no ale-py).

    Terminates every ``period`` steps. Obs frames carry the step counter so a reset
    (value 0) is distinguishable. Models ale's NEXT_STEP autoreset via a pending
    flag in the handle: a terminal step sets it; the next ``step_fn`` returns a fresh
    obs.
    """

    def __init__(self, num_envs: int = 1, period: int = 3, frames: int = 4,
                 h: int = 84, w: int = 84, n: int = 6) -> None:
        self.num_envs = num_envs
        self._period, self._frames, self._h, self._w = period, frames, h, w
        self.single_observation_space = type("S", (), {"shape": (frames, h, w)})()
        self.single_action_space = type("A", (), {"n": n})()

    def xla(self):
        frames, h, w, period = self._frames, self._h, self._w, self._period

        def obs(val):
            return jnp.full((1, frames, h, w), val, jnp.uint8)

        init = jnp.zeros((8,), jnp.uint8)  # [0]=step count, [1]=pending-reset flag

        def reset_fn(handle, seed):
            return jnp.zeros((8,), jnp.uint8), (obs(0), {})

        def step_fn(handle, actions):
            is_reset = handle[1] > 0
            count = handle[0] + jnp.uint8(1)
            term = (count % period == 0) & ~is_reset
            new_handle = (
                handle.at[0].set(jnp.where(is_reset, jnp.uint8(0), count))
                .at[1].set(jnp.where(term, jnp.uint8(1), jnp.uint8(0)))
            )
            obs_val = jnp.where(is_reset, jnp.uint8(0), count).astype(jnp.uint8)
            return new_handle, (
                obs(obs_val),
                jnp.where(is_reset, 0.0, 1.0).reshape((1,)).astype(jnp.float32),
                term.reshape((1,)), jnp.zeros((1,), bool), {},
            )

        return init, reset_fn, step_fn


def test_spaces_and_dtype():
    env = AtariEnvLike(_FakeVectorEnv(n=6))
    assert env.observation_space().shape == (84, 84, 4)
    assert env.observation_space().dtype == jnp.uint8
    assert env.action_space().n == 6
    # no env in this repo auto-resets; the agent does
    assert not hasattr(env, "auto_resets")


def test_reset_and_step_shapes_and_transpose():
    env = AtariEnvLike(_FakeVectorEnv())
    obs, state = env.reset(jax.random.key(0))
    # frames -> channels
    assert obs.shape == (84, 84, 4) and obs.dtype == jnp.uint8
    obs2, state2, reward, terminated, truncated, info = env.step(
        jax.random.key(1), state, jnp.int32(0)
    )
    assert obs2.shape == (84, 84, 4)
    assert reward.shape == () and terminated.shape == () and truncated.shape == ()
    # separate flags, not merged
    assert bool(terminated) is False and bool(truncated) is False
    assert info == {}


def test_no_in_step_reset_returns_true_boundary_obs():
    """The wrapper does NOT absorb ale's autoreset: on a terminal step it returns the
    *true*
    terminal observation (counter 3), not the fresh episode's (0). This is what makes a
    boundary transition store the state the agent actually reached (issue #1)."""
    env = AtariEnvLike(_FakeVectorEnv(period=3))
    _, state = env.reset(jax.random.key(0))
    seen = []
    for _ in range(3):
        obs, state, _r, terminated, _tr, _i = env.step(
            jax.random.key(1), state, jnp.int32(0)
        )
        seen.append((int(obs[0, 0, 0]), bool(terminated)))
    # obs 3 = true terminal, not reset(0)
    assert seen == [(1, False), (2, False), (3, True)]


def test_wrapper_owns_episode_cutoff():
    """Truncation comes from the wrapper's own agent-step counter, not from ale: this
    fake
    never ends on its own, yet ``truncated`` fires exactly at ``episode_cutoff`` and the
    returned observation is the true pre-truncation one.

    The counter keeps climbing (so ``truncated`` stays set) until the *agent* resets -
    same contract as pinball, whose ``time``-based truncation behaves identically
    if an agent ignores ``done``."""
    # fake never ends on its own
    env = AtariEnvLike(_FakeVectorEnv(period=1000), episode_cutoff=4)
    _, state = env.reset(jax.random.key(0))
    assert int(state.t) == 0
    rows = []
    for n in range(5):
        obs, state, _r, term, trunc, _i = env.step(
            jax.random.key(n), state, jnp.int32(0)
        )
        rows.append((int(obs[0, 0, 0]), bool(term), bool(trunc), int(state.t)))
    # fires at 4, stays set
    assert [r[2] for r in rows] == [False, False, False, True, True]
    # a cutoff truncates, never terminates
    assert not any(r[1] for r in rows)
    # only env.reset clears the counter
    assert [r[3] for r in rows] == [1, 2, 3, 4, 5]
    # true pre-truncation obs, not a reset
    assert rows[3][0] == 4

    _, fresh = env.reset(jax.random.key(0))                # the agent's reset clears it
    assert int(fresh.t) == 0


def test_num_envs_gt_one_rejected():
    with pytest.raises(ValueError):
        AtariEnvLike(_FakeVectorEnv(num_envs=2))


def test_truncated_transition_stores_true_obs():
    """**Regression test for issue #1.** A *truncated* Atari transition must store the
    true
    pre-truncation frame as ``next_obs``, not the fresh episode's first frame. This is
    the
    case DDQN's ``(1 - terminated)`` mask does NOT hide: a truncated target bootstraps,
    so a
    reset observation here would corrupt it.

    Runs the real agent loop over the real wrapper (with a fake FFI), so it covers the
    whole
    chain: wrapper-owned cutoff -> no in-step reset -> agent's conditional reset ->
    buffer."""
    CUTOFF = 4
    # truncations only
    env = AtariEnvLike(_FakeVectorEnv(period=1000), episode_cutoff=CUTOFF)
    from agents.ddqn import DDQNConfig, make_train
    cfg = DDQNConfig(TOTAL_TIMESTEPS=9, BUFFER_SIZE=16, BATCH_SIZE=2,
                     LEARNING_STARTS=9,
                     # no training: pure buffer inspection
                     NETWORK_PRESET="atarinet")
    out = jax.block_until_ready(jax.jit(make_train(cfg, env, None))(jax.random.key(0)))

    bs = out["runner_state"].buffer_state
    n = int(np.asarray(bs.current_index))
    exp = bs.experience
    trunc = np.asarray(exp.truncated)[0][:n].astype(bool)
    term = np.asarray(exp.terminated)[0][:n].astype(bool)
# every pixel of a fake frame carries the step counter, so one pixel identifies the
# frame

    nxt = np.asarray(exp.next_obs)[0][:n].reshape(n, -1)[:, 0]
    obs = np.asarray(exp.obs)[0][:n].reshape(n, -1)[:, 0]

    ends = np.flatnonzero(trunc)
    # truncates at agent steps 4 and 8
    assert list(ends) == [CUTOFF - 1, 2 * CUTOFF - 1]
    # a cutoff truncates, never terminates
    assert not term.any()
    # the TRUE pre-truncation frame
    np.testing.assert_array_equal(nxt[ends], [CUTOFF, CUTOFF])
    # 0 == fresh-episode frame == the bug
    assert (nxt[ends] != 0).all()
    # next transition starts from the reset
    np.testing.assert_array_equal(obs[ends + 1], [0, 0])


# --- DDQN on image obs in jit (fake env; reliable, no ale-py) ----------------

class _FakeImageEnv:
    """A GymEnv with (84,84,4) uint8 obs, to exercise the ``atarinet`` DDQN (uint8
    replay,
    agent-side conditional reset) under ``jax.jit``. Like the real envs it returns the
    true
    terminal obs and leaves the reset to the agent."""

    def __init__(
        self, h: int = 84, w: int = 84, c: int = 4, n: int = 6, period: int = 7
    ) -> None:
        self._shape, self._n, self._period = (h, w, c), n, period

    def observation_space(self, params=None):
        return type("B", (), {"shape": self._shape, "dtype": jnp.uint8})()

    def action_space(self, params=None):
        return type("D", (), {"n": self._n})()

    def reset(self, key, params=None):
        return jnp.zeros(self._shape, jnp.uint8), jnp.int32(0)

    def step(self, key, state, action, params=None):
        t = state + 1
        term = (t % self._period == 0)
        # true obs, incl. terminal
        obs = jnp.full(self._shape, (t % 256).astype(jnp.uint8), jnp.uint8)
        return obs, t, jnp.float32(1.0), term, jnp.asarray(False), {}


def test_ddqn_nature_cnn_jit_smoke():
    from agents.ddqn import DDQNConfig, make_train

    env = _FakeImageEnv()
    train = make_train(
        DDQNConfig(TOTAL_TIMESTEPS=120, BUFFER_SIZE=200, BATCH_SIZE=8,
                   LEARNING_STARTS=10,
                   TRAIN_FREQUENCY=2, TARGET_NETWORK_FREQUENCY=20,
                   EPSILON_FRACTION=0.5, NETWORK_PRESET="atarinet"),
        env, None,
    )
    out = jax.jit(train)(jax.random.key(0))
    jax.block_until_ready(out)
    m = out["metrics"]
    assert m["reward"].shape == (120,)
    assert {"reward", "terminated", "truncated", "loss", "epsilon"} <= set(m)
    assert np.isfinite(np.asarray(m["loss"])).all()
    assert int(np.asarray(m["terminated"]).sum()) >= 1  # auto-reset episodes occurred


# --- the vmappable guard ----------------------------------------------------
#
# Atari's ale-py FFI is stateful, so a vmapped run resets the emulator every step
# (the boundary-reset cond gets a per-seed predicate and lowers to a select) and
# then dies inside ale with a message that names no cause. Component rejects it up
# front instead.

def _atari_config(**kw):
    from environments.atari import AtariConfig
    from main import ExperimentConfig

    return ExperimentConfig(AGENT="ddqn", ENV="atari", ENV_HYPERS=AtariConfig(), **kw)


def test_atari_component_rejects_vmappable():
    from experiment import Component

    with pytest.raises(ValueError, match="vmappable=False"):
        Component(name="ddqn_pong", base=_atari_config(), seeds=[0], shard_size=1)


def test_atari_component_accepts_vmappable_false():
    from experiment import Component

    comp = Component(name="ddqn_pong", base=_atari_config(), seeds=[0],
                     shard_size=1, vmappable=False)
    assert comp.vmappable is False


def test_atari_reached_only_through_a_sweep_is_rejected():
    from environments.pinball import PinballConfig
    from experiment import Component
    from main import ExperimentConfig

    base = ExperimentConfig(AGENT="ddqn", ENV="pinball", ENV_HYPERS=PinballConfig())
    with pytest.raises(ValueError, match="'atari'"):
        Component(name="mixed", base=base,
                  sweep={"ENV": ["pinball", "atari"]}, seeds=[0])


def test_vmappable_component_on_a_pure_env_is_allowed():
    from environments.pinball import PinballConfig
    from experiment import Component
    from main import ExperimentConfig

    base = ExperimentConfig(AGENT="ddqn", ENV="pinball", ENV_HYPERS=PinballConfig())
    assert Component(name="ddqn_pinball", base=base, seeds=[0, 1]).vmappable is True


# --- real ale-py XLA tests (skip if not installed) --------------------------

def _ale_xla_available() -> bool:
    try:
        import ale_py._ale_py as c
        return hasattr(c, "VectorXLAReset")
    except Exception:
        return False


ale_only = pytest.mark.skipif(
    not _ale_xla_available(), reason="ale-py XLA build not installed"
)


@ale_only
def test_atari_real_cutoff_is_exact_and_wrapper_owned():
    """End-to-end against real ale: ``EPISODE_CUTOFF`` is exact in *agent steps*, and
    the
    wrapper is what enforces it - ale's own limit is deliberately set one step later, so
    a
    truncation never triggers ale's autoreset (the autoreset is what discarded the true
    pre-truncation observation). If ale were still the one truncating, this would fire
    at
    ``CUTOFF + 1``."""
    from environments import ENVIRONMENTS
    from environments.atari import AtariConfig

    CUTOFF = 12
    env, _ = ENVIRONMENTS["atari"].build(
        AtariConfig(GAME="pong", EPISODE_CUTOFF=CUTOFF)
    )
    _, state = env.reset(jax.random.key(0))
    first_trunc = None
    for n in range(1, CUTOFF + 4):
        _obs, state, _r, term, trunc, _i = env.step(
            jax.random.key(n), state, jnp.int32(0)
        )
        # pong can't terminate this fast
        assert not bool(term)
        if bool(trunc):
            first_trunc = n
            break
    assert first_trunc == CUTOFF
    # wrapper's own counter, agent resets next
    assert int(state.t) == CUTOFF
    # reset works mid-episode (ale behaviour X)
    _, fresh = env.reset(jax.random.key(1))
    assert int(fresh.t) == 0


@ale_only
def test_atari_real_truncated_transition_is_not_the_reset_obs():
    """**Issue #1 end-to-end, against the real emulator.** Drive ``ddqn`` over real
    ale and check the buffer at a truncation.

    The bug's exact signature was ``next_obs`` of the truncated transition being *the
    same
    frame* as the next transition's ``obs`` - both were the fresh episode's first frame,
    because the wrapper reset in-step and the agent then didn't reset. Now the boundary
    keeps
    the true pre-truncation frame, so those two must differ."""
    from agents.ddqn import DDQNConfig, make_train
    from environments import ENVIRONMENTS
    from environments.atari import AtariConfig

    CUTOFF = 20
    env, params = ENVIRONMENTS["atari"].build(
        AtariConfig(GAME="pong", EPISODE_CUTOFF=CUTOFF)
    )
    # LEARNING_STARTS past the horizon: no training, pure buffer inspection.
    cfg = DDQNConfig(TOTAL_TIMESTEPS=45, BUFFER_SIZE=64, BATCH_SIZE=2,
                     LEARNING_STARTS=45, NETWORK_PRESET="atarinet")
    out = jax.block_until_ready(
        jax.jit(make_train(cfg, env, params))(jax.random.key(0))
    )

    bs = out["runner_state"].buffer_state
    n = int(np.asarray(bs.current_index))
    exp = bs.experience
    trunc = np.asarray(exp.truncated)[0][:n].astype(bool)
    nxt, obs = np.asarray(exp.next_obs)[0][:n], np.asarray(exp.obs)[0][:n]

    ends = np.flatnonzero(trunc)
    # exact cutoff, twice in 45 steps
    assert list(ends) == [CUTOFF - 1, 2 * CUTOFF - 1]
    for e in ends:
        assert not np.array_equal(nxt[e], obs[e + 1]), (
            "truncated next_obs equals the following transition's obs -> "
            "it is the fresh "
            "episode's first frame, i.e. the issue #1 bug")


@ale_only
def test_atari_env_real_jit_scan():
    """The real ale-py env reset/step run inside a jitted lax.scan."""
    from environments import ENVIRONMENTS
    from environments.atari import AtariConfig

    env, _ = ENVIRONMENTS["atari"].build(AtariConfig(GAME="pong", EPISODE_CUTOFF=1000))
    obs, state = env.reset(jax.random.key(0))
    assert obs.shape == (84, 84, 4) and obs.dtype == jnp.uint8

    @jax.jit
    def rollout(state, keys):
        def one(st, k):
            _o, st, reward, _tm, _tr, _i = env.step(k, st, jnp.int32(0))
            return st, reward
        return jax.lax.scan(one, state, keys)

    _, rewards = rollout(state, jax.random.split(jax.random.key(1), 40))
    jax.block_until_ready(rewards)
    assert rewards.shape == (40,)


_REAL_DDQN_SMOKE = """
import jax
from agents.ddqn import DDQNConfig, make_train
from environments import ENVIRONMENTS
from environments.atari import AtariConfig
env, p = ENVIRONMENTS["atari"].build(
    AtariConfig(GAME="pong", FRAMESKIP=4, STICKY_ACTIONS=0.25, EPISODE_CUTOFF=30))
train = make_train(DDQNConfig(TOTAL_TIMESTEPS=120, BUFFER_SIZE=200, BATCH_SIZE=8,
    LEARNING_STARTS=10, TRAIN_FREQUENCY=4, TARGET_NETWORK_FREQUENCY=30,
    EPSILON_FRACTION=0.5, NETWORK_PRESET="atarinet"), env, p)
out = jax.jit(train)(jax.random.key(0)); jax.block_until_ready(out)
assert out["metrics"]["reward"].shape == (120,)
print("SMOKE_OK")
"""


@ale_only
def test_ddqn_atari_real_smoke():
    """Real DDQN + ale-py Pong under jit, in a subprocess so the known macOS-CPU
    ale FFI segfault (at episode-boundary reset) skips instead of killing pytest."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(_REPO)}
    r = subprocess.run([sys.executable, "-c", _REAL_DDQN_SMOKE],
                       capture_output=True, text=True, env=env, cwd=str(_REPO))
    if r.returncode < 0 or "Bus error" in r.stderr or "Segmentation fault" in r.stderr:
        pytest.skip("ale-py XLA DDQN crashed on this platform "
                    f"(macOS-CPU FFI flakiness): rc={r.returncode}")
    assert r.returncode == 0, r.stderr[-800:]
    assert "SMOKE_OK" in r.stdout
