"""Components for the ``atari_200m`` experiment (Atari Pong, 50M steps, 1 seed).

The full-length counterpart of ``atari_20m``: the classic Nature-DQN protocol of
50M agent steps (200M emulator frames at ``FRAMESKIP=4``), instead of the shorter
20M-frame variant. Two components on Atari Pong, each saved to its own database so
``analysis.ipynb`` can overlay them:

* ``dqn_pong``    - DQN with the same hypers as ``atari_20m``'s ``dqn_pong``
  (itself reproducing ``qrc-at-scale/experiments/atari-20m/Pong/dqn.json``), just
  10x the steps.
* ``random_pong`` - the uniform-random agent, a return baseline.

Atari's ale-py env is a stateful FFI that can't be ``jax.vmap``'d, so both components
set ``vmappable=False`` (the harness loops ``main`` per seed) and ``shard_size=1`` (one
~GB-scale replay buffer per process).

⚠️ Cluster-scale, and longer than ``atari_20m`` by 10x - meant for Linux-CUDA, not a
laptop. ``dqn_pong``'s ``BUFFER_SIZE`` is 100k rather than the 1M a faithful
reproduction would use: a 1M×(84,84,4) uint8 replay needs ~56GB obs+next_obs, more
than a Vulcan L40S's 48GB (see ``FUTURE.md``). Needs the ``atari`` extra; see
``scripts/install_ale_wheel.sh``. For a quick local check, override on the CLI:
    uv run python experiments/atari_200m/run.py single --component dqn_pong \\
        AGENT-HYPERS:dqn-config --AGENT-HYPERS.TOTAL-TIMESTEPS 300 \\
        --AGENT-HYPERS.BUFFER-SIZE 1000 --seeds 0

This experiment has not been run - see the atari_20m PR (#7) for a 5M-step trial
run's measured throughput, which this experiment's SLURM settings extrapolate from.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap the repo root so `import main` (repo-root module) resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.dqn import DQNConfig
from agents.random import RandomConfig
from environments.atari import AtariConfig
from experiment import Component
from main import ExperimentConfig

LABEL = "atari_200m"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_ATARI_PONG = AtariConfig(
    GAME="pong",                     # json: environment_settings.game
    FRAMESKIP=4,                     # json: environment_settings.frameskip
    STICKY_ACTIONS=0.25,             # json: environment_settings.sticky_actions
    EPISODE_CUTOFF=27_000,           # json: EPISODE_CUTOFF (agent steps)
)

COMPONENTS = [
    Component(
        name="dqn_pong",
        base=ExperimentConfig(
            AGENT="dqn",
            ENV="atari",
            AGENT_HYPERS=DQNConfig(
                TOTAL_TIMESTEPS=50_000_000,      # 200M frames at FRAMESKIP=4
                LR=6.25e-05,                     # json: metaParameters.LR
                ADAM_EPS=1.5e-4,                 # json: metaParameters.ADAM_EPS
                # json BUFFER_SIZE is 1_000_000, but obs+next_obs at (84,84,4)
                # uint8 needs ~56GB, more than a Vulcan L40S's 48GB - shrunk
                # until the buffer stops duplicating obs/next_obs (see FUTURE.md).
                BUFFER_SIZE=100_000,
                BATCH_SIZE=32,                   # json: BATCH_SIZE
                LEARNING_STARTS=20_000,          # json: LEARNING_STARTS
                TRAIN_FREQUENCY=4,               # json: TRAIN_FREQUENCY
                TARGET_NETWORK_FREQUENCY=8_000,  # json: TARGET_NETWORK_FREQUENCY
                GAMMA=0.99,                      # json: GAMMA
                EPSILON_START=1.0,               # json: EPSILON_START
                EPSILON_END=0.01,                # json: EPSILON_END
                EPSILON_FRACTION=0.05,           # json: EPSILON_FRACTION
                NETWORK_PRESET="nature_cnn",     # json: NETWORK_PRESET
                # TAU not in the json -> default 1.0 (hard target copy).
            ),
            ENV_HYPERS=_ATARI_PONG,
        ),
        seeds=[0],
        shard_size=1,        # one env per process
        vmappable=False,     # ale-py Atari FFI is not vmap-able
    ),
    Component(
        name="random_pong",
        base=ExperimentConfig(
            AGENT="random",
            ENV="atari",
            AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=50_000_000),
            ENV_HYPERS=_ATARI_PONG,
        ),
        seeds=[0],
        shard_size=1,
        vmappable=False,
    ),
]
