"""Components for the ``atari_20m`` experiment (Atari Pong, 5M steps).

One component on Atari Pong:

* ``dqn_pong`` - DQN reproducing ``qrc-at-
scale/experiments/atari-20m/Pong/dqn.json`` (2 seeds).

The ``random_pong`` baseline is dropped for now (see git history to restore it) -
add it back once it has actually been run, so ``analysis.ipynb`` has both series'
data to overlay.

Atari's ale-py env is a stateful FFI that can't be ``jax.vmap``'d, so the component
sets ``vmappable=False`` (the harness loops ``main`` per seed) and ``shard_size=1``
(one ~GB-scale replay buffer per process).

⚠️ Faithful to the json this is **cluster-scale** (5M steps) - meant for Linux-CUDA,
not a laptop. ``dqn_pong``'s ``BUFFER_SIZE`` is 100k rather than the json's 1M: a
1M×(84,84,4) uint8 replay needs ~56GB obs+next_obs, more than a Vulcan L40S's 48GB
(see ``FUTURE.md``). Needs the ``atari`` extra; see ``scripts/install_ale_wheel.sh``.
For a quick local check, override on the CLI:
    uv run python experiments/atari_20m/run.py single --component dqn_pong \\
        --AGENT-HYPERS.TOTAL-TIMESTEPS 300 --AGENT-HYPERS.BUFFER-SIZE 1000 --seeds 0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap the repo root so `import main` (repo-root module) resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.dqn import DQNConfig
from environments.atari import AtariConfig
from experiment import Component
from main import ExperimentConfig

LABEL = "atari_20m"
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
                TOTAL_TIMESTEPS=5_000_000,       # json: TOTAL_TIMESTEPS
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
        seeds=[0, 1],
        shard_size=1,        # one env per process
        vmappable=False,     # ale-py Atari FFI is not vmap-able
    ),
]
