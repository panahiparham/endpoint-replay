"""
Define: the weekly benchmark suite - DQN on 5 environments, fixed hypers.

Pinned to a single, already-established recipe per environment (no sweep), 10
seeds each. Decoupled from experiments/ on purpose: this is what the weekly
Vulcan run recomputes from scratch every time (see src/experiment/schedule.py
and slurm.wipe()), so its hypers must stay stable across weeks rather than
drift with whatever experiments/ happens to be exploring.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap the repo root so `import main` (repo-root module) resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.dqn import DQNConfig
from environments.catch import CatchConfig
from environments.classic_control import (
    AcrobotConfig,
    CartpoleConfig,
    MountainCarConfig,
)
from environments.pinball import PinballConfig
from experiment import Component
from main import ExperimentConfig

LABEL = "bench_core"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEEDS = list(range(10))

# The classic_control recipe (experiments/classic_control/config.py's
# _dqn_hypers()), shared by Acrobot, MountainCar and Cartpole.
_CLASSIC_CONTROL = DQNConfig(
    TOTAL_TIMESTEPS=100_000,
    LR=0.001,
    BUFFER_SIZE=10_000,
    BATCH_SIZE=64,
    LEARNING_STARTS=1_000,
    TRAIN_FREQUENCY=1,
    TARGET_NETWORK_FREQUENCY=128,
    TAU=1.0,
    GAMMA=0.99,
    EPSILON_START=0.1,
    EPSILON_END=0.1,
    NETWORK_PRESET="mlp",
)

# The pinball (easy) recipe (experiments/pinball/config.py's _PINBALL_LEARNER).
_PINBALL_EASY = DQNConfig(
    TOTAL_TIMESTEPS=100_000,
    LR=0.002,
    BUFFER_SIZE=10_000,
    BATCH_SIZE=32,
    LEARNING_STARTS=1_000,
    TARGET_NETWORK_FREQUENCY=100,
    TAU=1.0,
    EPSILON_START=0.1,
    EPSILON_END=0.1,
    HIDDEN_SIZE=32,
    GAMMA=0.99,
    TRAIN_FREQUENCY=1,
)

# The catch-jax README's recommended DQN hypers (see experiments/catch_tuning's
# config.py for the field-by-field mapping); LR=0.001 since catch_tuning's LR
# sweep hasn't been run yet to pick a winner.
_CATCH = DQNConfig(
    TOTAL_TIMESTEPS=50_000,
    LR=0.001,
    BUFFER_SIZE=100_000,
    BATCH_SIZE=32,
    LEARNING_STARTS=1_000,
    TRAIN_FREQUENCY=4,
    TARGET_NETWORK_FREQUENCY=128,
    TAU=1.0,
    GAMMA=0.9,
    EPSILON_START=0.01,
    EPSILON_END=0.01,
    HIDDEN_SIZE=32,
    NETWORK_PRESET="mlp",
)

COMPONENTS = [
    Component(
        name="dqn_acrobot",
        base=ExperimentConfig(
            AGENT="dqn", ENV="acrobot",
            AGENT_HYPERS=_CLASSIC_CONTROL, ENV_HYPERS=AcrobotConfig(EPISODE_CUTOFF=500),
        ),
        seeds=SEEDS,
        shard_size=None,  # one vmap of every seed - the right unit of work on a GPU
    ),
    Component(
        name="dqn_mountaincar",
        base=ExperimentConfig(
            AGENT="dqn", ENV="mountaincar",
            AGENT_HYPERS=_CLASSIC_CONTROL,
            ENV_HYPERS=MountainCarConfig(EPISODE_CUTOFF=1_000),
        ),
        seeds=SEEDS,
        shard_size=None,
    ),
    Component(
        name="dqn_cartpole",
        base=ExperimentConfig(
            AGENT="dqn", ENV="cartpole",
            AGENT_HYPERS=_CLASSIC_CONTROL, ENV_HYPERS=CartpoleConfig(EPISODE_CUTOFF=500),
        ),
        seeds=SEEDS,
        shard_size=None,
    ),
    Component(
        name="dqn_pinball_easy",
        base=ExperimentConfig(
            AGENT="dqn", ENV="pinball",
            AGENT_HYPERS=_PINBALL_EASY,
            ENV_HYPERS=PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000),
        ),
        seeds=SEEDS,
        shard_size=None,
    ),
    Component(
        name="dqn_catch",
        base=ExperimentConfig(
            AGENT="dqn", ENV="catch",
            AGENT_HYPERS=_CATCH,
            ENV_HYPERS=CatchConfig(
                ROWS=10, COLUMNS=5, SPAWN_PROBABILITY=0.1, EPISODE_CUTOFF=1_000,
            ),
        ),
        seeds=SEEDS,
        shard_size=None,
    ),
]
