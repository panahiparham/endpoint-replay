"""
Define: DQN on Classic Control environments (MountainCar, Cartpole, Acrobot).
Default Hypers, 100 seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap the repo root so `import main` (repo-root module) resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.dqn import DQNConfig
from environments.classic_control import (
    AcrobotConfig,
    CartpoleConfig,
    MountainCarConfig,
)
from experiment import Component
from main import ExperimentConfig

LABEL = "classic_control"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _dqn_hypers(total_timesteps: int = 100_000) -> DQNConfig:
    return DQNConfig(
        TOTAL_TIMESTEPS=total_timesteps,
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


COMPONENTS = [
    Component(
        name="dqn_mountaincar",
        base=ExperimentConfig(
            AGENT="dqn",
            ENV="mountaincar",
            AGENT_HYPERS=_dqn_hypers(),
            ENV_HYPERS=MountainCarConfig(EPISODE_CUTOFF=1_000),
        ),
        seeds=list(range(100)),
        shard_size=None,  # one vmap of every seed - the right unit of work on a GPU
    ),
    Component(
        name="dqn_cartpole",
        base=ExperimentConfig(
            AGENT="dqn",
            ENV="cartpole",
            AGENT_HYPERS=_dqn_hypers(),
            ENV_HYPERS=CartpoleConfig(EPISODE_CUTOFF=500),
        ),
        seeds=list(range(100)),
        shard_size=None,
    ),
    Component(
        name="dqn_acrobot",
        base=ExperimentConfig(
            AGENT="dqn",
            ENV="acrobot",
            AGENT_HYPERS=_dqn_hypers(),
            ENV_HYPERS=AcrobotConfig(EPISODE_CUTOFF=500),
        ),
        seeds=list(range(100)),
        shard_size=None,
    ),
]
