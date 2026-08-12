"""
Define: DQN, DDQN, and Random Agent on Pinball(Easy) Environment.
Default Hypers, 30 seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap the repo root so `import main` (repo-root module) resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.ddqn import DDQNConfig
from agents.dqn import DQNConfig
from agents.random import RandomConfig
from environments.pinball import PinballConfig
from experiment import Component
from main import ExperimentConfig

LABEL = "pinball"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Shared learner hypers so dqn_easy / ddqn_easy are a fair comparison.
_PINBALL_LEARNER = dict(
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

COMPONENTS = [
    Component(
        name="random_easy",
        base=ExperimentConfig(
            AGENT="random",
            ENV="pinball",
            AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=100_000),
            ENV_HYPERS=PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000),
        ),
        seeds=list(range(30)),
        shard_size=None,  # random is cheap: one shard, all 30 seeds vmapped together
    ),
    Component(
        name="dqn_easy",
        base=ExperimentConfig(
            AGENT="dqn",
            ENV="pinball",
            AGENT_HYPERS=DQNConfig(**_PINBALL_LEARNER),
            ENV_HYPERS=PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000),
        ),
        seeds=list(range(30)),
        shard_size=5,  # 30 seeds / 5 = 6 shards
    ),
    Component(
        name="ddqn_easy",
        base=ExperimentConfig(
            AGENT="ddqn",
            ENV="pinball",
            AGENT_HYPERS=DDQNConfig(**_PINBALL_LEARNER),
            ENV_HYPERS=PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000),
        ),
        seeds=list(range(30)),
        shard_size=5,
    ),
]
