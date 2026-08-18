"""
Define: DDQN on Pinball(Easy) Environment.
Default Hypers, 30 seeds.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

# Bootstrap the repo root so `import main` (repo-root module) resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.ddqn import DDQNConfig
from environments.pinball import PinballConfig
from experiment import Component
from main import ExperimentConfig

LABEL = "pinball"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

pinball_condig = PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000)

ddqn_config = DDQNConfig(
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

seeds = list(range(30))
shard_size = 5  # 30 seeds / 5 = 6 shards


COMPONENTS = [
    Component(
        name="large",
        base=ExperimentConfig(
            AGENT="ddqn",
            ENV="pinball",
            AGENT_HYPERS=ddqn_config,
            ENV_HYPERS=pinball_condig,
        ),
        seeds=seeds,
        shard_size=shard_size
    ),
    Component(
        name="small-10%",
        base=ExperimentConfig(
            AGENT="ddqn",
            ENV="pinball",
            AGENT_HYPERS=dataclasses.replace(ddqn_config, BUFFER_SIZE=1_000),
            ENV_HYPERS=pinball_condig,
        ),
        seeds=seeds,
        shard_size=shard_size
    ),
    Component(
        name="small-5%",
        base=ExperimentConfig(
            AGENT="ddqn",
            ENV="pinball",
            AGENT_HYPERS=dataclasses.replace(ddqn_config, BUFFER_SIZE=500),
            ENV_HYPERS=pinball_condig,
        ),
        seeds=seeds,
        shard_size=shard_size
    ),
]
