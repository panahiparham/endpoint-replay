"""
Define: DQN, DDQN, and Random Agent learning-rate sweeps across every environment
covered by the individual ``*_tuning`` experiments - Acrobot, Cartpole, Catch,
MountainCar, and Pinball's box/easy/empty/medium settings.

Each environment sweeps LR over its own range (10 seeds per agent/LR point, plus a
10-seed Random baseline), following that environment's own tuning recipe. See the
component-adding commits below for each environment's hypers and their provenance.
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
from environments.catch import CatchConfig
from environments.classic_control import (
    AcrobotConfig,
    CartpoleConfig,
    MountainCarConfig,
)
from environments.pinball import PinballConfig
from experiment import Component
from main import ExperimentConfig

LABEL = "tuning"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _components(
    env_key: str,
    env: str,
    learner: dict,
    env_hypers,
    lr_sweep: list[float],
    seeds: int = 10,
) -> list[Component]:
    """The dqn/ddqn/random trio for one environment, sweeping ``AGENT_HYPERS.LR``.

    Args:
        env_key: Suffix identifying this environment (and setting, for Pinball) in
            component names, e.g. ``"cartpole"`` or ``"pinball_box"``.
        env: The registry ``ENV`` name, e.g. ``"cartpole"`` or ``"pinball"``.
        learner: DQN/DDQN hypers shared by both agents, minus ``LR``.
        env_hypers: The environment's own hypers (e.g. ``CartpoleConfig``).
        lr_sweep: The learning rates to sweep for DQN and DDQN.
        seeds: Seeds per (agent, LR) point, and for the Random baseline.

    Returns:
        Three named components: ``dqn_<env_key>``, ``ddqn_<env_key>``,
        ``random_<env_key>``.
    """
    seed_list = list(range(seeds))
    return [
        Component(
            name=f"dqn_{env_key}",
            base=ExperimentConfig(
                AGENT="dqn", ENV=env,
                AGENT_HYPERS=DQNConfig(**learner), ENV_HYPERS=env_hypers,
            ),
            sweep={"AGENT_HYPERS.LR": lr_sweep},
            seeds=seed_list,
            shard_size=None,  # one vmap of every seed - the right unit of work on a GPU
        ),
        Component(
            name=f"ddqn_{env_key}",
            base=ExperimentConfig(
                AGENT="ddqn", ENV=env,
                AGENT_HYPERS=DDQNConfig(**learner), ENV_HYPERS=env_hypers,
            ),
            sweep={"AGENT_HYPERS.LR": lr_sweep},
            seeds=seed_list,
            shard_size=None,
        ),
        Component(
            name=f"random_{env_key}",
            base=ExperimentConfig(
                AGENT="random", ENV=env,
                AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=learner["TOTAL_TIMESTEPS"]),
                ENV_HYPERS=env_hypers,
            ),
            seeds=seed_list,
            shard_size=None,
        ),
    ]


# LR is swept per-component; every other hyper follows the classic_control
# Cartpole benchmark recipe.
CARTPOLE_LR_SWEEP = [4.0 ** -i for i in (2, 3, 4, 5, 6, 7, 8)]
_CARTPOLE_LEARNER = dict(
    TOTAL_TIMESTEPS=100_000,
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

# LR is swept per-component; every other hyper follows the classic_control
# Acrobot benchmark recipe.
ACROBOT_LR_SWEEP = [4.0 ** -i for i in (2, 3, 4, 5, 6, 7, 8)]
_ACROBOT_LEARNER = dict(
    TOTAL_TIMESTEPS=100_000,
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

# LR is swept per-component; every other hyper follows the classic_control
# MountainCar benchmark recipe.
MOUNTAINCAR_LR_SWEEP = [4.0 ** -i for i in (2, 3, 4, 5, 6, 7)]
_MOUNTAINCAR_LEARNER = dict(
    TOTAL_TIMESTEPS=100_000,
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

# LR is swept per-component; every other hyper follows the catch-jax README's
# recommended DQN hypers (NeverEndingRL suite): buffer 100k, batch 32, update every 4
# steps, hard target copy every 128, constant epsilon 0.01, gamma 0.9, a 2x32 MLP,
# 50k steps. LEARNING_STARTS is this repo's own default (not specified upstream).
CATCH_LR_SWEEP = [4.0 ** -i for i in (2, 3, 4, 5, 6, 7)]
_CATCH_LEARNER = dict(
    TOTAL_TIMESTEPS=50_000,
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
_CATCH_ENV_HYPERS = CatchConfig(
    ROWS=10, COLUMNS=5, SPAWN_PROBABILITY=0.1,
    EPISODE_CUTOFF=1_000_000_000,  # continuing task - never truncate within this run
)

# LR is swept per-component; every other hyper follows the pinball benchmark recipe
# (see experiments/pinball/config.py's _PINBALL_LEARNER).
PINBALL_LR_SWEEP = [4.0 ** -i for i in (2, 3, 4, 5, 6, 7)]
_PINBALL_LEARNER = dict(
    TOTAL_TIMESTEPS=100_000,
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

# Medium needed more training time than the other Pinball settings: 200k still
# looked needs-more-time, so this was upped again from 100k.
_PINBALL_MEDIUM_LEARNER = {**_PINBALL_LEARNER, "TOTAL_TIMESTEPS": 500_000}

COMPONENTS: list[Component] = [
    *_components(
        "cartpole", "cartpole", _CARTPOLE_LEARNER,
        CartpoleConfig(EPISODE_CUTOFF=500), CARTPOLE_LR_SWEEP,
    ),
    *_components(
        "acrobot", "acrobot", _ACROBOT_LEARNER,
        AcrobotConfig(EPISODE_CUTOFF=500), ACROBOT_LR_SWEEP,
    ),
    *_components(
        "mountaincar", "mountaincar", _MOUNTAINCAR_LEARNER,
        MountainCarConfig(EPISODE_CUTOFF=1_000), MOUNTAINCAR_LR_SWEEP,
    ),
    *_components(
        "catch", "catch", _CATCH_LEARNER, _CATCH_ENV_HYPERS, CATCH_LR_SWEEP,
    ),
    *_components(
        "pinball_box", "pinball", _PINBALL_LEARNER,
        PinballConfig(SETTING="box", EPISODE_CUTOFF=1_000), PINBALL_LR_SWEEP,
    ),
    *_components(
        "pinball_easy", "pinball", _PINBALL_LEARNER,
        PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000), PINBALL_LR_SWEEP,
    ),
    *_components(
        "pinball_empty", "pinball", _PINBALL_LEARNER,
        PinballConfig(SETTING="empty", EPISODE_CUTOFF=1_000), PINBALL_LR_SWEEP,
    ),
    *_components(
        "pinball_medium", "pinball", _PINBALL_MEDIUM_LEARNER,
        PinballConfig(SETTING="medium", EPISODE_CUTOFF=1_000), PINBALL_LR_SWEEP,
    ),
]
