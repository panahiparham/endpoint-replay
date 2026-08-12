# atari_50m

Atari **Pong**, 12.5M steps (50M frames), one component:

* `ddqn_pong` - Double DQN on the hyperparameters of
  `qrc-at-scale/experiments/atari-20m/Pong/dqn.json` (see `config.py` for the
  field-by-field mapping). Uses the `nature_cnn` Q-network. 2 seeds.

Uses ale-py's XLA env. No sweep.

## Setup

Atari needs the optional `atari` extra with ale-py's PR-#707 XLA build:

```bash
./scripts/install_ale_wheel.sh          # macOS-CPU / Linux-CPU
./scripts/install_ale_wheel.sh --cuda   # Linux-CUDA
```

(See the repo README "Atari (XLA) setup".)

## Scale ⚠️

This is **cluster-scale** - 12.5M agent steps, i.e. 50M frames at `FRAMESKIP=4`.
`ddqn_pong`'s `BUFFER_SIZE` is 100k rather than the json's 1M: obs+next_obs at
(84,84,4) uint8 would need ~56GB at 1M, more than a Vulcan L40S's 48GB (see
`FUTURE.md`). It is meant for **Linux-CUDA**, not a laptop. Atari's ale-py env can't be `jax.vmap`'d,
so the component sets `vmappable=False` (the harness runs `main` per seed) and
`shard_size=1` (one env + buffer per process); spread work across processes with
`--num-workers`.

Also note: on **macOS-CPU** the ale-py XLA FFI can intermittently segfault at
episode boundaries under the DDQN graph (stable on Linux-CUDA).

## Usage

```bash
uv run python experiments/atari_50m/run.py plan
uv run python experiments/atari_50m/run.py sweep --num-workers 2   # 2 seeds, one process each

# Quick local check (tiny run - override the heavy hypers):
uv run python experiments/atari_50m/run.py single --component ddqn_pong --seeds 0 \
    --AGENT-HYPERS.TOTAL-TIMESTEPS 500 --AGENT-HYPERS.BUFFER-SIZE 1000
```

> Per-hyper CLI overrides in `single` mode go through tyro on the selected
> component's config. Editing `config.py` avoids the CLI entirely.

## Analysis

Results store per-timestep `reward`/`terminated`/`truncated`/`loss`/`epsilon`
curves; `analysis.ipynb` plots `ddqn_pong`'s 2 seeds as individual episodic-return
curves.
