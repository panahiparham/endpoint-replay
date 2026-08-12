# Future work

Deferred improvements to the experiment harness (`src/experiment/core.py`) and the
core env/agent abstractions. Each item is written with enough detail to implement
later.

---

## 1. Dense sharding (filter pending *before* sharding)

### Today
`build_shards(configs, seeds, shard_size)` enumerates the **full** grid
(`for config: for seed-chunk`) - it never looks at what is already done. The
pending filter happens *inside* `run_shards`:

```python
pend = [s for s in shard.seeds if run_id(cfg, s) not in done]
```

Building shards over the full, done-independent grid is deliberate: it makes the
partition **identical in every worker**, so `shards[worker_index::num_workers]` is
guaranteed disjoint with zero coordination.

### Problem
On a **resume**, shards go sparse and uneven. Example: one config, 10 seeds,
`SHARD_SIZE=5`, 6 already done → the two shards carry 1 and 3 real runs instead of
a single full shard of 4. Workers can end up with little or no work, and the
`jax.vmap` batch is smaller than `SHARD_SIZE`.

### Proposed fix
Filter to pending **first**, then chunk each config's pending seeds into full
`SHARD_SIZE` shards (dropping configs with nothing pending, so no empty shards).

### Sharp edge to preserve correctness
Once the partition depends on `done`, **every worker must compute the same
`done`**, or `shards[w::N]` stops being disjoint (gaps / double-compute). Rules:

- Build the partition from a **worker-identical snapshot** = the consolidated
  `results.db` **only** (add a `_committed_run_ids(results_dir)` that reads
  `results.db`, not `parts/`). `results.db` is immutable during a launch -
  workers write per-worker `parts/`, and `consolidate` runs only at the barrier -
  so all workers see the same snapshot and produce the same dense partition.
- Keep a **defensive within-shard skip against the full done set**
  (`_existing_run_ids` = `results.db` + `parts/`) so an interrupted-then-resumed
  run still never redoes work; skipping only *removes* work, so it can't break
  disjointness.
- `INSERT OR IGNORE` in `_insert_run` stays the final backstop.

Only degraded case: relaunching a half-finished sweep **without consolidating
first** re-runs the un-consolidated part (safe, just wasteful) - avoided by the
local launcher's auto-consolidate or a manual `run.py consolidate`.

### Touch points
`build_shards`, `run_shards`, `pending_runs`, and the `plan` branch of
`run_experiment` in `src/experiment/core.py`; add `_committed_run_ids`.

---

## 2. vmap over hyperparameters (not just seeds)

### Today
`run_shards` does `jax.jit(jax.vmap(lambda k: main(cfg, k)))(keys)` - **one
compile per config**, vmapping over seeds only. A 100-config sweep = 100 compiles.

### When it's worth it
Batch several *configs* into one vmapped call when they differ only in **scalar
hypers that feed into math and don't change array shape or control flow**: `LR`,
`GAMMA`, `EPSILON_START/END`, `TAU`. Then sweeping N of them is 1 compile + 1
batched run. Biggest win on **GPU** and for **large scalar sweeps** (compile
amortization + parallel throughput).

### When it's impossible
Any hyper that changes **array shape** (`HIDDEN_SIZE`, `BUFFER_SIZE`,
`BATCH_SIZE`, network depth) or **trace structure / loop length**
(`TOTAL_TIMESTEPS` scan length, `SETTING` → different Pinball geometry,
`TARGET_NETWORK_FREQUENCY`/`TRAIN_FREQUENCY` if they alter control flow) must stay
static → separate compiles. So vmap-over-hypers only applies *within* a group of
configs that share all the static hypers.

### Change required (largest of the deferred items)
This is the `VMAPPABLE`/"blocks" machinery deliberately left out of the current
harness. To add:

1. Declare which hyper paths are vmappable (e.g. `{"AGENT_HYPERS.LR",
   "AGENT_HYPERS.GAMMA"}`).
2. Group configs into **blocks** that share identical *static* (non-vmappable)
   hypers; within a block, stack the vmappable hypers into arrays.
3. Redefine a shard as `(block, chunk of (hyper-point, seed) pairs)` and vmap
   `main` over `(seed_key, hyper_arrays)` instead of just `seed_key`.
4. **Rewrite each agent** so those hypers are consumed as *traced arrays*, not
   Python scalars - e.g. `optax.adam(LR)` and the epsilon schedule must accept a
   batched value. This touches the harness **and the agent** (`src/agents/ddqn.py`).

### Reference
The `gstar-flows` harness (`../gstar-flows/src/gstar_flows/experiment.py`)
implements exactly this pattern (`VMAPPABLE`, `group_blocks`, `Block`, stacked
`dyn` dict) and is the template to port from if/when this is picked up.

---

## 3. Should envs auto-reset instead? (evaluate; currently: no)

### Today
No env auto-resets. Every env (pinball, Atari) returns the **true boundary
observation** and the agent resets with `jax.lax.cond` on `terminated | truncated`.
One contract, no `auto_resets` flag, no branch in the agents. Atari was moved *to*
this convention in `8c48321` (kevroi/online-gsp#1) - before that it reset inside
`step`, which discarded the pre-truncation observation and corrupted every truncated
TD target.

### The constraint any auto-reset design must solve
At a boundary step the agent needs **two** observations for different jobs:

| Observation | Used for |
| --- | --- |
| true boundary obs `s_T` | the stored transition's `next_obs` (what a truncated target bootstraps from) |
| fresh episode obs `s_0'` | `last_obs`, i.e. the state the next action is chosen from |

`step` has one observation slot. Non-auto-reset envs deliver the second value through
`reset()` - that is the *whole* reason this convention works. An auto-resetting env has
only `step`, so it **must** add a second channel, and the channel choice is where the
cost lands:

- **`info` dict** - free-looking, but all three agents merge `**info` into their metrics,
  which are the `lax.scan` `ys` and get stacked over every timestep and written as
  curves. An `(84,84,4)` uint8 obs there is ~141 GB for the 5M-step `random_pong`
  component. Correct only if *every* agent remembers to strip the key; the leak is
  prevented by discipline, not by construction. Rejected once already.
- **A field in the env state** - the scan *carry*, so one observation (28 KB), constant
  in `T`, and structurally unable to reach metrics. Needs a protocol accessor
  (`final_observation(state)`) paired with the `auto_resets` flag.
- **A 7th element of the `step` tuple** - cleanest typing, same memory profile, but needs
  a new adapter around third-party `pinball_jax` (returns a 6-tuple, not ours to change)
  plus all agents and every fake env in the tests.

### When it would actually be worth doing
Only if we go **vectorised** (`num_envs > 1`). With a batch of sub-envs ending on
different steps, "the agent resets on done" stops being expressible as one `lax.cond`
over the whole state, and a masked `env.reset` cannot work for a stateful env at all -
per-sub-env auto-reset inside the env becomes the natural design, as it is in
gymnasium/envpool/ale. Secondary, weaker motives: matching the ecosystem's default, and
never-ending rollouts with no boundary branch in the agent.

Not worth it for the current single-env setup: it would reintroduce the exact
two-observation problem the current convention dissolves, in exchange for nothing.

### If it is picked up
Prefer the **env-state channel** or the **7-tuple**, never `info`. Keep DQN's
`(1 - terminated)` masking as-is; the trap is `next_obs`, not the target. And note that
`terminated` transitions hide the bug - their bootstrap is masked, so only *truncated*
ones show corruption, which is why this went unnoticed until the audit.

### Touch points
`src/envs/gym_env.py` (the `GymEnv` protocol + any accessor), both envs
(`envs/atari.py`, `environments/pinball.py` would need a wrapper),
the agent (`src/agents/ddqn.py` reset path), and
`tests/test_episode_boundaries.py` + `tests/test_atari.py`, whose contract statements and
fake envs both assert the current no-auto-reset convention.

---

## 4. Memory-efficient Atari replay buffer (restore `BUFFER_SIZE=1M`)

### Today
`agents/ddqn.py`'s Flashbax item buffer stores each `TimeStep`'s `obs` **and**
`next_obs` as full `(84,84,4)` uint8 arrays, and nothing in the codebase places
the buffer off the default (GPU, when CUDA is present) device. At the json's
faithful `BUFFER_SIZE=1_000_000` that is ~56 GB (28 GB obs + 28 GB next_obs),
which does not fit on a Vulcan L40S (48 GB). `experiments/atari_50m`'s
`ddqn_pong` currently runs at `BUFFER_SIZE=100_000` (~5.3 GB) instead.

### Proposed fix
Store each frame once and reconstruct `obs`/`next_obs` by indexing a shared
ring buffer of frames (the standard approach in most Atari DQN
implementations), which roughly halves memory instead of duplicating every
transition. An off-GPU (host-resident) buffer is a fallback if on-device
memory is still the constraint after that, but adds per-step transfer cost
and device-placement complexity inside the `jax.lax.scan` training loop.

### Touch points
`agents/ddqn.py` (buffer construction, `add`/`sample`), and
`experiments/atari_50m/config.py` (`BUFFER_SIZE` back to `1_000_000` once it
fits).

---

## 5. Pack multiple processes per GPU (better utilization)

### Today
`ddqn_pong` (`BUFFER_SIZE=100k`, `nature_cnn`, batch 32) was measured on a Vulcan
L40S over a full 5M-step run (job 453105, 154 `nvidia-smi` samples at 15s
intervals - see PR #7):

| metric | mean | max |
| --- | --- | --- |
| GPU memory used | 34,559 / 46,068 MiB (**75%**), constant | same |
| SM (compute) utilization | **~27%** | 38% |
| memory-bandwidth utilization | 2.4% | 3% |
| power draw | 132W of a 350W limit (**38%**) | 137.5W |

The 75%-memory figure is not actual demand - it is JAX's default
`XLA_PYTHON_CLIENT_PREALLOCATE=true` grabbing a fixed 75% of the device up front
regardless of what the process goes on to use. The real working set (the 100k-item
replay buffer at ~5.3GB, plus the small `nature_cnn` and its optimizer state, well
under 1GB) is a small fraction of that. One `ddqn_pong` process therefore reserves
most of the card's memory and roughly a quarter of its compute for its full ~68
minute run, and the harness's current per-Atari-component model (`shard_size=1`,
one SLURM task = one exclusive GPU) leaves the rest of the GPU idle throughout.

### Proposed fix
Run several `ddqn_pong`-scale processes concurrently on one physical GPU inside a
single SLURM allocation:

1. Cap each process's JAX memory footprint explicitly - either
   `XLA_PYTHON_CLIENT_PREALLOCATE=false` (allocate on demand) or `=true` with a
   small `XLA_PYTHON_CLIENT_MEM_FRACTION` (e.g. `0.15`, ~7GB on an L40S -
   comfortable headroom over the ~6GB measured actual need). Without this, two
   processes each grabbing the 75% default would already OOM the device.
2. Launch N such processes as siblings inside one SLURM task, all seeing the same
   GPU - the default `CUDA_VISIBLE_DEVICES` from one `--gpus-per-node=1`
   allocation is already shared by every process in that task, so nothing on the
   SLURM side has to change.
3. Pick N from the compute measurement, not just memory: ~27% mean SM utilization
   per process suggests **N=3** as a starting point (~80% aggregate, leaving
   headroom for scheduling/contention overhead), even though memory alone
   (3 x 7GB = 21GB of 46GB) would allow more.
4. Evaluate NVIDIA MPS (`nvidia-cuda-mps-control`) for the concurrent processes -
   default time-sliced GPU sharing context-switches between processes' kernels,
   which costs more overhead than MPS's shared-context model. MPS is
   user-launchable inside a single job (no root needed), but needs verifying it's
   permitted on Vulcan.

### Open question - validate empirically before committing to a scheme
This is a single-sample measurement (one job, one config). Before picking a
production N, actually run N packed processes and compare aggregate throughput
against N sequential solo runs - GPU-side contention (L2 cache pressure, memory
bandwidth, kernel launch overhead) could make each packed process meaningfully
slower than its solo throughput, in which case a smaller N (or none) wins. The
packing factor is also specific to this hyper config (`BUFFER_SIZE`, network,
batch size) - a bigger network or batch would use more of both axes and support
less packing.

### Touch points
Worker orchestration only, **not the experiment running core**:
`experiment.run_experiment`'s local `sweep --num-workers N` launcher and
`cluster.slurm.dispatch`'s per-task wrap command (`src/experiment/core.py`,
`src/experiment/slurm.py`) - e.g. a `--procs-per-gpu` knob that spawns that many
sibling subprocesses per task/worker, each with `XLA_PYTHON_CLIENT_MEM_FRACTION`
set accordingly, rather than assuming one worker == one exclusive GPU.
`cluster.toml` would gain a per-experiment `procs_per_gpu` (or similar) alongside
`gpus`.
