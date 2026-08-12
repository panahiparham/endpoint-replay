---
name: sanity-rlloop
description: "Run a short local sanity check of an agent-environment training loop in this repo: confirm the loop runs, transitions land in the replay buffer and the network updates from them, and the policy's actions actually reach the environment - then report observed metrics and a learning curve. TRIGGER - invoke when asked to sanity-check, smoke-test, or make sure an agent/env combination runs locally, or to run a short/local test of the training loop before a longer/cluster run."
---

# RL Loop Sanity Check

Verify, empirically and not just by reading the code, that an agent-environment
training loop works end to end on this machine: the loop runs, transitions are
stored and the network updates from them, and the actions actually sent to the
environment are the policy's.

## 0. Clarify before running

If not already given, ask one at a time and wait for an answer before the next:

- Which agent and which environment (by their registry names).
- How long to run (total steps). "A short run" is not a number - get one, or
  agree a rule (e.g. enough steps for a handful of completed episodes and a few
  hundred network updates).
- A memory ceiling for the replay buffer, if the agent has one (default 10GB).
- A short name for the run, used for the report filename.

Do not guess these silently - a wrong step count either produces a report with
no completed episodes in it, or wastes minutes on a slow environment for no
extra signal.

## 1. Explore before touching anything

Find, for this repo:

- The agent's implementation and its hyperparameters (buffer size, warmup /
  learning-starts, batch size, train frequency, target-network frequency).
- The environment's implementation and its settings (anything analogous to
  frame skip, sticky actions, episode cutoff, observation preprocessing).
- The config system, and any existing shipped config for this agent+environment
  pair. Prefer overriding a shipped config over inventing hypers from scratch -
  it keeps the run faithful to something already validated.
- The replay buffer implementation: what it stores, at what dtype/shape, and
  its resulting memory footprint at a given size. A common trap: a buffer that
  stores `obs` *and* `next_obs` as separate full-size fields needs roughly
  double the naive per-field estimate.
- The main entrypoint for a single run.
- Any existing tests that already exercise this agent+environment pair - reuse
  rather than duplicate them.
- Whether the agent even has a replay buffer / network to update. A baseline
  (e.g. a random-action agent) may not - skip that check rather than force it.

## 2. Size the run to the machine

If the agent has a replay buffer, compute its memory footprint at the
shipped/default size. If it exceeds the agreed ceiling, reduce buffer size
(and total steps if that alone isn't enough) until it fits, and state the
before/after numbers. Do not change anything else about the agent's objective
or algorithm - this step is about fitting on the machine, not tuning it.

If the agent has a warmup/learning-starts setting, set it small (e.g. 500) so
training updates start early instead of after tens of thousands of steps.

## 3. Instrument and run

Without modifying repo files, write a small standalone driver script (under a
scratch/tmp directory, not the repo) that:

- Calls the repo's real training entrypoint unmodified, with the sized-down
  hyperparameters from steps 1-2.
- Times build, compile, and execution separately.
- Tracks peak RAM and CPU time around the run (e.g. `resource.getrusage`).
- If there's a buffer: after the run, pulls its actual stored contents and
  diffs one field (e.g. reward) against the loop's own returned metrics, to
  prove the buffer holds the real interaction data rather than asserting it
  from source alone. Also compares the number of stored transitions to the
  number of steps taken.
- If there's a network: counts update steps against the count expected from
  (total steps, learning-starts, train-frequency).
- Extracts the action stream actually taken (from the buffer if there is one,
  otherwise from the metrics) and checks it: only valid actions, not constant,
  and its distribution shifts over the run - evidence the policy, not a stub,
  is driving it.
- Splits the reward stream on episode-boundary flags into per-episode
  return/length, keeping the timestep at which each episode ends.

Run it in a subprocess and check stderr for a crash signature (segfault/bus
error). Some environments have known platform-specific FFI flakiness - a crash
is a finding to report, not something to silently retry away.

## 4. Report

Keep it short. Structure:

1. **Config used** - one line per hyperparameter/setting changed from the
   shipped default, with why.
2. **Hypers used** - the full list of hyperparameters/settings the run
   actually used (agent and environment), values only - no explanation or
   justification.
3. **Checks** - pass/fail, each with the one number that proves it:
   - Loop runs to completion.
   - Transitions stored and network updates the expected number of times (mark
     N/A if the agent has no buffer/network).
   - Policy actions reach the environment (valid, non-constant, distribution
     shifts over the run).
4. **Metrics** - one small table: wall time (build/compile/run), steps/sec,
   CPU utilization, peak RAM, episodes completed, mean return.
5. **Learning curve** - raw return per completed episode plotted against the
   timestep at which the episode ended (not episode index). No rolling mean,
   no gridlines. Title it "<AGENT> on <ENVIRONMENT>" with the actual agent
   and environment names.

Post the report in the chat, and also write it to
`sanity-rlloop-<shortname>.md` in the repo root, with the learning curve saved
as a PNG next to it and embedded via a relative markdown image link.
