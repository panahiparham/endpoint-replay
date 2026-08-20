# endpoint-replay agent instructions

## Work in a dedicated worktree, by default

The main worktree is a shared resource: other sessions are often concurrently
active there regardless of what any one task is about, so don't wait for a
task to look like "parallel work" before isolating it. By default, every
task - including a single, self-contained one - gets its own worktree and
branch. If the task doesn't name a branch, derive a short-name from what it
does (e.g. `pinball-variant`, `rainbow-agent`) and prefix it by convention
(`feat/`, `experiment/`, `fix/`):

```bash
git worktree add ../endpoint-replay-<short-name> -b <branch> main   # new branch
git worktree add ../endpoint-replay-<short-name> <branch>           # resuming one
```

Share one venv across worktrees instead of running `uv sync` in each:

```bash
MAIN_WORKTREE=$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')
SHARED_VENV="$MAIN_WORKTREE/.venvs/shared"   # gitignored
[ -x "$SHARED_VENV/bin/python" ] || UV_PROJECT_ENVIRONMENT="$SHARED_VENV" uv sync
[ -e "../endpoint-replay-<short-name>/.venv" ] || ln -s "$SHARED_VENV" ../endpoint-replay-<short-name>/.venv
```

The `[ -x ... ] || uv sync` check is a fast-path skip, not a lock: if two
worktrees hit it before the venv exists, both run `uv sync` into the same
directory. Observed under real concurrency without corrupting the venv -
uv's own locking is what makes that safe, not this check.

If a worktree's task needs a new dependency, give it its own private venv
instead of syncing into `$SHARED_VENV` - a `uv add`/`uv sync` there would
rewrite the shared environment underneath every other worktree mid-run.
Separately, `endpoint-replay` itself is installed editable into the shared venv,
so `uv run` in a different worktree re-points that one entry at whichever
worktree ran it last - harmless (imports still resolve) but not
deterministic across worktrees.

When the work is done, push the branch and open a PR for review - that is
the default outcome, not a merge straight to `main`:

```bash
git push -u origin <branch>
```

Then open the PR (`gh pr create` or the GitHub MCP tool). Keep the worktree
and branch around until the PR merges or is closed - don't clean up early.

Push straight to `main` instead when told to directly (e.g. "push this
to main," "commit to main," "no PR needed"), or when the task adds, runs, or
analyses an experiment (a new experiment folder, a hyperparameter/config
change, a cluster run, an analysis-notebook update). Reserve PRs for new
features and bug fixes. Either way, clean up right away:

```bash
git checkout main && git merge --ff-only <branch>   # or: git push origin <branch>:main
```

Either way, once the branch is merged or abandoned, remove the worktree and
its branch. If abandoning, force both - the untracked `.venv` symlink
otherwise blocks a plain removal:

```bash
git worktree remove ../endpoint-replay-<short-name>            # merged
git branch -d <branch>

git worktree remove --force ../endpoint-replay-<short-name>    # abandoned
git branch -D <branch>
```

This mirrors the shared-venv pattern `cluster.toml` already uses for SLURM
jobs - one venv reused by every commit, re-synced only when the lockfile
changes - applied locally instead of on the cluster.

## Learning curves in experiment work

Experiment work (a trial run, a cluster job, a sweep, an analysis-notebook
update) needs a learning curve, not just numbers in a commit message.
Generate it with the experiment's own plotting helpers
(`experiment.plotting`), and commit it under `benchmark_plots/`.

This repo is public, so either raw form renders. Reference the plot as:

```
![alt text](https://github.com/<owner>/<repo>/blob/<branch>/<path>?raw=true)
```

Commit the plot under `benchmark_plots/` first, then link to it this way - in
the commit message when pushing straight to `main`, or in the PR body when
the same change also touches a feature or bug fix.
