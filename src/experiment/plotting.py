"""Shared plotting helpers for experiment analysis notebooks.

Kept out of :mod:`experiment.core` (and not re-exported from the package
``__init__``) so the harness stays import-light - this module imports matplotlib
(the ``analysis`` dependency group). Import it explicitly in a notebook::

    from experiment.plotting import seed_grids_for, plot_mean_ci, style

The pipeline turns the per-timestep ``reward``/``terminated``/``truncated`` curves
stored per run into per-seed *return-over-time* stacks, then a mean ± bootstrap-CI
band:

* :func:`episode_returns` - per-episode ``(end_timestep, return)`` for one run.
* :func:`interp_on_grid`  - one seed's curve as a function of timestep (linear
  interpolation onto a shared grid; NaN outside its observed range, no smoothing).
* :func:`seed_grids_for`  - the ``[n_seeds, len(grid)]`` raw-return stack for one
  component dir; :func:`ema_reward_grids_for` is the continuing-task alternative
  (no episode to derive a return from - smooths reward directly instead).
* :func:`weighted_lifetime_return` / :func:`weighted_lifetime_return_stack` - a run's
  episode returns collapsed to one length-weighted scalar, stacked per swept hyper
  value (e.g. learning rate) for a sensitivity curve.
* :func:`average_lifetime_reward` / :func:`average_lifetime_reward_stack` - the same
  idea for a continuing task (e.g. Catch), which has no episode to derive a return
  from: mean reward rate over the whole run instead. :func:`ema_reward` is its
  per-timestep (not collapsed) version, feeding :func:`ema_reward_grids_for`.
* :func:`mean_over_seeds` / :func:`bootstrap_mean_ci` - pointwise aggregates, defined
  only where *every* seed contributes (so the mean is always over the same seeds).
* :func:`plot_mean_ci` / :func:`style` - draw a band + mean line, and shared
  axes styling.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl

from experiment.core import load_curves, load_runs


def episode_returns(reward, terminated, truncated):
    """Find each completed episode's end timestep and return.

    Args:
        reward: Per-timestep reward for one run.
        terminated: Per-timestep termination flags.
        truncated: Per-timestep truncation flags.

    Returns:
        An ``(ends, returns)`` pair of arrays. An episode ends where
        ``terminated`` or ``truncated`` fires, and its return is the reward
        summed since the previous end.
    """
    reward = np.asarray(reward, dtype=float)
    done = (np.asarray(terminated) + np.asarray(truncated)) > 0
    ends = np.flatnonzero(done)
    if ends.size == 0:
        return np.array([]), np.array([])
    cumr = np.cumsum(reward)
    prev = np.concatenate(([0.0], cumr[ends[:-1]]))
    return ends, cumr[ends] - prev


def episode_lengths(ends):
    """Each episode's length in timesteps, from its end timestep.

    Args:
        ends: 0-indexed episode end timesteps, from :func:`episode_returns`.

    Returns:
        The matching lengths. ``ends`` are inclusive end indices, so the
        first episode's length is ``ends[0] + 1``, hence ``prepend=-1``
        (timestep -1: "before the run starts") rather than 0.
    """
    return np.diff(np.asarray(ends), prepend=-1)


def weighted_lifetime_return(reward, terminated, truncated):
    """Summarize one run as its length-weighted mean episode return.

    Longer episodes count for more than shorter ones, unlike a plain mean over
    episodes - useful when episode length itself varies with how well the
    agent is doing (e.g. Pinball).

    Args:
        reward: Per-timestep reward for one run.
        terminated: Per-timestep termination flags.
        truncated: Per-timestep truncation flags.

    Returns:
        ``sum(return_i * length_i) / sum(length_i)`` over completed episodes,
        or NaN if the run completed none.
    """
    ends, rets = episode_returns(reward, terminated, truncated)
    if ends.size == 0:
        return float("nan")
    return float(np.average(rets, weights=episode_lengths(ends)))


def average_lifetime_reward(reward):
    """Summarize one run as its mean per-step reward.

    A continuing task (e.g. Catch) never terminates or truncates within a
    run, so there is no episode to derive a return from at all - a
    length-weighted episode return (:func:`weighted_lifetime_return`) does
    not apply. Mean reward rate over the whole run is the summary instead.

    Args:
        reward: Per-timestep reward for one run.

    Returns:
        ``mean(reward)`` over every timestep in the run.
    """
    return float(np.mean(np.asarray(reward, dtype=float)))


def ema_reward(reward, beta=0.99):
    """Exponential moving average of per-timestep reward.

    The learning-curve counterpart of :func:`average_lifetime_reward` for a
    continuing task: with no episode boundaries to interpolate a return
    between, smoothing the raw reward stream directly is the only option.

    Args:
        reward: Per-timestep reward for one run.
        beta: Smoothing factor in ``[0, 1)``; higher means more smoothing.
            ``ema[i] = beta * ema[i - 1] + (1 - beta) * reward[i]``.

    Returns:
        The smoothed array, the same length as ``reward``. The first value
        is left unsmoothed - there is nothing to average against yet.
    """
    reward = np.asarray(reward, dtype=float)
    ema = np.empty_like(reward)
    if reward.size:
        ema[0] = reward[0]
        for i in range(1, reward.size):
            ema[i] = beta * ema[i - 1] + (1 - beta) * reward[i]
    return ema


def interp_on_grid(ends, rets, grid):
    """Interpolate one seed's episode returns onto a shared timestep grid.

    Sharing a grid lets seeds be averaged pointwise.

    Args:
        ends: Episode end timesteps, from :func:`episode_returns`.
        rets: The matching episode returns.
        grid: Timesteps to interpolate onto.

    Returns:
        Return as a function of timestep, NaN outside the observed range. There
        is no extrapolation and no smoothing.
    """
    if ends.size == 0:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, ends, rets, left=np.nan, right=np.nan)


def _interp_stack_for(results_dir, grid, run_ids, curve_fn):
    """Stack every seed's ``curve_fn(curves)`` interpolated onto a shared grid.

    Shared by :func:`seed_grids_for` and :func:`ema_reward_grids_for` - only
    the per-run ``(xs, ys)`` curve differs.

    Each component has its own database, so no per-config filtering is needed
    unless the database itself sweeps a hyper (e.g. a learning-rate sweep),
    in which case ``run_ids`` restricts to one sweep point's runs.

    Args:
        results_dir: The component's consolidated ``<name>.db``.
        grid: Timesteps every seed is interpolated onto.
        run_ids: Restrict to these run ids, or ``None`` for every run stored.
        curve_fn: Maps one run's ``load_curves`` dict to an ``(xs, ys)`` pair.

    Returns:
        An ``[n_seeds, len(grid)]`` array, one row per run.
    """
    results_dir = Path(results_dir)
    ids = (
        list(run_ids) if run_ids is not None
        else load_runs(results_dir, write_csv=False)["run_id"].to_list()
    )
    grids = []
    for rid in ids:
        xs, ys = curve_fn(load_curves(results_dir, rid))
        grids.append(interp_on_grid(xs, ys, grid))
    return np.vstack(grids) if grids else np.empty((0, len(grid)))


def seed_grids_for(results_dir, grid, run_ids=None):
    """Stack every seed's return-over-time for one component.

    For a continuing task with no episode to derive a return from, use
    :func:`ema_reward_grids_for` instead.

    Args:
        results_dir: The component's consolidated ``<name>.db``.
        grid: Timesteps every seed is interpolated onto.
        run_ids: Restrict to these run ids, or ``None`` for every run stored.

    Returns:
        An ``[n_seeds, len(grid)]`` array, one row per run.
    """
    def curve(c):
        return episode_returns(c["reward"], c["terminated"], c["truncated"])

    return _interp_stack_for(results_dir, grid, run_ids, curve)


def ema_reward_grids_for(results_dir, grid, beta=0.99, run_ids=None):
    """Stack every seed's EMA-smoothed reward-over-time for one component.

    The learning-curve counterpart of :func:`seed_grids_for` for a continuing
    task (e.g. Catch), which has no episode boundaries to derive a return
    from at all.

    Args:
        results_dir: The component's consolidated ``<name>.db``.
        grid: Timesteps every seed is interpolated onto.
        beta: Smoothing factor, passed to :func:`ema_reward`.
        run_ids: Restrict to these run ids, or ``None`` for every run stored.

    Returns:
        An ``[n_seeds, len(grid)]`` array, one row per run.
    """
    def curve(c):
        reward = np.asarray(c["reward"], dtype=float)
        return np.arange(reward.size), ema_reward(reward, beta=beta)

    return _interp_stack_for(results_dir, grid, run_ids, curve)


def _metric_stack(results_dir, column, values, metric_fn):
    """Per-seed ``metric_fn(curves)``, one column per swept hyper value.

    Shared by :func:`weighted_lifetime_return_stack` and
    :func:`average_lifetime_reward_stack` - only the per-run scalar differs.

    Args:
        results_dir: A component's consolidated ``<name>.db`` whose runs sweep
            ``column`` (e.g. ``"AGENT_HYPERS.LR"``) over shared seeds.
        column: The dotted config column swept, as flattened by
            :func:`~experiment.core.load_runs`.
        values: The values to pull, in output-column order.
        metric_fn: Maps one run's ``load_curves`` dict to a scalar.

    Returns:
        An ``[n_seeds, len(values)]`` array; a value with fewer runs than the
        widest column is padded with NaN.
    """
    results_dir = Path(results_dir)
    df = load_runs(results_dir, write_csv=False)
    columns = []
    for value in values:
        rids = df.filter(pl.col(column) == value).sort("seed")["run_id"].to_list()
        columns.append([metric_fn(load_curves(results_dir, rid)) for rid in rids])
    width = max((len(c) for c in columns), default=0)
    padded = [c + [float("nan")] * (width - len(c)) for c in columns]
    return np.array(padded).T if width else np.empty((0, len(values)))


def weighted_lifetime_return_stack(results_dir, column, values):
    """Per-seed weighted-lifetime-return, one column per swept hyper value.

    For a sensitivity curve: feed the result straight into
    :func:`bootstrap_mean_ci` / :func:`plot_mean_ci` with ``values`` as the
    x-axis grid. For a continuing task (e.g. Catch), use
    :func:`average_lifetime_reward_stack` instead.

    Args:
        results_dir: A component's consolidated ``<name>.db`` whose runs sweep
            ``column`` (e.g. ``"AGENT_HYPERS.LR"``) over shared seeds.
        column: The dotted config column swept, as flattened by
            :func:`~experiment.core.load_runs`.
        values: The values to pull, in output-column order.

    Returns:
        An ``[n_seeds, len(values)]`` array; a value with fewer runs than the
        widest column is padded with NaN.
    """
    def metric(c):
        return weighted_lifetime_return(c["reward"], c["terminated"], c["truncated"])

    return _metric_stack(results_dir, column, values, metric)


def average_lifetime_reward_stack(results_dir, column, values):
    """Per-seed average-lifetime-reward, one column per swept hyper value.

    The continuing-task counterpart of :func:`weighted_lifetime_return_stack`
    - see :func:`average_lifetime_reward`.

    Args:
        results_dir: A component's consolidated ``<name>.db`` whose runs sweep
            ``column`` (e.g. ``"AGENT_HYPERS.LR"``) over shared seeds.
        column: The dotted config column swept, as flattened by
            :func:`~experiment.core.load_runs`.
        values: The values to pull, in output-column order.

    Returns:
        An ``[n_seeds, len(values)]`` array; a value with fewer runs than the
        widest column is padded with NaN.
    """
    def metric(c):
        return average_lifetime_reward(c["reward"])

    return _metric_stack(results_dir, column, values, metric)


def mean_over_seeds(stack):
    """Average a seed stack pointwise.

    Args:
        stack: An ``[n_seeds, len(grid)]`` array.

    Returns:
        The mean, NaN wherever not every seed contributes. Holding the seed set
        fixed avoids an early bias toward the seeds that finish first.
    """
    stack = np.asarray(stack)
    with warnings.catch_warnings():
        # all-NaN edge columns
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    mean[(~np.isnan(stack)).sum(axis=0) < stack.shape[0]] = np.nan
    return mean


def bootstrap_mean_ci(stack, n_boot=10_000, lo=2.5, hi=97.5, seed=0):
    """Bootstrap a mean and confidence interval over seeds at each timestep.

    Args:
        stack: An ``[n_seeds, len(grid)]`` array.
        n_boot: Resamples of the seeds, drawn with replacement.
        lo: Lower percentile of the interval.
        hi: Upper percentile of the interval.
        seed: Seed for the resampling RNG.

    Returns:
        A ``(mean, ci_lo, ci_hi)`` triple, NaN wherever not every seed is
        present. With a single seed the band collapses onto the mean.
    """
    stack = np.asarray(stack)
    n, m = stack.shape
    valid = (~np.isnan(stack)).sum(axis=0) == n
    mean = np.full(m, np.nan)
    ci_lo = np.full(m, np.nan)
    ci_hi = np.full(m, np.nan)
    sub = stack[:, valid]                              # [n_seeds, n_valid], no NaNs
    mean[valid] = sub.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, sub.shape[1]))
    for s in range(0, n_boot, 1000):                   # chunked to bound memory
        e = min(s + 1000, n_boot)
        idx = rng.integers(0, n, size=(e - s, n))      # resample seed indices
        boot[s:e] = sub[idx].mean(axis=1)
    ci_lo[valid], ci_hi[valid] = np.percentile(boot, [lo, hi], axis=0)
    return mean, ci_lo, ci_hi


def plot_mean_ci(ax, grid, stack, label, color, n_boot=10_000):
    """Draw a mean line and its shaded bootstrap CI band onto an axis.

    Args:
        ax: The matplotlib axis to draw on.
        grid: Timesteps matching ``stack``'s columns.
        stack: An ``[n_seeds, len(grid)]`` array.
        label: Legend label for the mean line.
        color: Colour shared by the line and the band.
        n_boot: Resamples passed to :func:`bootstrap_mean_ci`.

    Returns:
        The plotted ``mean`` array. The band is masked to where every seed is
        present, so it has zero width for a single seed.
    """
    mean, ci_lo, ci_hi = bootstrap_mean_ci(stack, n_boot=n_boot)
    m = ~np.isnan(mean)
    ax.fill_between(grid[m], ci_lo[m], ci_hi[m], color=color, alpha=0.2)  # band
    ax.plot(grid[m], mean[m], lw=2.5, color=color, label=label)           # thick mean
    return mean


def style(ax, ylim=None, xlabel="Timestep", ylabel="Return"):
    """Apply the shared axes styling: no grid, no top/right spines.

    Args:
        ax: The matplotlib axis to style.
        ylim: Fixes the return range, which differs per env, e.g. Pinball
            ``(-1000, 0)``, Atari Pong ``(-21, 21)``.
        xlabel: The x-axis label, e.g. ``"Learning rate"`` for a sensitivity
            curve instead of the default learning-curve timestep axis.
        ylabel: The y-axis label, drawn horizontal and right-aligned.
    """
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
