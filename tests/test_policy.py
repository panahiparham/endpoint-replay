"""Epsilon-greedy action selection: tie-breaking and action probabilities.

The contract this suite pins down:

* An action tied for the maximum q-value is as likely as any other tied
  action, rather than ``jnp.argmax``'s lowest-index winner.
* With a unique maximum, the greedy action is drawn with probability
  ``epsilon / action_dim + (1 - epsilon)`` and every other action with
  ``epsilon / action_dim``.
* ``epsilon = 1`` is uniform over all actions and ``epsilon = 0`` never
  leaves the tied-maximum set.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.ddqn import _epsilon_greedy_action


N_SAMPLES = 20_000
TOLERANCE = 0.02


def _empirical_frequencies(q_values, epsilon, action_dim, seed=0):
    """Sample the policy ``N_SAMPLES`` times and return per-action frequency."""
    keys = jax.random.split(jax.random.PRNGKey(seed), N_SAMPLES)
    sample = jax.vmap(
        lambda k: _epsilon_greedy_action(
            jnp.asarray(q_values), jnp.asarray(epsilon), action_dim, k
        )
    )(keys)
    counts = np.bincount(np.asarray(sample), minlength=action_dim)
    return counts / N_SAMPLES


def test_all_actions_tied_is_uniform():
    """Every action tied for the maximum is drawn equally often."""
    freqs = _empirical_frequencies([1.0, 1.0, 1.0, 1.0], 0.0, 4)
    assert np.allclose(freqs, 0.25, atol=TOLERANCE)


def test_argmax_alone_would_be_degenerate():
    """The tie the policy resolves is one ``jnp.argmax`` would collapse."""
    q_values = jnp.asarray([1.0, 1.0, 1.0, 1.0])
    assert int(jnp.argmax(q_values)) == 0
    freqs = _empirical_frequencies(q_values, 0.0, 4)
    assert freqs[0] < 0.5


def test_partial_tie_splits_greedy_mass():
    """Two actions tied for the maximum split the greedy mass evenly."""
    freqs = _empirical_frequencies([2.0, 2.0, 0.0, 0.0], 0.0, 4)
    assert np.allclose(freqs[:2], 0.5, atol=TOLERANCE)
    assert np.allclose(freqs[2:], 0.0, atol=TOLERANCE)


def test_unique_max_matches_epsilon_greedy_probabilities():
    """A unique maximum gets ``eps/n + (1 - eps)``, the rest ``eps/n``."""
    epsilon, action_dim = 0.2, 4
    freqs = _empirical_frequencies([0.0, 5.0, 1.0, 2.0], epsilon, action_dim)
    expected_greedy = epsilon / action_dim + (1.0 - epsilon)
    assert np.isclose(freqs[1], expected_greedy, atol=TOLERANCE)
    for i in (0, 2, 3):
        assert np.isclose(freqs[i], epsilon / action_dim, atol=TOLERANCE)


def test_epsilon_one_is_uniform_regardless_of_q_values():
    """``epsilon = 1`` ignores the q-values entirely."""
    freqs = _empirical_frequencies([0.0, 100.0, -5.0, 3.0], 1.0, 4)
    assert np.allclose(freqs, 0.25, atol=TOLERANCE)


def test_epsilon_zero_never_leaves_the_tied_maximum():
    """``epsilon = 0`` puts zero mass on any non-maximal action."""
    freqs = _empirical_frequencies([0.0, 5.0, 1.0, 2.0], 0.0, 4)
    assert freqs[1] == 1.0


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.5, 1.0])
def test_probabilities_sum_to_one(epsilon):
    """The sampled distribution covers exactly the available actions."""
    freqs = _empirical_frequencies([0.0, 5.0, 1.0, 2.0], epsilon, 4)
    assert np.isclose(freqs.sum(), 1.0)


def test_is_jittable():
    """The policy compiles under ``jax.jit``."""
    fn = jax.jit(_epsilon_greedy_action, static_argnums=2)
    action = fn(
        jnp.asarray([0.0, 1.0, 0.0]),
        jnp.asarray(0.1),
        3,
        jax.random.PRNGKey(0),
    )
    assert action.dtype == jnp.int32
    assert 0 <= int(action) < 3
