"""Welford's parallel algorithm for variance calculation."""
from functools import reduce

import numpy as np


def welford(arr: np.ndarray, axis=None) -> tuple:
    """Compute Welford quantities along an axis.
    
    Args:
        arr: Input array
        axis: Axis to compute over. None flattens array. Default None.
    
    Returns:
        (n, mean, m2) where variance = m2 / (n - 1) if n > 1
    """
    if axis is None:
        n = arr.size
        mean = float(np.mean(arr))
        m2 = float(np.sum((arr - mean) ** 2))
    else:
        n = arr.shape[axis]
        mean = np.mean(arr, axis=axis)
        m2 = np.sum((arr - np.expand_dims(mean, axis=axis)) ** 2, axis=axis)
    return (n, mean, m2)


def merge(state_a: tuple, state_b: tuple) -> tuple:
    """Merge two Welford states."""
    n_a, mean_a, m2_a = state_a
    n_b, mean_b, m2_b = state_b
    if n_a == 0:
        if n_b == 0:
            return (0, 0.0, 0.0)
        return state_b
    if n_b == 0:
        return state_a
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    m2 = m2_a + m2_b + delta ** 2 * n_a * n_b / n
    return (n, mean, m2)


def variance(state: tuple, ddof=1) -> float:
    """Compute variance from Welford state.
    
    Args:
        state: Welford state tuple (n, mean, m2)
        ddof: Degrees of freedom (default 1 for unbiased estimate)
    """
    n, mean, m2 = state
    return m2 / (n - ddof) if n > ddof else 0.0


def std(state: tuple, ddof=0) -> float:
    """Compute standard deviation from Welford state.
    
    Args:
        state: Welford state tuple (n, mean, m2)
        ddof: Degrees of freedom (default 1 for unbiased estimate)
    """
    return np.sqrt(variance(state, ddof=ddof))

if __name__ == "__main__":
    x =  [np.empty((0, 3, 2))]+ [np.empty((0, 3, 2))]+[np.random.rand(1, 3, 2)]+[np.random.rand(1, 3, 2)]+[np.random.rand(np.random.randint(100), 3,2) for _ in range(10)]
    states = [welford(arr,axis=0) for arr in x]

    final_state = reduce(merge, states)
    print(variance(final_state))
    print(np.var(np.concatenate(x, axis=0), axis=0))
    print(std(final_state))
    print(np.std(np.concatenate(x, axis=0), axis=0))
    global_mean = final_state[1]
    global_std = std(final_state)
    _ , means, _ = zip(*states)
    means = np.stack(means, axis=0)
    means_z = (means[2:] - global_mean) / global_std
    means_z_exp = np.stack([np.mean((x_ - global_mean)/global_std, axis=0) for x_ in x[2:]], axis=0)
    np.testing.assert_allclose(means_z, means_z_exp)