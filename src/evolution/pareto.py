import numpy as np


def non_dominated_sort(values):
    """Fast non-dominated sort used in NSGA-II.

    Parameters
    ----------
    values : np.ndarray
        Population objective values with shape (n_individuals, n_objectives).

    Returns
    -------
    list[list[int]]
        A list of fronts, each front is a list of indices.
    """
    population_size = values.shape[0]
    S = [[] for _ in range(population_size)]
    n_dom = np.zeros(population_size, dtype=int)
    fronts = [[]]

    for p in range(population_size):
        for q in range(population_size):
            if p == q:
                continue
            if np.all(values[p] >= values[q]) and np.any(values[p] > values[q]):
                S[p].append(q)
            elif np.all(values[q] >= values[p]) and np.any(values[q] > values[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # remove last empty list
    return fronts


def crowding_distance(values, front):
    """Compute crowding distance for a given front.

    Parameters
    ----------
    values : np.ndarray
        Population objective values.
    front : list[int]
        Indices of individuals in the front.

    Returns
    -------
    dict[int, float]
        Mapping from individual index to its crowding distance.
    """
    if len(front) == 0:
        return {}

    num_objectives = values.shape[1]
    distance = {i: 0.0 for i in front}

    # Normalize values for each objective
    min_vals = values.min(axis=0)
    max_vals = values.max(axis=0)
    span = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
    norm = (values - min_vals) / span

    for m in range(num_objectives):
        sorted_idx = sorted(front, key=lambda i: norm[i, m])
        distance[sorted_idx[0]] = float('inf')
        distance[sorted_idx[-1]] = float('inf')
        for i in range(1, len(sorted_idx) - 1):
            prev_val = norm[sorted_idx[i - 1], m]
            next_val = norm[sorted_idx[i + 1], m]
            distance[sorted_idx[i]] += next_val - prev_val

    return distance
