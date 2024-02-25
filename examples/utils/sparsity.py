import numpy as np
from tqdm import tqdm


def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    i = 0
    for xi in tqdm(x[:-1]):
        diffsum += np.sum(np.abs(xi - x[i:]))
        i += 1
    return diffsum / (len(x) ** 2 * np.mean(x))


def space_log(n_users: int, n_items: int, sc: int = 1000) -> float:
    return np.log10(n_users * n_items / sc)


def shape_log(n_users: int, n_items: int) -> float:
    return np.log10(n_users / n_items)


def density_log(n_users: int, n_items: int, n_interactions: int) -> float:
    return np.log10(n_interactions / (n_users * n_items))
