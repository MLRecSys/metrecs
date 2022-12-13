from typing import Dict
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
from scipy.stats import entropy

import numpy as np
import math


def harmonic_number(n: int) -> float:
    """Returns an approximate value of n-th harmonic number.
    The harmonic number can be approximated using the first few terms of the Taylor series expansion:
    Source: http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5 / n - 1.0 / (12 * n**2) + 1.0 / (120 * n**4)


def normalized_scaled_harmonic_number_series(n: int) -> np.ndarray[float]:
    """Return an array of scaled normalized harmonic numbers

    Args:
        n (int): number of values in array to return

    Returns:
        np.ndarray[float]: an array with scaled normalized harmonic number

    >>> import numpy as np
    >>> normalized_scaled_harmonic_number_series(5)
        array([0.43795616, 0.21897808, 0.14598539, 0.10948904, 0.08759123])
    >>> sum(normalized_scaled_harmonic_number_series(5))
        0.9999998931376903
    """
    return np.array([1 / rank / harmonic_number(n) for rank in range(1, n + 1)])


def cosine_distances(X: ArrayLike) -> np.ndarray:
    """Implementation of the pairwice cosine similarity function
    Args:
        X: {array-like, sparse matrix} of shape (n_samples_X, n_features)
    Returns:
        distance matrix: ndarray of shape (n_samples_X, n_samples_Y)
    """
    distances = pdist(X, metric="cosine")
    return squareform(distances)


def compute_distribution(
    a: np.ndarray[str],
    weights: np.ndarray[float] = [],
    distribution: Dict[str, float] = {},
) -> Dict:
    """_summary_
    Args:
        a (np.ndarray[str]): _description_
        weights (np.ndarray[float], optional): _description_. Defaults to [].
        distribution (Dict[str, float], optional): _description_. Defaults to {}.
    Returns:
        Dict: _description_
    >>> a = np.array(["a", "b", "c", "c"])
    >>> w1 = np.array([1 / harmonic_number(val) for i, val in enumerate(range(1, len(a) + 1))])
    >>> w2 = np.array([1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)])
    >>> compute_distribution(a, weights=w1)
        {'a': 0.997789233416392, 'b': 0.6666442916569965, 'c': 1.0254528751419092}
    >>> compute_distribution(a, weights=w2)
        {'a': 0.4799997900047559, 'b': 0.23999989500237795, 'c': 0.2799998775027743}
    >>> compute_distr(a, False)
        {'a': 0.25, 'b': 0.25, 'c': 0.5}
    """
    distr = {} if not distribution else distribution
    weights = weights if np.any(weights) else np.ones(len(a)) / len(a)
    for item, weight in zip(a, weights):
        distr[item] = weight + distr.get(item, 0.0)
    return distr


def compute_distribution_multiple_categories(
    a: np.ndarray[str],
    weights: np.ndarray[float] = [],
    distribution: Dict[str, float] = {},
) -> Dict:
    """_summary_
    Args:
        a (np.ndarray[str]): _description_
        weights (np.ndarray[float], optional): _description_. Defaults to [].
        distribution (Dict[str, float], optional): _description_. Defaults to {}.
    Returns:
        Dict: _description_
    >>> a = np.array([['a', 'b', 'x'], ['b', 'c', 'x'], ['c', 'a', 'y'], ['c', 'b', 'x']])
    >>> w1 = np.array([1 / harmonic_number(val) for i, val in enumerate(range(1, len(a) + 1))])
    >>> w2 = np.array([1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)])
    >>> compute_distribution_multiple_categories(a, weights=w1)
        {'a': 0.5144141061845151, 'b': 0.7148111050260482, 'x': 0.7148111050260482, 'c': 0.5640323889329686, 'y': 0.1818176950457178}
    >>> compute_distribution_multiple_categories(a, weights=w2)
        {'a': 0.21333324000211373, 'b': 0.2799998775027743, 'x': 0.2799998775027743, 'c': 0.1733332575017174, 'y': 0.05333331000052843}
    >>> compute_distribution_multiple_categories(a, False)
        {'a': 0.16666666666666666, 'b': 0.25, 'x': 0.25, 'c': 0.25, 'y': 0.08333333333333333}
    """
    distr = {} if not distribution else distribution
    weights = weights if np.any(weights) else np.ones(len(a)) / len(a)
    for item, weight in zip(a, weights):
        n_items = len(item)
        for cat in item:
            distr[cat] = weight / n_items + distr.get(cat, 0.0)
    return distr
