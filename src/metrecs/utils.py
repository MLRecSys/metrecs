from typing import Dict, List
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
    >>> normalized_scaled_harmonic_number_series(6)
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


def compute_normalized_distribution(
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
    >>> weights = np.array([1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)])
    >>> compute_distribution(a, weights=weights)
        {'a': 0.4799997900047559, 'b': 0.23999989500237795, 'c': 0.2799998775027743}
    """
    n_values = len(a)

    distr = {} if not distribution else distribution
    weights = weights if np.any(weights) else np.ones(n_values) / n_values
    for item, weight in zip(a, weights):
        distr[item] = weight + distr.get(item, 0.0)
    return distr


# TODO: write clean make tests
def compute_normalized_distribution_multiple_categories(
    a: List[set[str]],
    weights: np.ndarray[float] = [],
    distribution: Dict[str, float] = {},
) -> Dict:
    """_summary_
    Args:
        a (np.ndarray[str]): list of list of a [LET PEOPLE KNOW THAT IS IS A SET OF UNIQUE ITEMS PER LIST]
        weights (np.ndarray[float], optional): _description_. Defaults to [].
        distribution (Dict[str, float], optional): _description_. Defaults to {}.
    Returns:
        Dict: _description_
    >>> a = [set(['a', 'x']), set(['b', 'c', 'x']), set(['c', 'a', 'y', 'x', 'q', 't']), set(['c', 'b'])]
    >>> weights = np.array([1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)])
    >>> compute_distribution_multiple_categories(a, weights=weights)
    """
    n_elements = len(a)

    distr = {} if not distribution else distribution
    weights = weights if np.any(weights) else np.ones(n_elements) / n_elements
    for item, weight in zip(a, weights):
        for cat in set(item):
            distr[cat] = weight + distr.get(cat, 0.0)  # use this
    norm_ = sum(distr.values())
    return {key: val / norm_ for key, val in distr.items()}


# {'a': 0.2933332050029064, 'x': 0.3199998600031706, 'b': 0.13999993875138714, 'c': 0.19333324875191557, 'y': 0.05333331000052843}
# a = np.array([['a', 'b', 'x'], ['b', 'c', 'x'], ['c', 'a', 'y'], ['c', 'b', 'x']])
# compute_distribution_multiple_categories(a)
# {'a': 0.16666666666666666, 'b': 0.25, 'x': 0.25, 'c': 0.25, 'y': 0.08333333333333333}
