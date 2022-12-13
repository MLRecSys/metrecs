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


def normalized_scaled_harmonic_number_series(n: int) -> ArrayLike[float]:
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


# TODO: write unit test (should output sklearn's cosine_distances !)
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
    weights: np.ndarray[float] = None,
    distribution: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Compute a normalized weigted distribution for a list of items that each can have a single representation assigned.

    Args:
        a (np.ndarray[str]): a list/array of items representation.
        weights (ArrayLike[float], optional): weights to assign each element in a. Defaults to None.
            * Following yields: len(weights) == len(a)
        distribution (Dict[str, float], optional): dictionary to assign the distribution values, if None it will be generated as {}. Defaults to None.
            * Use case; if you want to add distribution values to existing, one can input it.

    Returns:
        Dict[str, float]: dictionary with normalized distribution values

    >>> a = np.array(["a", "b", "c", "c"])
    >>> weights = np.array([1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)])
    >>> compute_distribution(a, weights=weights)
        {'a': 0.4799997900047559, 'b': 0.23999989500237795, 'c': 0.2799998775027743}
    >>> compute_distribution(a)
        {'a': 0.25, 'b': 0.25, 'c': 0.5}
    """

    n_elements = len(a)

    distr = {} if not distribution else distribution
    weights = weights if weights else np.ones(n_elements) / n_elements
    for item, weight in zip(a, weights):
        distr[item] = weight + distr.get(item, 0.0)
    return distr


# TODO: write unit test
def compute_normalized_distribution_multiple_categories(
    a: List[set[str]],
    weights: np.ndarray[float] = None,
    distribution: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Compute a normalized weigted distribution for a list of items that each can have multiple representation assigned.
    For instance, a list of news articles that each can have multiple categoies, e.g. 'politics' + 'economy' OR 'entertainment' + 'sport'.

    Args:
        a (List[set[str]]): a list of sets of items representation.
        weights (ArrayLike[float], optional): weights to assign each set in the list. Defaults to None.
            * Following yields: len(weights) == len(a)
        distribution (Dict[str, float], optional): dictionary to assign the distribution values, if None it will be generated as {}. Defaults to None.
            * Use case; if you want to add distribution values to existing, one can input it.
    Returns:
        Dict[str, float]: dictionary with normalized distribution values
    >>> a = [['a', 'x'], ['b', 'c', 'x'], ['c', 'a', 'y', 'x', 'q', 't'], ['c', 'b']]
    >>> weights = np.array([1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)])
    >>> compute_distribution_multiple_categories(a, weights=weights)
    """
    n_elements = len(a)

    distr = {} if not distribution else distribution
    weights = weights if not weights else np.ones(n_elements) / n_elements
    for item, weight in zip(a, weights):
        for cat in set(item):
            distr[cat] = weight + distr.get(cat, 0.0)
    norm_ = sum(distr.values())
    return {key: val / norm_ for key, val in distr.items()}
