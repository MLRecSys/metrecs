from typing import Dict, List, Iterable

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
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


# TODO: write unit test (should output sklearn's cosine_distances !)
def cosine_distances(X: np.ndarray) -> np.ndarray:
    """Implementation of the pairwice cosine similarity function
    Args:
        X: {array-like, sparse matrix} of shape (n_samples_X, n_features)
    Returns:
        distance matrix: ndarray of shape (n_samples_X, n_samples_Y)
    """
    distances = pdist(X, metric="cosine")
    return squareform(distances)


def compute_normalized_distribution(
    R: np.ndarray[str],
    weights: np.ndarray[float] = None,
    distribution: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Compute a normalized weigted distribution for a list of items that each can have a single representation assigned.

    Args:
        a (np.ndarray[str]): an array of items representation.
        weights (np.ndarray[float], optional): weights to assign each element in a. Defaults to None.
            * Following yields: len(weights) == len(a)
        distribution (Dict[str, float], optional): dictionary to assign the distribution values, if None it will be generated as {}. Defaults to None.
            * Use case; if you want to add distribution values to existing, one can input it.

    Returns:
        Dict[str, float]: dictionary with normalized distribution values

    >>> a = np.array(["a", "b", "c", "c"])
    >>> weights = np.array([1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)])
    >>> compute_normalized_distribution(a, weights=weights)
        {'a': 0.4799997900047559, 'b': 0.23999989500237795, 'c': 0.2799998775027743}
    >>> compute_normalized_distribution(a, weights=None)
        {'a': 0.25, 'b': 0.25, 'c': 0.5}
    """
    R = np.asarray(R)
    n_elements = len(R)

    distr = distribution if distribution else {}
    weights = (
        np.asarray(weights) if weights is not None else np.ones(n_elements) / n_elements
    )
    for item, weight in zip(R, weights):
        distr[item] = weight + distr.get(item, 0.0)
    return distr


# TODO: write unit test
def compute_normalized_distribution_multiple_categories(
    R: Iterable[set[str]],
    weights: np.ndarray[float] = None,
    distribution: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Compute a normalized weigted distribution for a list of items that each can have multiple representation assigned.
    For instance, a list of news articles that each can have multiple categoies, e.g. 'politics' + 'economy' OR 'entertainment' + 'sport'.

    Args:
        a (Iterable[set[str]]): a list of sets of items representation.
        weights (np.ndarray[float], optional): weights to assign each set in the list. Defaults to None.
            * Following yields: len(weights) == len(a)
        distribution (Dict[str, float], optional): dictionary to assign the distribution values, if None it will be generated as {}. Defaults to None.
            * Use case; if you want to add distribution values to existing, one can input it.
    Returns:
        Dict[str, float]: dictionary with normalized distribution values
    >>> a = [['a', 'x'], ['b', 'c', 'x'], ['c', 'a', 'y', 'x', 'q', 't'], ['c', 'b']]
    >>> weights = np.array([1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)])
    >>> compute_normalized_distribution_multiple_categories(a, weights=weights)
    """
    R = np.asarray(R)
    n_elements = len(R)

    distr = distribution if distribution else {}
    weights = (
        np.asarray(weights) if weights is not None else np.ones(n_elements) / n_elements
    )
    for item, weight in zip(R, weights):
        for cat in set(item):
            distr[cat] = weight + distr.get(cat, 0.0)
    norm_ = sum(distr.values())
    return {key: val / norm_ for key, val in distr.items()}


# TODO: write unit test
def opt_merge_max_mappings(dict1, dict2):
    """Merges two dictionaries based on the largest value in a given mapping.
    Parameters
    ----------
    dict1 : Dict[Any, Comparable]
    dict2 : Dict[Any, Comparable]
    Returns
    -------
    Dict[Any, Comparable]
        The merged dictionary
    """
    # we will iterate over `other` to populate `merged`
    merged, other = (dict1, dict2) if len(dict1) > len(dict2) else (dict2, dict1)
    merged = dict(merged)

    for key in other:
        if key not in merged or other[key] > merged[key]:
            merged[key] = other[key]
    return merged


# TODO: write unit test
def avoid_distribution_misspecification(s: Dict, q: Dict, alpha=0.001) -> Dict:
    """ """
    merged_dic = opt_merge_max_mappings(s, q)

    for key in sorted(merged_dic):
        q_score = q.get(key, 0.0)
        s_score = s.get(key, 0.0)
        s[key] = (1 - alpha) * s_score + alpha * q_score
        q[key] = (1 - alpha) * q_score + alpha * s_score
    # Sort
    q = {key: q[key] for key in sorted(q.keys())}
    s = {key: s[key] for key in sorted(s.keys())}
    return s, q


def user_level_RADio_categorical(
    user_recommendations: List[str],
    users_context: List[str],
    user_rec_weights: List[float] = None,
    users_context_weights: List[float] = None,
) -> float:
    """_summary_
    Args:
        user_recommendations (np.ndarray[str]): _description_
        users_context (np.ndarray[str]): _description_
        user_rec_weights (np.ndarray[float]): _description_
        users_context_weights (np.ndarray[float]): _description_
    Returns:
        float: _description_
    len(users_context) == len(users_context_weights)
    len(user_recommendations) == len(user_rec_weights)
    """
    q = compute_normalized_distribution(user_recommendations, weights=user_rec_weights)
    p = compute_normalized_distribution(users_context, weights=users_context_weights)
    qq, pp = avoid_distribution_misspecification(q, p)
    return float(distance.jensenshannon(list(qq.values()), list(pp.values()), base=2))


def user_level_RADio_multicategorical(
    user_recommendations: List[List[str]],
    users_context: List[List[str]],
    user_rec_weights: List[float] = None,
    users_context_weights: List[float] = None,
) -> float:
    """_summary_
    Args:
        user_recommendations (np.ndarray[np.ndarray[str]]): _description_
        users_context (np.ndarray[np.ndarray[str]): _description_
        user_rec_weights (np.ndarray[float]): _description_
        users_context_weights (np.ndarray[float]): _description_
    Returns:
        float: _description_
    len(users_context) == len(users_context_weights)
    len(user_recommendations) == len(user_rec_weights)
    """
    q = compute_normalized_distribution_multiple_categories(
        user_recommendations, weights=user_rec_weights
    )
    p = compute_normalized_distribution_multiple_categories(
        users_context, weights=users_context_weights
    )
    qq, pp = avoid_distribution_misspecification(q, p)
    return float(distance.jensenshannon(list(qq.values()), list(pp.values()), base=2))
