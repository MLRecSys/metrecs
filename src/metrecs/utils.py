from typing import Dict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from collections import Counter
import numpy as np
import math
from numpy.typing import ArrayLike
from numpy.linalg import norm
from scipy.spatial import distance


def harmonic_number(n):
    """Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5 / n - 1.0 / (12 * n**2) + 1.0 / (120 * n**4)

1 / harmonic_number(1)

items_rec = np.array([["a", "b", "a", "c", "a"], ["a", "c", "b", "c", "d"]] * 2)
context_distribution = {item: 0 for item in np.unique(items_rec)}
items = items_rec[0]


def compute_distribution(
    a: ArrayLike[str],
    dict_distribution: Dict[str, int] = {},
    rank_aware: bool = False,
):
    """
    Args:
        items (np.ndarray): subset of items in context_distribution
        context_distribution (Dict, optional): _description_. Defaults to None.
        rank_aware (bool, optional): _description_. Defaults to False.
    """
    n_items = len(X)
    rank_aware = np.array([harmonic_number(i) for i in range(1, n_items + 1)])[
        ::-1
    ]
    if rank_aware:
        for i, item in enumerate(items):
            context_distribution[item] += 1 * harmonic_number(n_items - i)
    else:
        context_distribution.update(Counter(own))
        dict_distribution.update(Counter(own))


def histogram(a: np.array, rank_aware=False):
    n = np.unique(a, return_counts=True)[1]
    sum_one_over_ranks = harmonic_number(len(a))
    if rank_aware:
        p = (n * 1 / np.sum(n)) / sum_one_over_ranks
    else:
        p = n * 1 / np.sum(n)
    return p


def cosine_distances(X: ArrayLike) -> np.ndarray:
    """Implementation of the pairwice cosine similarity function
    Args:
        X: {array-like, sparse matrix} of shape (n_samples_X, n_features)
    Returns:
        distance matrix: ndarray of shape (n_samples_X, n_samples_Y)
    """
    distances = pdist(X, metric="cosine")
    return squareform(distances)


# === DELETE:
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


def compute_jensen_shannon_divergence(s: np.ndarray, q: np.ndarray, alpha=0.001):
    """
    TODO: refactor into KL_function & JS_function
    * Jensen-Shannon (JS) divergence
    * Kullback-Leibler (KL)
    KL (p || q), the lower the better.
    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    if 0.99 <= sum(s) <= 1.01 or 0.99 <= sum(q) <= 1.01:
        raise ValueError("Assertion Error")

    ss = []
    qq = []
    merged_dic = opt_merge_max_mappings(s, q)
    for key in sorted(merged_dic.keys()):
        q_score = q.get(key, 0.0)
        s_score = s.get(key, 0.0)
        ss.append((1 - alpha) * s_score + alpha * q_score)
        qq.append((1 - alpha) * q_score + alpha * s_score)

    kl = entropy(ss, qq, base=2)
    jsd = JSD(ss, qq)
    return [kl, jsd]


def KL_symmetric(a, b):
    return (entropy(a, b, base=2) + entropy(b, a, base=2)) / 2


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    # return 0.5 * (KL(_P, _M) + KL(_Q, _M))
    # added the abs to catch situations where the disocunting causes a very small <0 value, check this more!!!!
    try:
        jsd_root = math.sqrt(
            abs(0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2)))
        )
    except ZeroDivisionError:
        print(P)
        print(Q)
        print()
        jsd_root = None
    return
