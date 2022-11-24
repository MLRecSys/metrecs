from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Iterable
from collections import Counter
import numpy as np
import math
from scipy.stats import entropy
from numpy.linalg import norm


def harmonic_number(n):
    """Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)

def histogram(a: np.array, adjusted = False):
    n = np.unique(a, return_counts = True)[1]
    sum_one_over_ranks = harmonic_number(len(a))
    if adjusted:
        p = n * 1 / np.sum(n) / sum_one_over_ranks
    else:
        p = n * 1 / np.sum(n)
    return(p)

def cosine_distances(X):
    distances = pdist(X, metric="cosine")
    return squareform(distances)


def scale_range(
    m: np.ndarray,
    r_min: float = None,
    r_max: float = None,
    t_min: float = 0,
    t_max: float = 1.0,
) -> None:
    """Scale an array between a range
    Source: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

    m -> ((m-r_min)/(r_max-r_min)) * (t_max-t_min) + t_min

    Args:
        m ∈ [r_min,r_max] denote your measurements to be scaled
        r_min denote the minimum of the range of your measurement
        r_max denote the maximum of the range of your measurement
        t_min denote the minimum of the range of your desired target scaling
        t_max denote the maximum of the range of your desired target scaling
    """
    if not r_min:
        r_min = np.min(m)
    if not r_max:
        r_max = np.max(m)
    return ((m - r_min) / (r_max - r_min)) * (t_max - t_min) + t_min


def opt_merge_max_mappings(dict1, dict2):
    """ Merges two dictionaries based on the largest value in a given mapping.
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

def compute_kl_divergence(s, q, alpha=0.001):
    """
    KL (p || q), the lower the better.
    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    try:
        assert 0.99 <= sum(s.values()) <= 1.01
        assert 0.99 <= sum(q.values()) <= 1.01
    except AssertionError:
        print("Assertion Error")
        pass
    kl_div = 0.
    ss = []
    qq = []
    merged_dic = opt_merge_max_mappings(s, q)
    for key in sorted(merged_dic.keys()):
        q_score = q.get(key, 0.)
        s_score = s.get(key, 0.)
        ss.append((1 - alpha) * s_score + alpha * q_score)
        qq.append((1 - alpha) * q_score + alpha * s_score)
        # by contruction they cannot be both 0
        # if s_score == 0 and q_score == 0:
        #     pass
        #     # raise Exception('Something is wrong in compute_kl_divergence')
        # elif s_score == 0:
        #     ss_score = (1 - alpha) * s_score + alpha * q_score
        #     ss.append(ss_score)
        #     qq.append(q_score)
        # elif q_score == 0:
        #     qq_score = (1 - alpha) * q_score + alpha * s_score
        #     ss.append(s_score)
        #     qq.append(qq_score)
        # else:
        #     ss.append(s_score)
        #     qq.append(q_score)
    kl = entropy(ss, qq, base=2)
    jsd = JSD(ss,qq)
    return [kl, jsd]


def KL_symmetric(a, b):
    return (entropy(a, b, base=2) + entropy(b, a, base=2))/2


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    # return 0.5 * (KL(_P, _M) + KL(_Q, _M))
    # added the abs to catch situations where the disocunting causes a very small <0 value, check this more!!!!
    try:
        jsd_root = math.sqrt(abs(0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2))))
    except ZeroDivisionError:
        print(P)
        print(Q)
        print()
        jsd_root = None
    return 
