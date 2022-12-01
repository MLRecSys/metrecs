import math
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict


def harmonic_number(n):
    """Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5 / n - 1.0 / (12 * n**2) + 1.0 / (120 * n**4)


# =====


def compute_distr(items, adjusted=False):
    """
    Calibration

    Compute the genre distribution for a given list of Items.
    """
    n = len(items)
    sum_one_over_ranks = harmonic_number(n)
    count = 0
    distr = {}
    for _, item in items.iterrows():
        count += 1
        topic_freq = distr.get(item.category, 0.0)
        distr[item.category] = (
            topic_freq + 1 * 1 / count / sum_one_over_ranks
            if adjusted
            else topic_freq + 1 * 1 / n
        )
    return distr


def compute_distr(items, adjusted=False):
    """
    FRAGMENTATION

    Compute the genre distribution for a given list of Items.
    """
    n = len(items)
    sum_one_over_ranks = harmonic_number(n)
    count = 0
    distr = {}
    for indx, item in enumerate(items):
        rank = indx + 1
        story_freq = distr.get(item, 0.0)
        print(1 / rank / sum_one_over_ranks)
        distr[item] = (
            story_freq + 1 * 1 / rank / sum_one_over_ranks
            if adjusted
            else story_freq + 1 * 1 / n
        )
        count += 1

    return distr


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
    """
    distribution = {}
    if not np.any(weights):
        weights = np.ones(len(a)) / len(a)
    # Compute:
    for item, weight in zip(a, weights):
        distribution[item] = weight + distribution.get(item, 0.0)
    return distribution


a = np.array(["a", "b", "c", "c"])
weights_jk = np.array(
    [1 / harmonic_number(val) for i, val in enumerate(range(1, len(a) + 1))]
)
weights_radio = np.array(
    [1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)]
)

compute_distribution(a, weights=weights_jk)
compute_distribution(a, weights=weights_radio)
compute_distr(a, True)

compute_distribution(a, distribution={})
compute_distr(a, False)
