from metrecs.utils import compute_distribution, harmonic_number
import numpy as np


def compute_distr(items, adjusted=False):
    """FRAGMENTATION
    Compute the genre distribution for a given list of Items.
    """
    n = len(items)
    sum_one_over_ranks = harmonic_number(n)
    count = 0
    distr = {}
    for indx, item in enumerate(items):
        rank = indx + 1
        story_freq = distr.get(item, 0.0)
        distr[item] = (
            story_freq + 1 * 1 / rank / sum_one_over_ranks
            if adjusted
            else story_freq + 1 * 1 / n
        )
        count += 1
    return distr


def test_compute_distribution():
    a = np.array(["a", "b", "c", "c"])
    weights_radio = np.array(
        [1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)]
    )
    assert compute_distribution(a, weights=weights_radio) == compute_distr(a, True)
    assert compute_distribution(a, distribution={}) == compute_distr(a, False)
