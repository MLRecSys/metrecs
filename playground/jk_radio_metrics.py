import numpy as np
from typing import Any
from numpy.typing import ArrayLike
from scipy.special import kl_div

from metrecs.utils import histogram

# ALL NEED CATELOG for distribution
# CATELOG ALL ITEMS USED TOTAL
# return jsd


def fragmentation(
    user_item_representaion: ArrayLike[Any],
    other_users_item_representaion: ArrayLike[ArrayLike[Any]],
    catelog: ArrayLike[Any],
):
    """
    input:
        user_item_representaion: subset array of catelog
        other_users_item_representaion: array with arrays; each array is a subset of catelog

        ASSERT:
            len(user_item_representaion) == other_users_item_representaion.shape[1]
    """

    own_hist = histogram(user_item_representaion)  # add catelog
    # add catelog
    ref_hist = np.apply_along_axis(histogram, 1, other_users_item_representaion)
    [kl, jsd] = np.mean([compute_kl_divergence(own_hist, p) for p in ref_hist])
    return (kl, jsd)


def calibration(
    user_item_representaion: ArrayLike, user_history: ArrayLike, catelog: ArrayLike
):
    """
    input
        1d array (items) subset of catelog
        1d array (hist) subset of catelog
        1d array (catelog) ALL dem items
    """
    own_hist = histogram(user_item_representaion)  # add catelog
    ref_hist = histogram(user_history)  # add catelog

    [kl, jsd] = np.mean([compute_kl_divergence(own_hist, p) for p in ref_hist])
    return (kl, jsd)


def representation(user_item_representaion, catelog):
    """
    input
        1d array (items) subset of catelog (['a', 'b', 'c'])
        1d array (catelog) ALL possible items ['a', 'b', 'c', 'd', 'e']
    """
    own_hist = histogram(user_item_representaion)  # add catelog
    ref_hist = histogram(catelog)  # add catelog
    [kl, jsd] = np.mean([compute_kl_divergence(own_hist, p) for p in ref_hist])
    return (kl, jsd)
