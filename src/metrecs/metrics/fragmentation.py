import numpy as np
from numpy.typing import ArrayLike
from typing import Any, Dict
from metrecs.utils import histogram
from metrecs.utils import compute_kl_js_divergence
from scipy.spatial import distance
from collections import Counter

def fragmentation(
    ref: ArrayLike[ArrayLike[Any]], own: ArrayLike[Any], catelog: ArrayLike[Any]
) -> float:
    """
    Calculates to what extent users have been exposed to the same "item clusters".
    An "item cluster" is considered a set of items that are related.

    We measure Fragmentation as the average of Jensen-Shannon (JS) divergence between every pair of users' recommendations.
    Fragmentation tells us whether users exist in their own bubble of items or are part of a greater public sphere.

    For Kullback-Leibler (KL) and JS divergence, P(ref[u]) is the rank-aware distribution of "item clusters"
    for a reference user u, and Q(own) the same but for the current user.

    Args:
        ref: Two-dimensional array of ordered "item cluster" ids from the other users
        own: One-dimensional array of ordered "item cluster" ids of our current user
        catelog: One-dimensional array of possible "item cluster" in catelog
    """
    # TODO ASSERT: len(user_item_representaion) == other_users_item_representaion.shape[1]

    # add catelog:
    own_hist = histogram(own)
    # add catelog:
    ref_hist = np.apply_along_axis(histogram, 1, ref)
    _, jsd = np.mean([compute_kl_js_divergence(own_hist, p) for p in ref_hist])
    return jsd


# the difference in distribution between the issued recommendations (Q) and its context (P). Eac
# The context P may refer to either the overall supply of available items,
# the user profile, such as the reading history or explicitly stated preferences, or the recommendations that were issued to other users


def fragmentation(
    recommendations: ArrayLike[ArrayLike[str]], context: ArrayLike[str] = None
) -> float:
    """
    Calculates to what extent users have been exposed to the same "item clusters".
    An "item cluster" is considered a set of items that are related.

    We measure Fragmentation as the average of JS divergence between every pair of users' recommendations.
    Fragmentation tells us whether users exist in their own bubble of items or are part of a greater public sphere.

    For KL and JS divergence, P(ref[u]) is the rank-aware distribution of "item clusters" for a reference user u,
    and Q(own) the same but for the current user.

    Args:
        recommendations: Two-dimensional array of ordered "item cluster" ids from the other users

    Returns:
        float: fragmentation score
    """
    # TODO ASSERT: len(user_item_representaion) == other_users_item_representaion.shape[1]
    if not context:
        context = recommendations

    recommendation_distribution = np.apply_along_axis(histogram, 1, recommendations)
    context_distribution = np.apply_along_axis(histogram, 0, context)

    for index, recommendation in enumerate(recommendations):

        user_distribution = histogram(recommendations, rank_aware=True)
        context = np.delete(ref_hist, index, axis=0)

        jsd = np.mean([distance.jensenshannon(user_distribution, p) for p in ref_hist])
        print(jsd)
    return jsd



Counter(recommendations[0])

n = np.unique(recommendations[0], return_counts=True)[1]

own_hist = histogram(recommendations[0], rank_aware=False)
ref_hist = np.apply_along_axis(histogram, 1, recommendations)

jsd = [distance.jensenshannon(own_hist, p) for p in ref_hist]


distance.jensenshannon(own_hist, ref_hist)
