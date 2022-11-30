from numpy.typing import ArrayLike
from metrecs.utils import cosine_distances
from typing import Callable
import numpy as np


def intralist_diversity(
    recommendations: ArrayLike[ArrayLike],
    distance_function: Callable = cosine_distances,
) -> float:
    """Smyth and McClave [2001]
    * Source: https://link.springer.com/chapter/10.1007/3-540-44593-5_25
    * Measure the intra-list diversity of a recommendation list R(|R|>1) as the average pairwise distance between recommendations:
        Diversity(R) = ( sum_{i∈R} sum_{j∈R\{i}} dist(i, j) )  / ( |R|(|R|-1) )

    Args:
        recommendations (ArrayLike[ArrayLike]): {array-like, sparse matrix} of shape (n_samples_X, n_features)
        distance_function (Callable): function to compute the pairwise distances of recommendations vector representations. Default is cosine_distances

    Returns:
        float: diversity score
    """
    if len(recommendations.shape) == 1:
        # Less than or equal to 1 recommendations in recommendation list
        diversity = np.nan
    else:
        R = recommendations.shape[0]
        pairwise_distances = distance_function(recommendations)
        diversity = np.sum(pairwise_distances) / (R * (R - 1))
    return diversity
