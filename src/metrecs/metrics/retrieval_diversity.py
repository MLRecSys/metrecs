import numpy as np


def retrieval_diversity(items: np.ndarray, distance_function) -> float:
    """Smyth and McClave [2001]
    * Source: https://link.springer.com/chapter/10.1007/3-540-44593-5_25
    * Measure the intra-list diversity of a recommendation list R(|R|>1) as the average pairwise distance between items:
        Diversity(R) = ( sum_{i∈R} sum_{j∈R\{i}} dist(i, j) )  / ( |R|(|R|-1) )

    Args:
        items (np.ndarray): Array-like with item vector representations.
        distance_function (_type_): function to compute the pairwise distances of items vector representations

    Returns:
        float: retrieval_diversity score
    """
    pairwise_distances = distance_function(items)
    R = items.shape[0]

    if R <= 1:
        diversity = np.nan
    else:
        diversity = np.sum(pairwise_distances) / (R * (R - 1))
    return diversity
