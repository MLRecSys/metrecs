import numpy as np
from numpy.typing import ArrayLike

from metrecs.utils import histogram
from metrecs.utils import compute_kl_js_divergence


def fragmentation(ref: ArrayLike[ArrayLike[str]], own: ArrayLike[str]) -> float:
    """
    Calculates to what extent users have been exposed to the same "item clusters".
    An "item cluster" is considered a set of items that are related.

    We measure Fragmentation as the average of JS divergence between every pair of users’ recommendations.
    Fragmentation tells us whether users exist in their own bubble of items or are part of a greater public sphere.

    For KL and JS divergence, 𝑃(ref[u]) is the rank-aware distribution of "item clusters" for a reference user u,
    and 𝑄(own) the same but for the current user.

    Args:
        ref: Two-dimensional array of ordered "item cluster" ids from the other users
        own: One-dimensional array of ordered "item cluster" ids of our current user

    """

    own_hist = histogram(own)
    ref_hist = np.apply_along_axis(histogram, 1, ref)
    _, jsd = np.mean([compute_kl_js_divergence(own_hist, p) for p in ref_hist])
    return jsd
