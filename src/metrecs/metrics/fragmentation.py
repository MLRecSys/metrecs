import numpy as np
from scipy.special import kl_div

from metrecs.utils import histogram
from metrecs.utils import compute_kl_divergence

def fragmentation(ref, own):
    """

    """
    # TODO: for vectorization
    # own_recs_hist_rep = np.tile(own_recs_hist, (pop_recs.shape[0], 1))

    own_hist = histogram(own)
    ref_hist = np.apply_along_axis(histogram, 1, ref)
    [kl, jsd] = np.mean([compute_kl_divergence(own_hist, p) for p in ref_hist])
    return(kl, jsd)
