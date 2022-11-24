import numpy as np
from scipy.special import kl_div

import histogram

def fragmentation(ref, own):
    """

    """
    # TODO: for vectorization
    # own_recs_hist_rep = np.tile(own_recs_hist, (pop_recs.shape[0], 1))

    own_hist = histogram(own)
    ref_hist = np.apply_along_axis(histogram, 1, ref)
    div = np.mean([kl_div(own_hist, p) for p in ref_hist])
    return(div)
