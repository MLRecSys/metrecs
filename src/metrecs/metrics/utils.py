from typing import Dict, List, Iterable
from collections import Counter
import numpy as np


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
