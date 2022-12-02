import math
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict
from metrecs.utils import harmonic_number

#
import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
import math

from metrecs.utils import compute_distribution

from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon


def avoid_distribution_misspecification(s: Dict, q: Dict, alpha=0.001) -> Dict:
    """ """
    merged_dic = np.unique(list(s) + list(q)).tolist()
    for key in sorted(merged_dic):
        q_score = q.get(key, 0.0)
        s_score = s.get(key, 0.0)
        s[key] = (1 - alpha) * s_score + alpha * q_score
        q[key] = (1 - alpha) * q_score + alpha * s_score
    # Sort
    q = {key: q[key] for key in sorted(q.keys())}
    s = {key: s[key] for key in sorted(s.keys())}
    return s, q


a = np.array(["a", "b", "c", "c"])
b = np.array(["a", "b", "b", "d"])
s = compute_distribution(a)
q = compute_distribution(b)
ss, qq = avoid_distribution_misspecification(s, q)
jensenshannon(list(ss.values()), list(qq.values()), base=2)


user_recommendation = np.array(["a", "b", "c", "c"])
other_recommendations = np.array([["a", "b", "b", "d"], ["d", "q", "a", "a"]])


def user_level_fragmentation(
    user_items: np.ndarray[np.ndarray],
    other_recommendations: np.ndarray[np.ndarray],
) -> float:
    s = compute_distribution(user_items)
    frag = []
    for user in other_recommendations:
        q = compute_distribution(user)
        ss, qq = avoid_distribution_misspecification(s, q)
        frag.append(jensenshannon(list(ss.values()), list(qq.values()), base=2))
    return frag


def model_level_fragmentation(recommendation: np.ndarray[np.ndarray]) -> float:
    indecies = np.arange(recommendation.shape[0])
    model_lvl = []
    for row_index in range(recommendation.shape[0]):
        rec = recommendation[row_index]
        others = recommendation[indecies != row_index, :]
        model_lvl.append(np.mean(user_level_fragmentation(rec, others)))
    return model_lvl


user_level_fragmentation(user_recommendation, other_recommendations)
recommendation = np.array(
    [["a", "b", "c", "c"], ["a", "q", "b", "d"], ["d", "q", "a", "a"]]
)
model_level_fragmentation(recommendation)
