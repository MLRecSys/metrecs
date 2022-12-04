import math
import pandas as pd
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, List
from metrecs.utils import harmonic_number

#
import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
import math

from metrecs.utils import compute_distribution

from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon


def JSD_root(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    # return math.sqrt(0.5 * (KL(_P, _M) + KL(_Q, _M)))
    # added the abs to catch situations where the disocunting causes a very small <0 value, check this more!!!!
    try:
        jsd_root = math.sqrt(
            abs(0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2)))
        )
    except ZeroDivisionError:
        print(P)
        print(Q)
        print()
        jsd_root = None
    return jsd_root


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


def user_level_fragmentation_categorical(
    user_items: np.ndarray[str],
    other_recommendations: np.ndarray[np.ndarray[str]],
    positional_weights: np.ndarray[float] = [],
) -> List[float]:
    """
    Args:
        user_items (np.ndarray[np.ndarray]): _description_
        other_recommendations (np.ndarray[np.ndarray]): _description_
    Returns:
        float: _description_
    len(user_items) == other_recommendations.shape[1]
    """
    s = compute_distribution(user_items, weighs=positional_weights)
    frag = []
    for user in other_recommendations:
        q = compute_distribution(user, weighs=positional_weights)
        ss, qq = avoid_distribution_misspecification(s, q)
        # frag.append(jensenshannon(list(ss.values()), list(qq.values()), base=2))
        frag.append(JSD_root(list(ss.values()), list(qq.values())))
    return frag


def model_level_fragmentation_categorical(
    recommendation: np.ndarray[np.ndarray[str]],
) -> List[float]:
    """_summary_
    Args:
        recommendation (np.ndarray[np.ndarray]): _description_
    Returns:
        float: _description_
    """
    indecies = np.arange(recommendation.shape[0])
    model_lvl = []
    for row_index in range(recommendation.shape[0]):
        rec = recommendation[row_index]
        others = recommendation[indecies != row_index, :]
        model_lvl.append(np.mean(user_level_fragmentation_categorical(rec, others)))
    return model_lvl


def user_level_calibration_categorical(
    user_items: np.ndarray,
    users_history_items: np.ndarray,
    user_items_weights: np.ndarray = [],
    users_history_items_weights: np.ndarray = [],
) -> float:
    """_summary_
    Args:
        user_items (np.ndarray[np.ndarray]): _description_
        other_recommendations (np.ndarray[np.ndarray]): _description_
    Returns:
        float: _description_
    len(users_history_items) == len(users_history_items_weights)
    len(user_items) == len(user_items_weights)
    """
    s = compute_distribution(user_items, weights=user_items_weights)
    q = compute_distribution(users_history_items, weights=users_history_items_weights)
    ss, qq = avoid_distribution_misspecification(s, q)
    # return jensenshannon(list(ss.values()), list(qq.values()), base=2)
    return JSD_root(list(ss.values()), list(qq.values()))


# user_level_calibration_categorical == user_level_representation_categorical


def user_level_representation_categorical(
    user_item_representations: np.ndarray,
    pool_item_representations: np.ndarray,
    user_item_representations_weights: np.ndarray = [],
    pool_item_representations_weights: np.ndarray = [],
) -> float:
    """_summary_
    Args:
        user_items (np.ndarray[np.ndarray]): _description_
        other_recommendations (np.ndarray[np.ndarray]): _description_
    Returns:
        float: _description_
    len(users_history_items) == len(users_history_items_weights)
    len(user_items) == len(user_items_weights)
    """
    s = compute_distribution(
        user_item_representations, weights=user_item_representations_weights
    )
    q = compute_distribution(
        pool_item_representations, weights=pool_item_representations_weights
    )
    ss, qq = avoid_distribution_misspecification(s, q)
    # return jensenshannon(list(ss.values()), list(qq.values()), base=2)
    return JSD_root(list(ss.values()), list(qq.values()))


def model_level_calibration_categorical(
    users: np.ndarray[np.ndarray[str]],
    users_history_items: np.ndarray[np.ndarray[str]],
    positional_weight_func: Callable = None,
) -> List[float]:
    calibration_scores = []
    for user, hist in zip(users, users_history_items):
        user_weights = (
            [] if not positional_weight_func else positional_weight_func(user_weights)
        )
        user_hist_weights = (
            [] if not positional_weight_func else positional_weight_func(hist)
        )
        calibration_scores.append(
            user_level_calibration_categorical(
                user, hist, user_weights, user_hist_weights
            )
        )
    return calibration_scores


def model_level_representation_categorical(
    user_item_representations: np.ndarray[np.ndarray[str]],
    pool_item_representations: np.ndarray[str],
    positional_weight_func: Callable = None,
) -> List[float]:
    """
    Args:
        user_item_representations (np.ndarray[np.ndarray[str]]): _description_
        pool_item_representations (np.ndarray[str]): _description_
        positional_weight_func (Callable, optional): _description_. Defaults to None.
    Returns:
        List[float]: _description_
    """
    pool_weights = (
        []
        if not positional_weight_func
        else positional_weight_func(pool_item_representations)
    )
    representations_scores = []
    for user in user_item_representations:
        user_weights = (
            [] if not positional_weight_func else positional_weight_func(user_weights)
        )
        representations_scores.append(
            user_level_representation_categorical(
                user, pool_item_representations, user_weights, pool_weights
            )
        )
    return representations_scores


a = np.array(["a", "b", "c", "c"])
b = np.array(["a", "b", "b", "d"])
s = compute_distribution(a)
q = compute_distribution(b)
ss, qq = avoid_distribution_misspecification(s, q)
jensenshannon(list(ss.values()), list(qq.values()), base=2)


user_recommendation = np.array(["a", "b", "c", "c"])
other_recommendations = np.array([["a", "b", "b", "d"], ["d", "q", "a", "a"]])


user_level_fragmentation_categorical(user_recommendation, other_recommendations)
recommendation = np.array(
    [["a", "b", "c", "c"], ["a", "q", "b", "d"], ["d", "q", "a", "a"]]
)
model_level_fragmentation_categorical(recommendation)

user_category = ["a", "b", "c", "c"]
user_history_category = ["a", "b", "c", "c", "q", "i", "p", "a"]

user_category = np.array([["a", "b", "c", "c"], ["a", "q", "d", "a"]])
user_history_category = np.array(
    [["a", "b", "c", "c", "q", "i", "p", "a"], ["a", "b", "c", "c", "q", "i", "p", "a"]]
)

user_level_calibration_categorical(user_category[0], user_history_category[0])
model_level_calibration_categorical(user_category, user_history_category)
