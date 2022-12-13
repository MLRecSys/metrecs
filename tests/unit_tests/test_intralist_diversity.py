from metrecs.metrics.intralist_diversity import intralist_diversity
from metrecs.utils import cosine_distances
import numpy as np


def test_instralist_diversity():
    single_item = np.array([1, 0, 0])
    items_orthogonal = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    items = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]])

    assert intralist_diversity(items, cosine_distances) == 0.5976310729378175
    assert intralist_diversity(items_orthogonal, cosine_distances) == 1.0
    assert np.isnan(
        intralist_diversity(single_item, cosine_distances)
    ), "should not have been able to compute with only one input recommendation"
