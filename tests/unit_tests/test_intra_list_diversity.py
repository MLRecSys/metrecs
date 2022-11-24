from metrecs.metrics.intra_list_diversity import intra_list_diversity
from metrecs.utils import cosine_distances
import numpy as np

items = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

assert intra_list_diversity(items, cosine_distances) == 1.0, ""
assert np.isnan(
    intra_list_diversity(items[0], cosine_distances)
), "should not have been able to compute with only one input recommendation"
