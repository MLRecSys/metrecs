from metrecs.metrics.intralist_diversity import intralist_diversity
from metrecs.utils import cosine_distances
import numpy as np

items = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

assert intralist_diversity(items, cosine_distances) == 1.0, ""
assert np.isnan(
    intralist_diversity(items[0], cosine_distances)
), "should not have been able to compute with only one input recommendation"
