from metrecs.metrics.retrieval_diversity import retrieval_diversity
from metrecs.utils import cosine_distances
import numpy as np

items = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

assert retrieval_diversity(items, cosine_distances) == 1.0, ""
assert np.isnan(retrieval_diversity(items[0], cosine_distances)), ""
