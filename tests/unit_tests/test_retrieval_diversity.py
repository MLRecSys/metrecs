from metrecs.metrics.retrieval_diversity import retrieval_diversity
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

item_representation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
item_representation = np.array([[1,0], [1, 0]])

np.linalg.norm(item_representation)
np.linalg.norm(np.array([1,0])-np.array([1,0]))

retrieval_diversity(item_representation, cosine_distances)
retrieval_diversity(item_representation, np.linalg.norm)
