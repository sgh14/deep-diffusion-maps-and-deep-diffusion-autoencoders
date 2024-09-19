import numpy as np
from sklearn.metrics import pairwise_distances

def get_sigma(X, q=0.5):
    distances = pairwise_distances(X)
    distances = distances.flatten()
    sigma = np.quantile(distances[distances > 0], q)

    return sigma
