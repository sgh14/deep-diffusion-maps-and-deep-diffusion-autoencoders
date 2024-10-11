import numpy as np
from scipy.spatial.distance import pdist


def get_sigma(X, q=0.5):
    """
    Compute the sigma value as the q-th quantile of pairwise distances.
    
    Parameters:
    X (ndarray): An (n_samples, n_features) array of data points.
    q (float): The quantile to use for sigma (default is 0.5).
    
    Returns:
    sigma (float): The computed sigma value.
    """
    # Option 1: Use pdist for efficiency (gives a 1D array of distances)
    distances = pdist(X, metric='euclidean')
    
    # Compute the q-th quantile, avoid zeros because there are no self-distances in pdist
    sigma = np.quantile(distances, q)
    
    return sigma