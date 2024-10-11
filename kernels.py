import numpy as np
from sklearn.metrics import pairwise_distances


def rbf_kernel(X, Y=None, gamma=None):
    """
    Compute the Radial Basis Function (RBF) kernel between each pair of samples in X and Y using pairwise_distances.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples_X, n_features)
        Input data where n_samples_X is the number of samples and n_features is the number of features.
    Y : numpy.ndarray, shape (n_samples_Y, n_features), optional
        Input data where n_samples_Y is the number of samples and n_features is the number of features.
        If None, use X for both arguments.
    gamma : float, optional
        The gamma parameter for the RBF kernel. Defaults to 1/n_features if None.

    Returns
    -------
    K : numpy.ndarray, shape (n_samples_X, n_samples_Y)
        The RBF kernel matrix.
    """
    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Compute pairwise squared Euclidean distances
    distances = pairwise_distances(X, Y, metric='sqeuclidean')
    
    # Apply the RBF kernel
    K = np.exp(-gamma * distances)

    return K


def laplacian_kernel(X, Y=None, gamma=None):
    """
    Compute the Laplacian kernel between each pair of samples in X and Y using pairwise_distances.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples_X, n_features)
        Input data where n_samples_X is the number of samples and n_features is the number of features.
    Y : numpy.ndarray, shape (n_samples_Y, n_features), optional
        Input data where n_samples_Y is the number of samples and n_features is the number of features.
        If None, use X for both arguments.
    gamma : float, optional
        The gamma parameter for the Laplacian kernel. Defaults to 1/n_features if None.

    Returns
    -------
    K : numpy.ndarray, shape (n_samples_X, n_samples_Y)
        The Laplacian kernel matrix.
    """
    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Compute pairwise Manhattan distances (L1 norm)
    distances = pairwise_distances(X, Y, metric='manhattan')

    # Apply the Laplacian kernel function
    K = np.exp(-gamma * distances)

    return K
