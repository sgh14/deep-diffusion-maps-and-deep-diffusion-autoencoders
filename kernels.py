import numpy as np
from numba import njit


@njit
def rbf_kernel(X, Y, gamma=None):
    """
    Compute the Radial Basis Function (RBF) kernel between each pair of samples in X and Y.

    The RBF kernel is defined as:
    K(x, y) = exp(-gamma * ||x - y||^2)
    where ||x - y||^2 is the squared Euclidean distance between the vectors x and y.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples_X, n_features)
        Input data where n_samples_X is the number of samples and n_features is the number of features.
    Y : numpy.ndarray, shape (n_samples_Y, n_features)
        Input data where n_samples_Y is the number of samples and n_features is the number of features.
    gamma : float
        The gamma parameter for the RBF kernel. This is a non-negative parameter that controls
        the width of the Gaussian kernel. If None, default to 1/n_features.

    Returns
    -------
    K : numpy.ndarray, shape (n_samples_X, n_samples_Y)
        The RBF kernel matrix.
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Compute the squared Euclidean distances
    X_norm = np.sum(X ** 2, axis=-1)[:, np.newaxis]
    Y_norm = np.sum(Y ** 2, axis=-1)[np.newaxis, :]
    distances = X_norm + Y_norm - 2 * np.dot(X, Y.T)

    # Apply the RBF kernel function
    K = np.exp(-gamma * distances)

    return K


@njit
def laplacian_kernel(X, Y, gamma=None):
    """
    Compute the Laplacian kernel between each pair of samples in X and Y.

    The Laplacian kernel is defined as:
    K(x, y) = exp(-gamma * ||x - y||_1)
    where ||x - y||_1 is the L1 norm (Manhattan distance) between the vectors x and y.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples_X, n_features)
        Input data where n_samples_X is the number of samples and n_features is the number of features.
    Y : numpy.ndarray, shape (n_samples_Y, n_features)
        Input data where n_samples_Y is the number of samples and n_features is the number of features.
    gamma : float
        The gamma parameter for the Laplacian kernel. This is a non-negative parameter that controls
        the width of the kernel.

    Returns
    -------
    K : numpy.ndarray, shape (n_samples_X, n_samples_Y)
        The Laplacian kernel matrix.
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    n_samples_X, n_samples_Y = X.shape[0], Y.shape[0]
    K = np.zeros((n_samples_X, n_samples_Y), dtype=np.float64)

    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            distance = np.sum(np.abs(X[i] - Y[j]))
            K[i, j] = np.exp(-gamma * distance)

    return K
