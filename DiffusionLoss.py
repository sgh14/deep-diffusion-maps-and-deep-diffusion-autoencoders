import tensorflow as tf
import numpy as np
from numba import njit
from TopologicalLoss import TopologicalLoss
from kernels import rbf_kernel, laplacian_kernel


class DiffusionLoss(TopologicalLoss):
    """
    A class to compute the Diffusion Loss, which is a type of Topological Loss.
    
    This loss is based on diffusion distances, which capture the connectivity of data points
    in a manifold by simulating a diffusion process.
    """

    def __init__(
        self,
        X,
        sigma=None,
        steps=1,
        kernel='rbf',
        alpha=1,
        precomputed_distances=False,
        name="diffusion_loss"
    ):
        """
        Initialize the DiffusionLoss.

        Args:
            X (np.array): The input data or precomputed distances.
            sigma (float, optional): The bandwidth parameter for the kernel. Defaults to None.
            steps (int, optional): The number of time steps for the diffusion process. Defaults to 1.
            kernel (str, optional): The type of kernel to use ('rbf' or 'laplacian'). Defaults to 'rbf'.
            alpha (float, optional): Normalization factor. Defaults to 1.
            precomputed_distances (bool, optional): Whether X contains precomputed distances. Defaults to False.
            name (str, optional): Name of the loss function. Defaults to "diffusion_loss".
        """
        # Store the diffusion distances for the training data
        if precomputed_distances:
            D = X
        else:
            # Compute the kernel matrix
            K = self.get_kernel(X, X, sigma, kernel, alpha)
            # Compute diffusion probabilities and degree vector
            P, d = self.diffusion_probabilities(K, steps)
            # Compute diffusion distances
            D = self.diffusion_distances(P, d)
        
        super().__init__(D=D, name=name)


    @staticmethod
    def get_kernel(X, Y, sigma, kernel, alpha):
        """
        Compute the kernel matrix.

        Args:
            X (np.array): First set of points.
            Y (np.array): Second set of points.
            sigma (float): Bandwidth parameter.
            kernel (str): Type of kernel ('rbf' or 'laplacian').
            alpha (float): Normalization factor.

        Returns:
            np.array: The computed kernel matrix.
        """
        gamma = 1 / (2 * sigma ** 2)
        if kernel == 'laplacian':
            K = laplacian_kernel(X, Y, gamma=gamma)
        elif kernel == 'rbf':
            K = rbf_kernel(X, Y, gamma=gamma)
        else:
            raise ValueError("Unsupported kernel")

        d_i_alpha = np.sum(K, axis=1)**alpha
        # D_i_alpha_inv = np.diag(d_i ** (-1))
        d_j_alpha = np.sum(K, axis=0)**alpha
        # D_j_alpha_inv = np.diag(d_j ** (-1))
        # Compute k_ij/(d_i^alpha * d_j^alpha)
        K_alpha = K/np.outer(d_i_alpha, d_j_alpha) # D_i_alpha_inv @ K @ D_j_alpha_inv

        return K_alpha
    

    @staticmethod
    @njit
    def diffusion_probabilities(K, steps=1):
        """
        Compute diffusion probabilities.

        Args:
            K (np.array): Kernel matrix.
            t (int, optional): Number of time steps. Defaults to 1.

        Returns:
            tuple: (P, d) where P is the diffusion probability matrix and d is the degree vector.
        """
        d_i = np.sum(K, axis=1)
        # D_i_inv = np.diag(d_i ** (-1))
        # Compute k_ij^{(alpha)}/d_i^{(alpha)}
        P = K / d_i[:, np.newaxis] # D_i_inv @ K 
        # Compute P^t if t > 1
        if steps > 1:
            P = np.linalg.matrix_power(P, steps)

        return P, d_i


    @staticmethod
    @njit
    def diffusion_distances(P, d):
        """
        Compute diffusion distances.

        Args:
            P (np.array): Diffusion probability matrix.
            d (np.array): Degree vector.

        Returns:
            np.array: Matrix of diffusion distances.
        """
        D = np.zeros(P.shape)
        # Compute stationary distribution
        pi = d / np.sum(d)
        for i in range(P.shape[0]):
            for j in range(i+1, P.shape[1]):
                # Compute diffusion distance between points i and j
                D_ij = np.sqrt(np.sum(((P[i, :] - P[j, :])**2) / pi))
                # Store the distance (matrix is symmetric)
                D[i, j] = D_ij
                D[j, i] = D_ij

        return D
