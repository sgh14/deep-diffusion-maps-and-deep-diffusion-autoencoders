from TopologicalAE import TopologicalAE
from DiffusionLoss import DiffusionLoss


class DeepDiffusionAE(TopologicalAE):
    """
    DeepDiffusionAE is a specialized autoencoder that incorporates a diffusion loss
    component. This class inherits from TopologicalAE.

    Methods
    -------
    compile(X, sigma=None, steps=1, kernel='rbf', precomputed_distances=False, diffusion_weight=1.0, **kwargs):
        Compiles the autoencoder model with a specified diffusion loss.
    """

    def compile(
        self,
        X,
        sigma=None,
        steps=1,
        kernel='rbf',
        alpha=1,
        precomputed_distances=False,
        diffusion_weight=1.0,
        **kwargs
    ):
        """
        Compiles the autoencoder model with diffusion loss and mean squared error (MSE) loss.

        Parameters
        ----------
        X : array-like
            The input data used for calculating the diffusion loss.
        sigma : float, optional
            The bandwidth parameter for the RBF kernel. If None, it will be estimated.
        steps : int, optional
            The number of diffusion time steps.
        kernel : str, optional
            The type of kernel to use for diffusion (default is 'rbf').
        alpha : float, optional (default=1)
            Normalization factor.
        precomputed_distances : bool, optional
            If True, indicates that the distances are precomputed.
        diffusion_weight : float, optional
            The weight of the diffusion loss in the total loss function.
        **kwargs : dict
            Additional keyword arguments passed to the compile method of the underlying autoencoder.
        """
        # Create an instance of DiffusionLoss with the provided parameters.
        diffusion_loss = DiffusionLoss(
            X,
            sigma=sigma,
            steps=steps,
            kernel=kernel,
            alpha=alpha,
            precomputed_distances=precomputed_distances
        )
        
        # Compile the autoencoder model with specified losses and loss weights.
        self.autoencoder.compile(
            loss={'encoder': diffusion_loss, 'decoder': 'mse'},  # Define loss for encoder and decoder.
            loss_weights={'encoder': diffusion_weight, 'decoder': 1.0},  # Set loss weights for encoder and decoder.
            **kwargs  # Additional arguments passed to the compile method.
        )
