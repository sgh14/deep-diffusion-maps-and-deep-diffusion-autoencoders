import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from TopologicalLoss import TopologicalLoss


class TopologicalAE:
    """
    A class representing a Topological Autoencoder.

    This autoencoder incorporates a topological loss term in addition to
    the standard reconstruction loss, aiming to preserve topological 
    structures in the encoded space.
    """

    def __init__(self, encoder, decoder, name=None):
        """
        Initialize the Topological Autoencoder.

        Args:
            encoder (keras.Model): The encoder part of the autoencoder.
            decoder (keras.Model): The decoder part of the autoencoder.
            name (str, optional): Name for the autoencoder model.
        """
        # Set up the encoder
        self.encoder = encoder
        self.encoder.summary()

        # Set up the decoder
        self.decoder = decoder
        self.decoder.summary()

        # Construct the full autoencoder
        inputs = Input(shape=encoder.layers[0].input_shape[1:])
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(inputs, [encoded, decoded], name=name)
        self.autoencoder.summary()

  
    def compile(self, D, topological_weight=1.0, **kwargs):
        """
        Compile the autoencoder model.

        This method sets up the loss functions and their weights.

        Args:
            D (array-like): Distance matrix for topological loss calculation.
            topological_weight (float, optional): Weight for the topological loss. Defaults to 1.0.
            **kwargs: Additional arguments to pass to the Keras compile method.
        """
        self.autoencoder.compile(
            loss={'encoder': TopologicalLoss(D), 'decoder': 'mse'},
            loss_weights={'encoder': topological_weight, 'decoder': 1.0},
            **kwargs
        )

  
    def fit(self, X, **kwargs):
        """
        Fit the autoencoder to the input data.

        Args:
            X (array-like): The input data to train on.
            **kwargs: Additional arguments to pass to the Keras fit method.

        Returns:
            history: A History object containing training details.
        """
        # Generate indices for topological loss calculation
        indices = np.array(list(range(X.shape[0])))
        
        # Train the model
        history = self.autoencoder.fit(
            x=X,
            y={'encoder': indices, 'decoder': X},  # Use indices for encoder, original data for decoder
            **kwargs
        )

        return history

  
    def encode(self, X):
        """
        Encode the input data using the encoder part of the autoencoder.

        Args:
            X (array-like): The input data to encode.

        Returns:
            array-like: The encoded representation of the input data.
        """
        X_red = self.encoder(X)
      
        return X_red

  
    def decode(self, X_red):
        """
        Decode the encoded data using the decoder part of the autoencoder.

        Args:
            X_red (array-like): The encoded data to decode.

        Returns:
            array-like: The reconstructed data.
        """
        X_rec = self.decoder(X_red)
      
        return X_rec
