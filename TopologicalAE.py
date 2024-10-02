import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss, MeanSquaredError
from TopologicalLoss import TopologicalLoss


class LossNormalizer(Loss):
    def __init__(self, loss_fn, beta=0.9, **kwargs):
        super(LossNormalizer, self).__init__(**kwargs)
        self.loss_fn = loss_fn
        self.beta = beta
        self.loss_mean = tf.Variable(1.0, trainable=False)  # For scaling
    
    def call(self, y_true, y_pred):
        # Compute the loss
        loss = self.loss_fn(y_true, y_pred)
        # Detach the rolling mean (no gradient update)
        detached_loss_mean = tf.stop_gradient(self.loss_mean)
        # Dynamically normalize the loss
        normalized_loss = loss / (detached_loss_mean + 1e-8)
        # Update running mean
        loss_new_mean = self.beta*self.loss_mean + (1 - self.beta)*tf.reduce_mean(loss)
        self.loss_mean.assign(loss_new_mean)
        
        return normalized_loss


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
        inputs = Input(shape=self.encoder.input_shape[1:])
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(inputs=inputs, outputs=[encoded, decoded], name=name)
        self.autoencoder.summary()

  
    def compile(self, D, topological_weight=1.0, beta=0.9, **kwargs):
        """
        Compile the autoencoder model.

        This method sets up the loss functions and their weights.

        Args:
            D (array-like): Distance matrix for topological loss calculation.
            topological_weight (float, optional): Weight for the topological loss. Defaults to 1.0.
            **kwargs: Additional arguments to pass to the Keras compile method.
        """
        self.autoencoder.compile(
            loss=[
                LossNormalizer(TopologicalLoss(D), beta),
                LossNormalizer(MeanSquaredError(), beta)
            ],
            loss_weights=[
                topological_weight,
                (1 - topological_weight)
            ],
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
            y=[indices, X],
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
