import tensorflow as tf


class TopologicalLoss(tf.keras.losses.Loss):
    """
    Custom Keras loss function that calculates the loss based on the difference 
    between topological distances and Euclidean distances in the embedded space.

    Attributes:
        D (tf.Tensor): Precomputed topological distances between training samples.
        name (str): Name of the loss function.
    """
    
    def __init__(self, D, name="topological_loss"):
        """
        Initializes the TopologicalLoss instance.

        Args:
            D (numpy.ndarray or tf.Tensor): Precomputed topological distances.
            name (str): Optional. Name of the loss function. Defaults to "topological_loss".
        """
        super().__init__(name=name)
        # Store the topological distances for the training data
        self.D = tf.constant(D, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Calculates the loss by comparing topological distances with Euclidean distances
        in the embedded space.

        Args:
            y_true (tf.Tensor): True labels or indices of the samples.
            y_pred (tf.Tensor): Predicted embeddings of the samples.

        Returns:
            tf.Tensor: Computed loss value.
        """
        # Cast y_true to integer type for indexing
        indices = tf.cast(y_true, tf.int32)
        X_red = y_pred

        # Get the number of samples in the batch
        n = tf.shape(indices)[0]

        # Create a meshgrid of indices and create a mask for the upper triangular part
        i, j = tf.meshgrid(tf.range(n), tf.range(n), indexing='ij')
        mask = tf.greater(j, i)

        # Apply the mask to get the indices of the upper triangular part
        i = tf.boolean_mask(i, mask)
        j = tf.boolean_mask(j, mask)

        # Get the topological distances between batch samples using the provided indices
        pairs_ij = tf.stack([tf.gather(indices, i), tf.gather(indices, j)], axis=1)
        pairs_ij = tf.squeeze(pairs_ij)
        D_ij = tf.gather_nd(self.D, pairs_ij)

        # Get the Euclidean distances between batch samples in the embedded space
        X_i = tf.gather(X_red, i)
        X_j = tf.gather(X_red, j)
        d_ij = tf.norm(X_i - X_j, axis=1)

        # Compute Mean Squared Error (MSE) between topological and Euclidean distances
        loss = tf.reduce_mean(tf.square(D_ij - d_ij))

        return loss
