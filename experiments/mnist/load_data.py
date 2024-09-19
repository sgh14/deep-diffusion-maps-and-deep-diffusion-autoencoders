import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


def get_data(npoints=2000, test_size=0.5, seed=123, noise=0.5, n_classes=6):
    np.random.seed(seed)
    # Load the images
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    # Select n_classes first classes
    selection = y < n_classes
    X = X[selection]
    y = y[selection]
    # Scale pixels to [0, 1] interval
    X = X / 255.0
    # Shuffle the training data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    # Add white noise
    X = X + np.random.normal(loc=0.0, scale=noise, size=X.shape)
    # Clip the pixel values in the [0, 1] interval
    X = np.clip(X, 0.0, 1.0)
    # Select only the first npoints
    X = X[:npoints]
    y = y[:npoints]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train_small = X_train[::2]
    y_train_small = y_train[::2]

    data_train = (X_train, y_train)
    data_test = (X_test, y_test)
    data_train_small = (X_train_small, y_train_small)

    return data_train, data_test, data_train_small


def get_datasets(npoints=2000, test_size=0.5, seed=123, noise=1):
    data_clean_train, data_clean_test, data_clean_train_small = get_data(npoints, test_size, seed, 0)
    data_noisy_train, data_noisy_test, data_noisy_train_small = get_data(npoints, test_size, seed, noise)
    # Train-test datasets
    datasets_train = [
        data_clean_train_small, data_clean_train,
        data_noisy_train_small, data_noisy_train
    ]
    datasets_test = [
        data_clean_test, data_clean_test,
        data_noisy_test, data_noisy_test
    ]

    return datasets_train, datasets_test
