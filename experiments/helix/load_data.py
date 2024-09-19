import numpy as np
from sklearn.model_selection import train_test_split

from experiments.helix.plot_results import my_colormap1D


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_data(npoints=2000, test_size=0.5, seed=123, noise=0):
    np.random.seed(seed)
    theta = np.linspace(0, 2*np.pi, npoints)
    eps = np.random.normal(loc=0, scale=noise, size=(3, npoints))
    x1 = np.cos(theta) + eps[0]
    x2 = np.sin(2*theta) + eps[1]
    x3 = np.sin(3*theta) + eps[2]
    
    X = np.stack((x1, x2, x3), axis=1)
    # TODO: another option is to reparametrize the curve using its natural parameter (curve length)
    y = theta
    y = np.array([my_colormap1D(t) for t in normalize(theta)])
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
