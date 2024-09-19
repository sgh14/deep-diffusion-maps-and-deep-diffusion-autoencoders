import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split

from experiments.swiss_roll.plot_results import my_colormap2D


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_data(npoints=2000, test_size=0.5, seed=123, noise=0):
    X, color = make_swiss_roll(npoints, random_state=seed, noise=noise)
    dimension_1 = normalize(color)
    dimension_2 = normalize(X[:, 1])
    y = np.array([my_colormap2D(x, y) for x, y in zip(dimension_1, dimension_2)])
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
