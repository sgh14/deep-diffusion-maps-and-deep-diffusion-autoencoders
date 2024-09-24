import numpy as np
from sklearn.model_selection import train_test_split
from skfda import datasets


def get_data(test_size=0.5, seed=123, noise=0.5):
    np.random.seed(seed)
    X, y = datasets.fetch_phoneme(return_X_y=True)
    new_points = X.grid_points[0][:150]
    new_data = X.data_matrix[:, :150]
    domain_range=(np.min(new_points), np.max(new_points))
    X = X.copy(
        grid_points=new_points,
        data_matrix=new_data,
        domain_range=domain_range,
    )
    X = X(np.linspace(*domain_range, 128))
    X = X + np.random.normal(loc=0.0, scale=noise, size=X.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed, shuffle=True
    )
    
    X_train_small = X_train[::2]
    y_train_small = y_train[::2]

    data_train = (X_train, y_train)
    data_test = (X_test, y_test)
    data_train_small = (X_train_small, y_train_small)

    return data_train, data_test, data_train_small


def get_datasets(test_size=0.5, seed=123, noise=1):
    data_clean_train, data_clean_test, data_clean_train_small = get_data(test_size, seed, 0)
    data_noisy_train, data_noisy_test, data_noisy_train_small = get_data(test_size, seed, noise)
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
