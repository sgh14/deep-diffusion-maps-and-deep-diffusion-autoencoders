import numpy as np
from os import path

from DiffusionLoss import DiffusionLoss
from aux_functions import get_sigma
from experiments.mnist.plot_results import plot_original, plot_projection, plot_interpolations, plot_history
from experiments.mnist.load_data import get_datasets
from experiments.mnist.metrics import compute_metrics
from experiments.utils import build_encoder, build_decoder


root = 'experiments/mnist/results/DDM'

q_vals = [0.005, 0.005, 0.005, 0.005]
steps_vals = [100, 100, 100, 100]
alpha_vals = [1, 1, 1, 1]
kernel = 'rbf'

titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]

datasets_train, datasets_test = get_datasets(npoints=10000, test_size=0.1, seed=123, noise=0.25)

for (X, y), title in zip(datasets_train, titles):
    plot_original(
        X, y, title, path.join(root, 'train_orig'), images_per_class=2, grid_shape=(3, 4)
    )

for (X, y), title in zip(datasets_test, titles):
    plot_original(
        X, y, title, path.join(root, 'test_orig'), images_per_class=2, grid_shape=(3, 4)
    )

for i in range(len(titles)):
    q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
    X_train, y_train = datasets_train[i]
    X_test, y_test = datasets_test[i]
    title = titles[i]

    img_shape = X_train.shape[1:]
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    
    sigma = get_sigma(X_train, q)
    encoder = build_encoder(input_shape=(X_train.shape[-1],), units=128, n_components=2)
    loss = DiffusionLoss(X_train, sigma=sigma, steps=steps, kernel=kernel, alpha=alpha)
    encoder.compile(optimizer='adam', loss=loss)
    indices = np.array(list(range(len(X_train))))
    hist_enc = encoder.fit(X_train, indices, epochs=300, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)
    X_train_red = encoder(X_train)
    X_test_red = encoder(X_test)

    decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
    decoder.compile(optimizer='adam', loss='mse')
    hist_dec = decoder.fit(X_train_red, X_train, epochs=300, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)
    X_train_rec = decoder(X_train_red)
    X_test_rec = decoder(X_test_red)
    
    X_train_rec = X_train_rec.numpy().reshape((X_train_rec.shape[0], *img_shape))
    X_test_rec = X_test_rec.numpy().reshape((X_test_rec.shape[0], *img_shape))

    plot_projection(X_train_red, y_train, title, path.join(root, 'train_red'))
    plot_original(
        X_train_rec, y_train, title,
        path.join(root, 'train_rec'),
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_projection(X_test_red, y_test, title, path.join(root, 'test_red'))
    plot_original(
        X_test_rec, y_test, title,
        path.join(root, 'test_rec'),
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_interpolations(
        X_test_red, y_test, title,
        decoder,
        path.join(root, 'test_interp'),
        img_shape,
        class_pairs = [(i, i+1) for i in range(0, 6, 2)],
        n_interpolations=4
    )
    plot_history(hist_enc, path.join(root, 'histories', 'encoder', title), log_scale=True)
    plot_history(hist_dec, path.join(root, 'histories', 'decoder', title), log_scale=True)

    compute_metrics(X_test, X_test_red, X_test_rec, y_test, title, root)        
