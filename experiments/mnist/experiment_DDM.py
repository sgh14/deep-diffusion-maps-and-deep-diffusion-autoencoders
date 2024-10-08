import time
import numpy as np
import tensorflow as tf
from os import path

from DiffusionLoss import DiffusionLoss
from aux_functions import get_sigma
from experiments.mnist.plot_results import plot_original, plot_projection, plot_interpolations, plot_history
from experiments.mnist.load_data import get_datasets
from experiments.mnist.metrics import compute_metrics
from experiments.utils import build_conv_encoder, build_conv_decoder


seed = 123
tf.random.set_seed(seed)
root = 'experiments/mnist/results/DDM'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]

datasets_train, datasets_test = get_datasets(npoints=10000, test_size=0.1, seed=seed, noise=0.25)

q_vals = [0.0075, 0.0075, 0.0075, 0.0075]
steps_vals = [1, 1, 1, 1]
alpha_vals = [0, 0, 0, 0]
kernel = 'rbf'

for i in range(len(titles)):
    q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
    experiment = f'percentile_{q}-steps_{steps}-alpha_{alpha}'
    X_train, y_train = datasets_train[i]
    X_test, y_test = datasets_test[i]
    # Añadimos la dimensión de canal
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    title = titles[i]
    sigma = get_sigma(X_train.reshape((X_train.shape[0], -1)), q)
    
    print(experiment, '-', title)  
    encoder = build_conv_encoder(input_shape=X_train.shape[1:], filters=8, n_components=2, zero_padding=(2, 2), dropout=0.2)
    tic = time.perf_counter()
    loss = DiffusionLoss(X_train.reshape((X_train.shape[0], -1)), sigma=sigma, steps=steps, kernel=kernel, alpha=alpha)
    encoder.compile(optimizer='adam', loss=loss)
    indices = np.array(list(range(len(X_train))))
    hist_enc = encoder.fit(X_train, indices, epochs=100, validation_split=0.1, shuffle=False, batch_size=64, verbose=0)
    X_train_red = encoder(X_train)
    tac = time.perf_counter()
    X_test_red = encoder(X_test)
    toc = time.perf_counter()

    decoder = build_conv_decoder(output_shape=X_train.shape[1:], filters=8, n_components=2, cropping=(2, 2), dropout=0.2)
    decoder.compile(optimizer='adam', loss='mse')
    hist_dec = decoder.fit(X_train_red, X_train, epochs=100, validation_split=0.1, shuffle=False, batch_size=64, verbose=0)
    X_train_rec = decoder(X_train_red).numpy()
    X_test_rec = decoder(X_test_red).numpy()

    time_in_sample = tac - tic
    time_out_of_sample = toc - tac
    
    plot_original(
        X_train, y_train,
        path.join(root, title), 'train_orig',
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_original(
        X_test, y_test,
        path.join(root, title), 'test_orig',
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_projection(X_train_red, y_train, path.join(root, title, experiment), 'train_red',)
    plot_original(
        X_train_rec, y_train,
        path.join(root, title, experiment), 'train_rec',
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_projection(X_test_red, y_test, path.join(root, title, experiment), 'test_red',)
    plot_original(
        X_test_rec, y_test,
        path.join(root, title, experiment), 'test_rec',
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_interpolations(
        X_train_red,
        y_train,
        decoder,
        path.join(root, title, experiment),
        'train_interp',
        class_pairs = [(i, i+1) for i in range(0, 5)],
        n_interpolations=6
    )
    plot_interpolations(
        X_test_red,
        y_test,
        decoder,
        path.join(root, title, experiment),
        'test_interp',
        class_pairs = [(i, i+1) for i in range(0, 5)],
        n_interpolations=6
    )
    plot_history(hist_enc, path.join(root, 'histories', title, experiment, 'encoder'), log_scale=True)
    plot_history(hist_dec, path.join(root, 'histories', title, experiment, 'decoder'), log_scale=True)

    compute_metrics(
        X_train,
        y_train,
        X_train_red,
        X_train_rec,
        X_test,
        y_test,
        X_test_red,
        X_test_rec,
        time_in_sample,
        time_out_of_sample,
        title,
        output_dir=path.join(root, title, experiment)
    )