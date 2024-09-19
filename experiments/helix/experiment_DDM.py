import numpy as np
from os import path

from DiffusionLoss import DiffusionLoss
from aux_functions import get_sigma
from experiments.helix.plot_results import plot_original, plot_projection, plot_history
from experiments.helix.load_data import get_datasets
from experiments.utils import build_encoder, build_decoder
from experiments.helix.metrics import compute_metrics


root = 'experiments/helix/results/DDM'

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
datasets_train, datasets_test = get_datasets(npoints=2000, test_size=0.5, seed=123, noise=0.5)
for (X, y), title in zip(datasets_train, titles):
    plot_original(X, y, title, path.join(root, 'train_orig'))

for (X, y), title in zip(datasets_test, titles):
    plot_original(X, y, title, path.join(root, 'test_orig'))

for i in range(len(titles)):
    q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
    X_train, y_train = datasets_train[i]
    X_test, y_test = datasets_test[i]
    title = titles[i]
    sigma = get_sigma(X_train, q)

    encoder = build_encoder(input_shape=(X_train.shape[-1],), units=128, n_components=2)
    loss = DiffusionLoss(X_train, sigma=sigma, steps=steps, kernel=kernel, alpha=alpha)
    encoder.compile(optimizer='adam', loss=loss)
    indices = np.array(list(range(len(X_train))))
    hist_enc = encoder.fit(X_train, indices, epochs=50, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)
    X_train_red = encoder(X_train)
    X_test_red = encoder(X_test)

    decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
    decoder.compile(optimizer='adam', loss='mse')
    hist_dec = decoder.fit(X_train_red, X_train, epochs=50, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)
    X_train_rec = decoder(X_train_red)
    X_test_rec = decoder(X_test_red)

    plot_projection(X_train_red, y_train, title, path.join(root, 'train_red'))
    plot_original(X_train_rec, y_train, title, path.join(root, 'train_rec'))
    plot_projection(X_test_red, y_test, title, path.join(root, 'test_red'))
    plot_original(X_test_rec, y_test, title, path.join(root, 'test_rec'))
    plot_history(hist_enc, path.join(root, 'histories', 'encoder', title), log_scale=True)
    plot_history(hist_dec, path.join(root, 'histories', 'decoder', title), log_scale=True)

    compute_metrics(X_test, X_test_rec, title, root)        
