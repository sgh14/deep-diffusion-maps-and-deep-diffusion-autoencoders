from os import path

from DeepDiffusionAE import DeepDiffusionAE
from aux_functions import get_sigma
from experiments.helix.plot_results import plot_original, plot_projection, plot_history
from experiments.helix.load_data import get_datasets
from experiments.utils import build_encoder, build_decoder
from experiments.helix.metrics import compute_metrics


root = 'experiments/helix/results/DDAE'

diffusion_weights = [0.1, 0.5, 0.9]
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

for diffusion_weight in diffusion_weights:
    experiment = f'diffusion_weight_{diffusion_weight}'
    for i in range(len(titles)):
        q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
        X_train, y_train = datasets_train[i]
        X_test, y_test = datasets_test[i]
        title = titles[i]
        sigma = get_sigma(X_train, q)

        encoder = build_encoder(input_shape=(X_train.shape[-1],), units=128, n_components=2)
        decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
        autoencoder = DeepDiffusionAE(encoder, decoder)
        autoencoder.compile(X_train, sigma=sigma, steps=steps, kernel=kernel, alpha=alpha, diffusion_weight=diffusion_weight, optimizer='adam')
        history = autoencoder.fit(X_train, epochs=300, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)

        X_train_red = autoencoder.encode(X_train)
        X_test_red = autoencoder.encode(X_test)
        X_train_rec = autoencoder.decode(X_train_red)
        X_test_rec = autoencoder.decode(X_test_red)

        plot_projection(X_train_red, y_train, title, path.join(root, experiment, 'train_red'))
        plot_original(X_train_rec, y_train, title, path.join(root, experiment, 'train_rec'))
        plot_projection(X_test_red, y_test, title, path.join(root, experiment, 'test_red'))
        plot_original(X_test_rec, y_test, title, path.join(root, experiment, 'test_rec'))
        plot_history(history, path.join(root, experiment, 'histories', title), log_scale=True)

        compute_metrics(X_test, X_test_rec, title, path.join(root, experiment))        
