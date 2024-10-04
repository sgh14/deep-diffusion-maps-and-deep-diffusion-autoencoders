import time
import numpy as np
import tensorflow as tf
from os import path

from DeepDiffusionAE import DeepDiffusionAE
from aux_functions import get_sigma
from experiments.phoneme.plot_results import plot_original, plot_projection, plot_history
from experiments.phoneme.load_data import get_datasets
from experiments.phoneme.metrics import compute_metrics
from experiments.utils import build_seq_encoder, build_seq_decoder


seed = 123
tf.random.set_seed(seed)
root = 'experiments/phoneme/results/DDAE'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]

datasets_train, datasets_test = get_datasets(test_size=0.2, seed=seed, noise=0.25)

diffusion_weights = np.arange(0.0, 1.01, 0.05)
q_vals = [0.75, 0.75, 0.75, 0.75]
steps_vals = [1, 1, 1, 1]
alpha_vals = [0, 0, 0, 0]
kernel = 'rbf'

for diffusion_weight in diffusion_weights:
    for i in range(len(titles)):
        q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
        experiment = f'percentile_{q}-steps_{steps}-alpha_{alpha}'
        experiment = path.join(experiment, f'diffusion_weight_{diffusion_weight:.2f}')
        X_train, y_train = datasets_train[i]
        X_test, y_test = datasets_test[i]
        title = titles[i]
        sigma = get_sigma(X_train.reshape((X_train.shape[0], -1)), q)
        
        print(experiment, '-', title)  
        encoder = build_seq_encoder(input_shape=X_train.shape[1:], filters=8, n_components=2, zero_padding=0)
        decoder = build_seq_decoder(output_shape=X_train.shape[1:], filters=8, n_components=2, cropping=0)
        autoencoder = DeepDiffusionAE(encoder, decoder)
        tic = time.perf_counter()
        autoencoder.compile(
            X_train.reshape((X_train.shape[0], -1)),
            sigma=sigma,
            steps=steps,
            kernel=kernel,
            alpha=alpha,
            diffusion_weight=diffusion_weight,
            beta=0.9,
            optimizer='adam'
        )
        history = autoencoder.fit(X_train, epochs=50, validation_split=0.1, shuffle=False, batch_size=64, verbose=0)

        X_train_red = autoencoder.encode(X_train)
        tac = time.perf_counter()
        X_test_red = autoencoder.encode(X_test)
        toc = time.perf_counter()
        X_train_rec = autoencoder.decode(X_train_red).numpy()
        X_test_rec = autoencoder.decode(X_test_red).numpy()

        time_in_sample = tac - tic
        time_out_of_sample = toc - tac
        
        plot_original(X_train, y_train, path.join(root, title), 'train_orig')
        plot_original(X_test, y_test, path.join(root, title), 'test_orig')
        plot_projection(X_train_red, y_train, path.join(root, title, experiment), 'train_red',)
        plot_original(X_train_rec, y_train, path.join(root, title, experiment), 'train_rec')
        plot_projection(X_test_red, y_test, path.join(root, title, experiment), 'test_red',)
        plot_original(X_test_rec, y_test, path.join(root, title, experiment), 'test_rec')
        
        plot_history(history, path.join(root, 'histories', title, experiment), log_scale=True)

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