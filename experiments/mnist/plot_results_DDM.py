import numpy as np
import h5py
from os import path
from experiments.mnist.plot_results import plot_original, plot_projection, plot_history, plot_interpolations
from tensorflow import keras
from experiments.utils import ConvBlock2D, UpConvBlock2D


root = 'experiments/mnist/results/DDM'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
results_file = 'results.h5'
history_file = 'history.h5'
decoder_file = 'decoder.h5'
img_shape = (28, 28, 1)

q_vals = [0.0075, 0.0075, 0.0075, 0.0075]
steps_vals = [1, 1, 1, 1]
alpha_vals = [0, 0, 0, 0]

for i in range(len(titles)):
    title = titles[i]
    q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
    experiment = f'quantile_{q}-steps_{steps}-alpha_{alpha}'
    output_dir = path.join(root, title, experiment)
    for hist_name in ('hist_enc', 'hist_dec'):
        history = {}
        with h5py.File(path.join(output_dir, hist_name + '.h5'), 'r') as file:
            for key in file.keys():
                history[key] = np.array(file[key])

        plot_history(history, output_dir, hist_name, log_scale=True)

    decoder = keras.models.load_model(
        path.join(output_dir, decoder_file),
        custom_objects={'ConvBlock2D': ConvBlock2D, 'UpConvBlock2D': UpConvBlock2D},
        compile=False
    )
    for subset in ('train', 'test'):
        with h5py.File(path.join(output_dir, results_file), 'r') as file:
            X_orig = np.array(file['/X_' + subset]).reshape(-1, *img_shape)
            X_red = np.array(file['/X_' + subset + '_red'])
            X_rec = np.array(file['/X_' + subset + '_rec']).reshape(-1, *img_shape)
            y = np.array(file['/y_' + subset])
        
        plot_original(X_orig, y, output_dir, subset + '_orig', images_per_class=2, grid_shape=(3, 4))
        plot_projection(X_red, y, output_dir, subset + '_red')
        plot_original(X_rec, y, output_dir, subset + '_rec', images_per_class=2, grid_shape=(3, 4))
        plot_interpolations(X_red, y, decoder, output_dir, subset + '_interp', class_pairs = [(i, i+1) for i in range(0, 5)], n_interpolations=6)
