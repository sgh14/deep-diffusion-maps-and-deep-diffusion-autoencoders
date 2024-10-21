import numpy as np
import h5py
from os import path
from experiments.helix.plot_results import plot_original, plot_projection, plot_history


root = 'experiments/helix/results/DDM'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
results_file = 'results.h5'

q_vals = [2e-2, 2e-2, 5e-3, 5e-3]
steps_vals = [100, 100, 100, 100]
alpha_vals = [1, 1, 1, 1]

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

    for subset in ('train', 'test'):
        with h5py.File(path.join(output_dir, results_file), 'r') as file:
            X_orig = np.array(file['/X_' + subset])
            X_red = np.array(file['/X_' + subset + '_red'])
            X_rec = np.array(file['/X_' + subset + '_rec'])
            y = np.array(file['/y_' + subset])
        
        plot_original(X_orig, y, output_dir, subset + '_orig')
        plot_projection(X_red, y, output_dir, subset + '_red')
        plot_original(X_rec, y, output_dir, subset + '_rec')