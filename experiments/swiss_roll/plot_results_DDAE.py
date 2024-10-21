import numpy as np
import h5py
from os import path
from experiments.swiss_roll.plot_results import plot_original, plot_projection, plot_history


root = 'experiments/swiss_roll/results/DDAE'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
results_file = 'results.h5'
history_file = 'history.h5'

diffusion_weights = np.arange(0.0, 1.01, 0.1)
q_vals = [5e-3, 5e-3, 2.5e-3, 2.5e-3]
steps_vals = [100, 100, 100, 100]
alpha_vals = [1, 1, 1, 1]

for diffusion_weight in diffusion_weights:
    for i in range(len(titles)):
        title = titles[i]
        q, steps, alpha = q_vals[i], steps_vals[i], alpha_vals[i]
        experiment = f'quantile_{q}-steps_{steps}-alpha_{alpha}'
        experiment = path.join(experiment, f'diffusion_weight_{diffusion_weight:.2f}')
        output_dir = path.join(root, title, experiment)
        history = {}
        with h5py.File(path.join(output_dir, history_file), 'r') as file:
            for key in file.keys():
                history[key] = np.array(file[key])

        plot_history(history, output_dir, 'history', log_scale=True)

        for subset in ('train', 'test'):
            with h5py.File(path.join(output_dir, results_file), 'r') as file:
                X_orig = np.array(file['/X_' + subset])
                X_red = np.array(file['/X_' + subset + '_red'])
                X_rec = np.array(file['/X_' + subset + '_rec'])
                y = np.array(file['/y_' + subset])
            
            plot_original(X_orig, y, output_dir, subset + '_orig')
            plot_projection(X_red, y, output_dir, subset + '_red')
            plot_original(X_rec, y, output_dir, subset + '_rec')