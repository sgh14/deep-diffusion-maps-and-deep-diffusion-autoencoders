from matplotlib import pyplot as plt
import numpy as np
from os.path import join


def plot_history(history, results_dir, log_scale=False):
    h = history.history
    keys = [key for key in h.keys() if not key.startswith('val_')]
    for key in keys:
        y = np.array([h[key], h['val_' + key]])
        fig, ax = plt.subplots()
        if log_scale:
            ax.semilogy(y[0], label='Training')
            ax.semilogy(y[1], label='Validation')
        else:
            ax.plot(y[0], label='Training')
            ax.plot(y[1], label='Validation')

        ax.set_ylabel(key)
        ax.set_xlabel('Epoch')
        ax.legend()
        fig.savefig(join(results_dir, key + '.png'))


def plot_results(X_red, y, results_dir, cmap=plt.get_cmap('Set1')):
    n_components = X_red.shape[1]
    for i in range(n_components-1):
        fig, ax = plt.subplots()
        ax.scatter(X_red[:, i], X_red[:, i+1], c=cmap(y))

        ax.set_xlabel(f'Feature {i+1}')
        ax.set_ylabel(f'Feature {i+2}')
        fig.savefig(join(results_dir, f'X_red_{i+1}-{i+2}.png'))
