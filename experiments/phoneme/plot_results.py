import os
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('experiments/science.mplstyle')
colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
classes = {0: 'aa', 1: 'ao', 2: 'dcl', 3: 'iy', 4: 'sh'}

def plot_original(X, y, output_dir, filename, n_samples=30):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
    for i in range(n_samples):
        ax.plot(X[i], color=colors[y[i]], linewidth=1, alpha=0.75)
    
    ax.set_box_aspect(1)
    # Create a list of handles and labels for the legend
    unique_y = np.unique(y)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[val], markersize=10) for val in unique_y]
    labels = [classes[val] for val in unique_y]  # Adjust labels based on your case

    # Add the legend below the plot, with ncol=number of unique y values for one-row legend
    fig.legend(handles, labels, loc='lower center', ncol=len(unique_y), columnspacing=0.5, bbox_to_anchor=(0.5, -0.1))


    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, filename + format))
    
    plt.close(fig)


def plot_projection(X, y, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
    ax.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in y])#y, cmap=cmap)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2), useMathText=True)
    # Set box aspect ratio instead of axis aspect to ensure square subplots
    ax.set_xlabel(r'$\Tilde{x}$')
    ax.set_ylabel(r'$\Tilde{y}$')
    ax.set_box_aspect(1)

    # Create a list of handles and labels for the legend
    unique_y = np.unique(y)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[val], markersize=10) for val in unique_y]
    labels = [classes[val] for val in unique_y]  # Adjust labels based on your case

    # Add the legend below the plot, with ncol=number of unique y values for one-row legend
    fig.legend(handles, labels, loc='lower center', ncol=len(unique_y), columnspacing=0.5, bbox_to_anchor=(0.5, -0.1))

    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, filename + format))
    
    plt.close(fig)


def plot_history(history, output_dir, log_scale=False):
    os.makedirs(output_dir, exist_ok=True)

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

        ax.set_ylabel(key.capitalize())
        ax.set_xlabel('Epoch')
        ax.legend()
        for format in ('.pdf', '.png', '.svg'):
            fig.savefig(os.path.join(output_dir, key + format))
        
        plt.close(fig)