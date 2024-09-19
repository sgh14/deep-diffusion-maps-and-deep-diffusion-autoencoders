import os
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('experiments/science.mplstyle')
colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']


# Function to sample 2 images per class from the dataset
def sample_images_per_class(X, y, images_per_class=2):
    selected_images = []
    selected_labels = []
    
    n_classes = len(np.unique(y))
    for class_label in range(n_classes):
        class_indices = np.where(y == class_label)[0]
        selected_indices = class_indices[:images_per_class]
        selected_images.extend(X[selected_indices])
        selected_labels.extend(y[selected_indices])
    
    return np.array(selected_images), np.array(selected_labels)


def plot_images(axes, X, y=[]):
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X[i], cmap='gray')
        if len(y) > 0:
            ax.set_title(y[i])

        ax.axis('off')
    
    return axes


def plot_original(
    X,
    y,
    title,
    output_dir,
    images_per_class=2,
    grid_shape=(3, 4)
):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        grid_shape[0], grid_shape[1],
        figsize=(3, 3),
        gridspec_kw={'wspace': 0.2, 'hspace': 0.2}
    )
    X, y = sample_images_per_class(X, y, images_per_class)
    axes = plot_images(axes, X, y)
    fig.tight_layout()
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, title + format))
    
    plt.close(fig)


def plot_projection(X, y, title, output_dir):
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
    labels = [str(val) for val in unique_y]  # Adjust labels based on your case

    # Add the legend below the plot, with ncol=number of unique y values for one-row legend
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1) # loc='lower center', ncol=len(unique_y)//2, bbox_to_anchor=(0.5, -0.1))

    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, title + format))
    
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


# Helper function to compute class centroids in latent space
def compute_centroids(X_red, y):
    n_classes = len(np.unique(y))
    centroids = [np.mean(X_red[y == i], axis=0) for i in range(n_classes)]

    return np.array(centroids)

# Helper function to interpolate between two centroids
def interpolate(x1, x2, n_interpolations=4):
    alphas = np.linspace(0, 1, n_interpolations)
    interpolations = [(1 - alpha) * x1 + alpha * x2 for alpha in alphas]
    
    return np.array(interpolations)


def interpolate_images(X_red, y, decoder, class_pairs, n_interpolations, image_shape):
    centroids = compute_centroids(X_red, y)
    interpolated_images = []
    for class1, class2 in class_pairs:
        centroid1 = centroids[class1]
        centroid2 = centroids[class2]
        interpolations = interpolate(centroid1, centroid2, n_interpolations)
        interpolated_images.append(
            decoder(interpolations).numpy().reshape(-1, *image_shape)
        )

    interpolated_images = np.vstack(interpolated_images)

    return interpolated_images


# Function to generate and plot interpolations between two classes
def plot_interpolations(
    X_red,
    y,
    title,
    decoder,
    output_dir,
    image_shape,
    class_pairs,
    n_interpolations=4
):
    os.makedirs(output_dir, exist_ok=True)

    grid_shape = (len(class_pairs), n_interpolations)
    fig, axes = plt.subplots(
        grid_shape[0], grid_shape[1],
        figsize=(3, 3),
        gridspec_kw={'wspace': 0.2, 'hspace': 0}
    )
    X_interp = interpolate_images(
            X_red, y, decoder, class_pairs, n_interpolations, image_shape
    )
    axes = plot_images(axes, X_interp)
    fig.tight_layout()
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, title + format))
    
    plt.close(fig)
