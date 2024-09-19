import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

'''
Purity measures the frequency of data belonging to the same cluster sharing the
same class label, while Accuracy measures the frequency of data from the same
class appearing in a single cluster.
'''

def purity_score(y_true, y_pred):
    # Compute confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    # Find the maximum values in each row (each class)
    max_in_rows = np.amax(matrix, axis=1)

    # Sum the maximum values found
    purity = np.sum(max_in_rows) / np.sum(matrix)

    return purity


def clustering_accuracy(y_true, y_pred):
    # Compute the confusion matrix
    matrix = confusion_matrix(y_true, y_pred)

    # Use the linear_sum_assignment method to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-matrix)

    # Calculate the accuracy using the optimal assignment
    accuracy = matrix[row_ind, col_ind].sum() / np.sum(matrix)

    return accuracy


def compute_metrics(
    X_orig,
    X_red,
    X_rec,
    y,
    title,
    output_dir
):
    results = {'title': [], 'purity': [], 'accuracy': [], 'rec_error': []}
    n_classes = len(np.unique(y))
    # Initialize K-Means with n_classes clusters
    k_means = KMeans(n_clusters=n_classes)
    # Fit K-Means to the reduced data
    clusters = k_means.fit_predict(X_red)
    # Calculate the purity of the resulting clusters
    clusters_purity = purity_score(y, clusters)
    # Calculate the accuracy of the resulting clusters
    clusters_accuracy = clustering_accuracy(y, clusters)
    # Reconstruction error
    rec_error = np.linalg.norm(X_orig.flatten() - X_rec.flatten())

    results['title'].append(title)
    results['purity'].append(clusters_purity)
    results['accuracy'].append(clusters_accuracy)
    results['rec_error'].append(rec_error)
    
    # Save the results to a .txt file
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'metrics-' + title + '.txt'), sep='\t', index=False)
