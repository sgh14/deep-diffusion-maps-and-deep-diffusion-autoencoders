from os import path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import trustworthiness
import h5py


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
    output_dir
):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_train_rec = X_train_rec.reshape(X_train_rec.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test_rec = X_test_rec.reshape(X_test_rec.shape[0], -1)
    
    p = 0.01
    metric = 'euclidean'
    rec_error_in_sample = np.mean(np.linalg.norm(X_train - X_train_rec, axis=1))
    rec_error_out_of_sample = np.mean(np.linalg.norm(X_test - X_test_rec, axis=1))
    trustworthiness_in_sample = trustworthiness(X_train, X_train_red, n_neighbors=round(p*len(X_train)), metric=metric)
    trustworthiness_out_of_sample = trustworthiness(X_test, X_test_red, n_neighbors=round(p*len(X_test)), metric=metric)
    continuity_in_sample = trustworthiness(X_train_red, X_train, n_neighbors=round(p*len(X_train)), metric=metric)
    continuity_out_of_sample = trustworthiness(X_test_red, X_test, n_neighbors=round(p*len(X_test)), metric=metric)

    n_classes = len(np.unique(y_train))
    k_means = KMeans(n_clusters=n_classes)
    clusters = k_means.fit_predict(X_train_red)
    purity_in_sample = purity_score(y_train, clusters)
    accuracy_in_sample = clustering_accuracy(y_train, clusters)

    n_classes = len(np.unique(y_test))
    k_means = KMeans(n_clusters=n_classes)
    clusters = k_means.fit_predict(X_test_red)
    purity_out_of_sample = purity_score(y_test, clusters)
    accuracy_out_of_sample = clustering_accuracy(y_test, clusters)

    results = {
        'title': [title],
        'rec_error_in_sample': [rec_error_in_sample],
        'rec_error_out_of_sample': [rec_error_out_of_sample],
        'trustworthiness_in_sample': [trustworthiness_in_sample],
        'trustworthiness_out_of_sample': [trustworthiness_out_of_sample],
        'continuity_in_sample': [continuity_in_sample],
        'continuity_out_of_sample': [continuity_out_of_sample],
        'time_in_sample':[time_in_sample],
        'time_out_of_sample':[time_out_of_sample],
        'purity_in_sample': [purity_in_sample],
        'purity_out_of_sample': [purity_out_of_sample],
        'accuracy_in_sample': [accuracy_in_sample],
        'accuracy_out_of_sample': [accuracy_out_of_sample],
    }
    
    df = pd.DataFrame(results)
    df.to_csv(path.join(output_dir, 'metrics.txt'), sep='\t', index=False)

    # for (X, X_red, name) in ((X_train, X_train_red, 'train'), (X_test, X_test_red, 'test')):
    #     curves = {'k': [], 'T': [], 'C': []}
    #     k_vals = [1] + list(range(10, round(0.5*len(X)), 10))
    #     for k in k_vals:
    #         curves['k'].append(k)
    #         curves['T'].append(trustworthiness(X, X_red, n_neighbors=k, metric=metric))
    #         curves['C'].append(trustworthiness(X_red, X, n_neighbors=k, metric=metric))
            
    #     df = pd.DataFrame(curves)
    #     df.to_csv(path.join(output_dir, 'metrics-curves-' + name + '.txt'), sep='\t', index=False)

    with h5py.File(path.join(output_dir, 'results.h5'), "w") as file:
        file.create_dataset("X_train", data=X_train, compression='gzip')
        file.create_dataset("X_train_red", data=X_train_red, compression='gzip')
        file.create_dataset("X_train_rec", data=X_train_rec, compression='gzip')
        file.create_dataset("y_train", data=y_train, compression='gzip')
        file.create_dataset("X_test", data=X_test, compression='gzip')
        file.create_dataset("X_test_red", data=X_test_red, compression='gzip')
        file.create_dataset("X_test_rec", data=X_test_rec, compression='gzip')
        file.create_dataset("y_test", data=y_test, compression='gzip')

