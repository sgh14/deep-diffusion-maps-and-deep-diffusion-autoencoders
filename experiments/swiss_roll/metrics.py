from os import path
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
import h5py


def compute_metrics(
    X_train,
    X_train_red,
    X_train_rec,
    X_test,
    X_test_red,
    X_test_rec,
    time_in_sample,
    time_out_of_sample,
    title,
    output_dir
):
    p = 0.01
    metric = 'euclidean'
    rec_error_in_sample = np.mean(np.linalg.norm(X_train - X_train_rec, axis=1))
    rec_error_out_of_sample = np.mean(np.linalg.norm(X_test - X_test_rec, axis=1))
    trustworthiness_in_sample = trustworthiness(X_train, X_train_red, n_neighbors=round(p*len(X_train)), metric=metric)
    trustworthiness_out_of_sample = trustworthiness(X_test, X_test_red, n_neighbors=round(p*len(X_test)), metric=metric)
    continuity_in_sample = trustworthiness(X_train_red, X_train, n_neighbors=round(p*len(X_train)), metric=metric)
    continuity_out_of_sample = trustworthiness(X_test_red, X_test, n_neighbors=round(p*len(X_test)), metric=metric)

    results = {
        'title': [title],
        'rec_error_in_sample': [rec_error_in_sample],
        'rec_error_out_of_sample': [rec_error_out_of_sample],
        'trustworthiness_in_sample': [trustworthiness_in_sample],
        'trustworthiness_out_of_sample': [trustworthiness_out_of_sample],
        'continuity_in_sample': [continuity_in_sample],
        'continuity_out_of_sample': [continuity_out_of_sample],
        'time_in_sample':[time_in_sample],
        'time_out_of_sample':[time_out_of_sample]
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
        file.create_dataset("X_test", data=X_train, compression='gzip')
        file.create_dataset("X_test_red", data=X_test_red, compression='gzip')
        file.create_dataset("X_test_rec", data=X_test_rec, compression='gzip')