import os
import numpy as np
import pandas as pd


def compute_metrics(
    X_orig,
    X_rec,
    title,
    output_dir
):
    results = {'title': [], 'rec_error': []}
    rec_error = np.linalg.norm((X_orig - X_rec).numpy().flatten())
    results['title'].append(title)
    results['rec_error'].append(rec_error)
    
    # Save the results to a .txt file
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'metrics-' + title + '.txt'), sep='\t', index=False)
