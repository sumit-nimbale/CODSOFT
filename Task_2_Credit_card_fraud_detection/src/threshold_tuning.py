import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def tune_thresholds(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.9, 0.05)

    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        results.append([
            t,
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred, zero_division=0)
        ])

    return pd.DataFrame(
        results,
        columns=['Threshold', 'Precision', 'Recall', 'F1']
    )
