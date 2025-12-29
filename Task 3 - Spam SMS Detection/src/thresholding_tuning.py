import numpy as np
from sklearn.metrics import f1_score

def tune_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        scores.append((t, f1_score(y_true, preds)))

    return max(scores, key=lambda x: x[1])
