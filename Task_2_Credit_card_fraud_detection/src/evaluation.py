from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }
    return metrics


def print_classification(y_true, y_pred):
    return classification_report(
        y_true,
        y_pred,
        target_names=['Legitimate', 'Fraud']
    )


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
