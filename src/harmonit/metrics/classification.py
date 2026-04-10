import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="binary")
    metrics["recall"] = recall_score(y_true, y_pred, average="binary")
    metrics["f1"] = f1_score(y_true, y_pred, average="binary")

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics