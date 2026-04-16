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
    
    # Detect number of classes
    num_classes = len(np.unique(y_true))
    average_mode = "binary" if num_classes == 2 else "weighted"
    
    metrics["precision"] = precision_score(y_true, y_pred, average=average_mode, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average_mode, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average_mode, zero_division=0)

    # ROC-AUC only for binary classification
    if y_prob is not None and num_classes == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics