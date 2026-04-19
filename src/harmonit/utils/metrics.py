import numpy as np

def confusion_and_balanced_acc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    # per-class recall; avoid divide-by-zero
    recalls = []
    for c in range(num_classes):
        denom = cm[c, :].sum()
        if denom > 0:
            recalls.append(cm[c, c] / denom)
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return cm, acc, bal_acc