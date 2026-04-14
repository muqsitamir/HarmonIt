import numpy as np
from harmonit.metrics.classification import compute_classification_metrics

y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0])
y_prob = np.array([0.1, 0.9, 0.4, 0.2])

metrics = compute_classification_metrics(y_true, y_pred, y_prob)

for k, v in metrics.items():
    print(k, ":", v)