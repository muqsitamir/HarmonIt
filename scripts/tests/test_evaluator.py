import numpy as np
from harmonit.metrics.evaluator import evaluate_all_metrics

# Fake classification data
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0])
y_prob = np.array([0.1, 0.9, 0.4, 0.2])

# Fake site labels
y_true_site = [0, 1, 0, 1]
y_pred_site = [0, 1, 1, 1]

# Fake images
img1 = np.random.rand(64, 64)
img2 = img1 + np.random.normal(0, 0.1, img1.shape)

# Fake features
feat1 = np.random.rand(128)
feat2 = feat1.copy()

# Fake distributions
dist1 = np.random.normal(0, 1, 1000)
dist2 = np.random.normal(0, 1, 1000)

results = evaluate_all_metrics(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    y_true_site=y_true_site,
    y_pred_site=y_pred_site,
    img1=img1,
    img2=img2,
    feat1=feat1,
    feat2=feat2,
    dist1=dist1,
    dist2=dist2,
)

for k, v in results.items():
    print(k, ":", v)