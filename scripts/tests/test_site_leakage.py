from harmonit.metrics.site_leakage import site_classification_accuracy

y_true = [0, 1, 0, 1]
y_pred = [0, 1, 1, 1]

acc = site_classification_accuracy(y_true, y_pred)
print("Site accuracy:", acc)