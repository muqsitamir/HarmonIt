import numpy as np
from harmonit.metrics.feature_consistency import feature_similarity

# Case 1: identical features
f1 = np.random.rand(128)
f2 = f1.copy()

sim_same = feature_similarity(f1, f2)
print("Similarity (same):", sim_same)

# Case 2: different features
f3 = np.random.rand(128)
sim_diff = feature_similarity(f1, f3)
print("Similarity (different):", sim_diff)

f4 = -f1
print("Similarity (opposite):", feature_similarity(f1, f4))