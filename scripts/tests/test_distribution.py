import numpy as np
from harmonit.metrics.distribution_alignment import kl_divergence, wasserstein_dist

# Same distribution
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)

print("KL (same):", kl_divergence(x, y))
print("Wasserstein (same):", wasserstein_dist(x, y))

# Different distribution
z = np.random.normal(3, 1, 1000)

print("KL (different):", kl_divergence(x, z))
print("Wasserstein (different):", wasserstein_dist(x, z))