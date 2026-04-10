# KL divergence
from scipy.stats import entropy

def kl_divergence(p, q):
    p = np.asarray(p) + 1e-8
    q = np.asarray(q) + 1e-8
    return entropy(p, q)

# Wasserstein distance
from scipy.stats import wasserstein_distance

def wasserstein_dist(x, y):
    return wasserstein_distance(x, y)