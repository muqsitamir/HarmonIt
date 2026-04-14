# KL divergence
from scipy.stats import entropy

import numpy as np
from scipy.stats import entropy

def kl_divergence(p, q, bins=50):
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    p_hist += 1e-8
    q_hist += 1e-8

    return entropy(p_hist, q_hist)

# Wasserstein distance
from scipy.stats import wasserstein_distance

def wasserstein_dist(x, y):
    return wasserstein_distance(x, y)