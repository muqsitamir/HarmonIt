from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def feature_similarity(feat1, feat2):
    return cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0][0]

def cross_correlation(feat1, feat2):
    """
    Computes normalized cross-correlation between two feature vectors.
    """

    feat1 = np.asarray(feat1).flatten()
    feat2 = np.asarray(feat2).flatten()

    if feat1.shape != feat2.shape:
        raise ValueError("Features must have same shape")

    # Normalize
    feat1 = (feat1 - feat1.mean()) / (feat1.std() + 1e-8)
    feat2 = (feat2 - feat2.mean()) / (feat2.std() + 1e-8)

    return float(np.mean(feat1 * feat2))