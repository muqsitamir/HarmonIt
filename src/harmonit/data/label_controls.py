"""Deterministic subject-level negative controls."""
import numpy as np


def shuffled_subject_labels(labels, seed=12345):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("Expected one label per subject")
    return np.random.RandomState(seed).permutation(labels)
