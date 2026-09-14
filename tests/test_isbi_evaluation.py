import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from eval_isbi2027 import validate_cohort
from harmonit.metrics.subject_evaluation import (
    balanced_accuracy_draws, pixel_metrics, probability_histogram,
    stratified_indices, summarize_subjects,
)


class SubjectEvaluationTests(unittest.TestCase):
    def test_fixed_scale_constant_output(self):
        result = pixel_metrics(np.zeros(10), np.full(10, .5))
        self.assertAlmostEqual(result["psnr"], 6.020599913)
        self.assertTrue(np.isnan(result["cross_correlation"]))

    def test_histogram_shift_is_detected(self):
        raw = np.linspace(0, .2, 1000)
        result = pixel_metrics(raw, raw + .7)
        self.assertGreater(result["subject_kl_raw_harm"], 10)
        self.assertAlmostEqual(result["subject_wasserstein_raw_harm"], .7)

    def test_tails_and_endpoint(self):
        hist = probability_histogram([-.1, 0, .5, 1, 1.1])
        self.assertAlmostEqual(hist.sum(), 1)
        self.assertAlmostEqual(hist[0], .2, places=5)
        self.assertAlmostEqual(hist[-1], .2, places=5)
        self.assertAlmostEqual(hist[-2], .2, places=5)

    def test_identity_not_silently_excluded(self):
        indices = stratified_indices([0, 0], 20)
        result = summarize_subjects([10., np.inf], indices)
        self.assertIsNone(result["estimate"])
        self.assertIsNone(result["ci95"])
        self.assertEqual(result["n_posinf"], 1)
        self.assertEqual(result["finite_only_mean"], 10.)

    def test_bootstrap_keeps_site_counts_and_pairs(self):
        sites = np.array([0, 0, 1, 1, 1])
        a = stratified_indices(sites, 100)
        b = stratified_indices(sites, 100)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal((sites[a] == 0).sum(axis=1), np.full(100, 2))
        delta = np.arange(5.) - (np.arange(5.) + 2)
        np.testing.assert_allclose(delta[a].mean(axis=1), -2)

    def test_source_ba_uses_present_classes_with_target_predictions(self):
        sites = np.array([0, 0, 1, 1])
        indices = stratified_indices(sites, 20)
        point, draws = balanced_accuracy_draws(sites, [0, 0, 5, 5], indices)
        self.assertEqual(point, .5)
        np.testing.assert_allclose(draws, .5)

    def test_split_overlap_rejected(self):
        manifest = pd.DataFrame({"subject_id": ["a", "b", "c"]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_cohort(manifest, {"train": ["a"], "val": ["a", "b"], "test": ["c"]}, "test")

    def test_subject_shuffle_not_class_renaming(self):
        from harmonit.data.label_controls import shuffled_subject_labels
        labels = np.repeat(np.arange(4), 30)
        shuffled = shuffled_subject_labels(labels)
        np.testing.assert_array_equal(np.bincount(labels), np.bincount(shuffled))
        self.assertGreater(len(set(shuffled[labels == 0])), 1)
        np.testing.assert_array_equal(shuffled, shuffled_subject_labels(labels))


if __name__ == "__main__":
    unittest.main()
