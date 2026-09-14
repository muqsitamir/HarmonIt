"""Intensity-only site probe on foreground histograms (protocol amendment 5).

Removes head geometry and texture from the probe's view: features are foreground
intensity histograms, with foreground defined on the raw slice. Compares a probe trained
on raw training slices with probes trained on each method's training outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harmonit.metrics.subject_evaluation import balanced_accuracy_draws, interval, probability_histogram

NPZ = {"histogram_matching": "histogram_matching_slices.npz", "cyclegan_tuned": "cyclegan_nyu_slices.npz",
       "diffusion_20k": "diffusion_img2img_nyu_slices.npz"}
OWN_TEST = {"histogram_matching": "histogram_matching", "cyclegan_tuned": "cyclegan_tuned",
            "diffusion_20k": "diffusion_20k_redraw"}


def features(images, raw, threshold=0.02):
    return np.stack([probability_histogram(img[0][r[0] > threshold]) for img, r in zip(images, raw)])


def load(path, key):
    with np.load(path, allow_pickle=False) as data:
        return features(data[key], data["raw_images"]), data["site_ids"].astype(np.int64), \
            data["subject_ids"].astype(str)


def fit(x_train, y_train, x_val, y_val):
    best = None
    for c in (0.01, 0.1, 1, 10):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        ba = np.mean([(pred[y_val == k] == k).mean() for k in np.unique(y_val)])
        if best is None or ba > best[0]:
            best = (float(ba), c, model)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports", required=True)
    p.add_argument("--eval-run", required=True, help="Complete run with raw_reference.npz and bootstrap indices")
    p.add_argument("--out", required=True)
    p.add_argument("--target-site-id", type=int, default=5)
    args = p.parse_args()
    exports, run = Path(args.exports), Path(args.eval_run)
    protocol = json.loads((run / "protocol.json").read_text())
    artifacts = dict(spec.split("=", 1) for spec in protocol["args"]["artifact"])
    indices = np.load(run / "bootstrap_indices_source_non_nyu.npy")
    with np.load(run / "raw_reference.npz", allow_pickle=False) as data:
        raw_test, test_sites, test_subjects = data["raw_images"], data["site_ids"], data["subject_ids"].astype(str)
    source = test_sites != args.target_site_id

    def evaluate(model, images):
        pred = model.predict(features(images, raw_test))
        estimate, draws = balanced_accuracy_draws(test_sites[source], pred[source], indices)
        return estimate, draws

    test_images = {"raw": raw_test}
    for name, path in artifacts.items():
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["subject_ids"].astype(str), test_subjects):
                raise ValueError(f"{name}: subject order differs")
            test_images[name] = data["images"]

    report = {"raw_trained": {}, "own_trained": {}}
    ref = "histogram_matching"  # any export carries the same raw slices (integrity report)
    x_tr, y_tr, _ = load(exports / ref / "train" / NPZ[ref], "raw_images")
    x_va, y_va, _ = load(exports / ref / "val" / NPZ[ref], "raw_images")
    val_ba, c, raw_model = fit(x_tr, y_tr, x_va, y_va)
    report["raw_trained"]["validation"] = {"ba": val_ba, "C": c}
    raw_draws = {}
    for name, images in test_images.items():
        estimate, draws = evaluate(raw_model, images)
        raw_draws[name] = (estimate, draws)
        report["raw_trained"][name] = interval(estimate, draws)

    for method, test_name in OWN_TEST.items():
        x_tr, y_tr, _ = load(exports / method / "train" / NPZ[method], "images")
        x_va, y_va, _ = load(exports / method / "val" / NPZ[method], "images")
        val_ba, c, model = fit(x_tr, y_tr, x_va, y_va)
        estimate, draws = evaluate(model, test_images[test_name])
        base_estimate, base_draws = raw_draws[test_name]
        report["own_trained"][method] = {
            "validation": {"ba": val_ba, "C": c}, "test_artifact": test_name,
            "source_ba": interval(estimate, draws),
            "own_minus_raw_trained": interval(estimate - base_estimate, draws - base_draws),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    fmt = lambda v: f"{v['estimate']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
    print("raw-trained validation", report["raw_trained"]["validation"])
    for name in test_images:
        print(f"raw-trained on {name:22s} {fmt(report['raw_trained'][name])}")
    for method, entry in report["own_trained"].items():
        print(f"own-trained {method:20s} val {entry['validation']['ba']:.3f}  test {fmt(entry['source_ba'])}  "
              f"own-raw {fmt(entry['own_minus_raw_trained'])}")


if __name__ == "__main__":
    main()
