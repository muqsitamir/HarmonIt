"""Five-fold cross-validated site decodability for HACA3 over all ABIDE subjects (amendment 16).

HACA3 never saw ABIDE, so its outputs for all 1,112 subjects are untouched. Folds are stratified by
site (seed 20261007); within the training folds a stratified 10% is the validation set. For raw
slices, HACA3 outputs and the preprocessing-only control (amendment 15), a whole-head slice probe
(amendment 2 recipe, best validation epoch, one seed per fold) and an intensity-only probe
(amendment 5) are trained and tested on the held-out fold; the raw-trained probes are also tested
on the HACA3 and control outputs. Source BA pools held-out predictions of all non-NYU subjects,
with 2,000 within-site bootstrap replicates (seed 20260913), paired across probes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmonit.metrics.subject_evaluation import balanced_accuracy_draws, interval, stratified_indices  # noqa: E402
from harmonit.utils.metrics import confusion_and_balanced_acc  # noqa: E402
from isbi2027_histogram_probe import features, fit  # noqa: E402
from train_slice_probe import NUM_CLASSES, augment, predict  # noqa: E402

TARGET_SITE = 5
SOURCES = ("raw", "haca3", "preproc")


def load_cohort(haca3_export, preproc_export):
    """All subjects: raw slices, HACA3 outputs and control outputs, aligned by subject."""
    parts = {key: [] for key in ("raw", "haca3", "preproc", "sites", "subjects")}
    for split in ("train", "val", "test"):
        with np.load(Path(haca3_export) / split / "haca3_slices.npz", allow_pickle=False) as h, \
                np.load(Path(preproc_export) / split / "haca3_preproc_slices.npz", allow_pickle=False) as c:
            if not np.array_equal(h["subject_ids"], c["subject_ids"]) or \
                    np.abs(h["raw_images"] - c["raw_images"]).max() > 1e-6:
                raise ValueError(f"{split}: HACA3 and control exports differ in subjects or raw slices")
            parts["raw"].append(h["raw_images"]), parts["haca3"].append(h["images"])
            parts["preproc"].append(c["images"]), parts["sites"].append(h["site_ids"])
            parts["subjects"].append(h["subject_ids"].astype(str))
    cohort = {k: np.concatenate(v) for k, v in parts.items()}
    for source in SOURCES:
        cohort[source] = cohort[source].astype(np.float32)
    return cohort


def train_probe(x_train, y_train, x_val, y_val, seed, device, epochs=10, steps=50, batch=64, lr=3e-4):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best, state = -1.0, None
    for _ in range(epochs):
        model.train()
        for _ in range(steps):
            idx = rng.randint(0, len(y_train), batch)
            x = augment(torch.from_numpy(x_train[idx]).to(device), rng)
            optim.zero_grad(set_to_none=True)
            criterion(model(x), torch.from_numpy(y_train[idx]).to(device)).backward()
            optim.step()
        _, _, bal = confusion_and_balanced_acc(y_val, predict(model, x_val, device), NUM_CLASSES)
        if bal > best:
            best, state = bal, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--haca3-export", required=True, help="Directory with <split>/haca3_slices.npz")
    p.add_argument("--preproc-export", required=True, help="Directory with <split>/haca3_preproc_slices.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    data = load_cohort(args.haca3_export, args.preproc_export)
    sites = data["sites"].astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hist = {s: features(data[s], data["raw"]) for s in SOURCES}
    # Held-out predictions: (probe kind, trained on, tested on) -> per-subject predictions.
    preds = {}
    tests = {"raw": SOURCES, "haca3": ("haca3",), "preproc": ("preproc",)}
    folds = StratifiedKFold(args.folds, shuffle=True, random_state=20261007)
    log = []
    for fold, (train_idx, test_idx) in enumerate(folds.split(np.zeros(len(sites)), sites)):
        fit_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=sites[train_idx], random_state=20261007)
        for source in SOURCES:
            model, val_ba = train_probe(data[source][fit_idx], sites[fit_idx], data[source][val_idx], sites[val_idx],
                                        seed=fold + 1, device=device)
            val_h, c, logistic = fit(hist[source][fit_idx], sites[fit_idx], hist[source][val_idx], sites[val_idx])
            log.append(dict(fold=fold, source=source, image_val_ba=round(val_ba, 4), hist_val_ba=round(val_h, 4), C=c))
            print(log[-1], flush=True)
            for target in tests[source]:
                for kind, out in (("image", predict(model, data[target][test_idx], device)),
                                  ("histogram", logistic.predict(hist[target][test_idx]))):
                    preds.setdefault((kind, source, target), np.full(len(sites), -1))[test_idx] = out
    source_mask = sites != TARGET_SITE
    labels = sites[source_mask]
    indices = stratified_indices(labels)
    draws = {key: balanced_accuracy_draws(labels, value[source_mask], indices) for key, value in preds.items()}
    report = {"n_subjects": int(len(sites)), "n_source": int(source_mask.sum()), "folds": args.folds,
              "fold_seed": 20261007, "bootstrap_seed": 20260913, "replicates": 2000, "training": log,
              "results": {f"{k}/trained_{s}/tested_{t}": interval(*draws[(k, s, t)]) for (k, s, t) in draws}}
    for kind in ("image", "histogram"):
        for source in ("haca3", "preproc"):
            own, raw = draws[(kind, source, source)], draws[(kind, "raw", source)]
            report["results"][f"{kind}/own_minus_raw_trained/{source}"] = interval(own[0] - raw[0], own[1] - raw[1])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for key, value in report["results"].items():
        print(f"{key:48s} {value['estimate']:.3f} {np.round(value['ci95'], 3).tolist()}")


if __name__ == "__main__":
    main()
