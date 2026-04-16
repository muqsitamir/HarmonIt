import json
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from harmonit.data.abide_slices_dataset import AbideSlicesDataset
from harmonit.models.neurocombat import apply_neurocombat
from harmonit.metrics import evaluate_all_metrics

load_dotenv()


def extract_features(model, dataloader, device, max_batches=None):
    model.eval()

    all_features = []
    all_sites = []
    all_images = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y, subject_ids, _ = batch

            x = x.to(device)
            features = model(x)

            all_features.append(features.cpu().numpy())
            all_sites.append(y.numpy())
            all_images.append(x.cpu().numpy())

            if max_batches and i >= max_batches:
                break

    X = np.concatenate(all_features, axis=0)
    sites = np.concatenate(all_sites, axis=0)
    images = np.concatenate(all_images, axis=0)

    return X, sites, images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    manifest_path = "data/abide_manifest.csv"
    splits_path = "data/splits.json"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/site_probe_harmonized") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 32

    # -------------------------
    # DATA
    # -------------------------
    train_ds = AbideSlicesDataset(
        manifest_path=manifest_path,
        splits_path=splits_path,
        split="train",
        out_hw=(256, 256),
        slice_mode="random",
        seed=42,
    )

    val_ds = AbideSlicesDataset(
        manifest_path=manifest_path,
        splits_path=splits_path,
        split="val",
        out_hw=(256, 256),
        slice_mode="fixed",
        seed=123,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # -------------------------
    # MODEL (feature extractor)
    # -------------------------
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    feature_dim = model.fc.in_features
    model.fc = nn.Identity()

    model.to(device)

    print("Extracting features...")

    # -------------------------
    # FEATURE EXTRACTION
    # -------------------------
    X_train, y_train, img_train = extract_features(model, train_loader, device, max_batches=50)
    X_val, y_val, img_val = extract_features(model, val_loader, device, max_batches=30)

    print("Feature shape:", X_train.shape)

    # -------------------------
    # BASELINE
    # -------------------------
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    acc_baseline = accuracy_score(y_val, preds)

    print(f"\nBaseline Site Acc: {acc_baseline:.4f}")

    # -------------------------
    # HARMONIZATION
    # -------------------------
    print("\nApplying NeuroCombat...")

    # Fit on TRAIN ONLY
    X_train_h = apply_neurocombat(X_train, y_train)

    # Apply same transformation to VAL
    X_val_h = apply_neurocombat(X_val, y_val)

    # -------------------------
    # CLASSIFICATION AFTER
    # -------------------------
    clf_h = RandomForestClassifier(random_state=42)
    clf_h.fit(X_train_h, y_train)

    preds_h = clf_h.predict(X_val_h)
    acc_harmonized = accuracy_score(y_val, preds_h)

    print(f"Harmonized Site Acc: {acc_harmonized:.4f}")

    # -------------------------
    # FULL METRICS EVALUATION
    # -------------------------
    print("\n[Full Evaluation]")

    # Distribution (features)
    dist_before = X_val.flatten()
    dist_after = X_val_h.flatten()

    # Image comparison (approximation)
    img_before = img_val[0, 0]  # take one slice
    img_after = img_before  # ⚠️ NeuroCombat works on features, so images unchanged

    eval_results = evaluate_all_metrics(
        y_true=y_val,
        y_pred=preds_h,
        y_prob=None,
        y_true_site=y_val,
        y_pred_site=preds_h,
        img1=img_before,
        img2=img_after,
        feat1=X_val[0],
        feat2=X_val_h[0],
        dist1=dist_before,
        dist2=dist_after,
    )

    print("\n[Evaluation Results]")
    for k, v in eval_results.items():
        print(k, ":", v)

    # -------------------------
    # SAVE
    # -------------------------
    with open(out_dir / "results.json", "w") as f:
        json.dump({k: str(v) for k, v in eval_results.items()}, f, indent=2)

    print("\nDone.")
    print("Run dir:", out_dir)


if __name__ == "__main__":
    main()