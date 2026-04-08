import json
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import mlflow
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from harmonit.data.abide_slices_dataset import AbideSlicesDataset
from harmonit.models.neurocombat import apply_neurocombat

load_dotenv()


def extract_features(model, dataloader, device, max_batches=None):
    model.eval()

    all_features = []
    all_sites = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y, subject_ids, _ = batch

            x = x.to(device)
            features = model(x)  # shape: [B, feature_dim]

            all_features.append(features.cpu().numpy())
            all_sites.append(y.numpy())

            if max_batches and i >= max_batches:
                break

    X = np.concatenate(all_features, axis=0)
    sites = np.concatenate(all_sites, axis=0)

    return X, sites


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    manifest_path = "data/abide_manifest.csv"
    splits_path = "data/splits.json"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/site_probe_harmonized") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 32

    # MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("site_probe_harmonized")

    with mlflow.start_run(run_name=run_id):

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
        model.fc = nn.Identity()  # REMOVE classifier

        model.to(device)

        print("Extracting features...")

        # -------------------------
        # FEATURE EXTRACTION
        # -------------------------
        X_train, y_train = extract_features(model, train_loader, device, max_batches=100)
        X_val, y_val = extract_features(model, val_loader, device, max_batches=50)

        print("Feature shape:", X_train.shape)

        # -------------------------
        # BASELINE (no harmonization)
        # -------------------------
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        acc_baseline = accuracy_score(y_val, preds)

        print(f"Baseline Site Acc: {acc_baseline:.4f}")
        mlflow.log_metric("baseline_site_acc", acc_baseline)

        # -------------------------
        # HARMONIZATION
        # -------------------------
        print("Applying NeuroCombat...")

        # Combine train + val for harmonization
        X_all = np.concatenate([X_train, X_val], axis=0)
        sites_all = np.concatenate([y_train, y_val], axis=0)

        # Apply NeuroCombat ONCE
        X_all_h = apply_neurocombat(X_all, sites_all)

        # Split back
        X_train_h = X_all_h[:len(X_train)]
        X_val_h = X_all_h[len(X_train):]

        # -------------------------
        # CLASSIFICATION AFTER HARMONIZATION
        # -------------------------
        clf_h = RandomForestClassifier(random_state=42)
        clf_h.fit(X_train_h, y_train)

        preds_h = clf_h.predict(X_val_h)
        acc_harmonized = accuracy_score(y_val, preds_h)

        print(f"Harmonized Site Acc: {acc_harmonized:.4f}")
        mlflow.log_metric("harmonized_site_acc", acc_harmonized)

        # -------------------------
        # SAVE RESULTS
        # -------------------------
        results = {
            "baseline_site_acc": float(acc_baseline),
            "harmonized_site_acc": float(acc_harmonized),
            "feature_dim": int(feature_dim),
        }

        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact(str(out_dir / "results.json"))

        print("\nDone.")
        print("Run dir:", out_dir)


if __name__ == "__main__":
    main()