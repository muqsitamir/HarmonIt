import json
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import mlflow
import mlflow.pytorch
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from data.abide_slices_dataset import AbideSlicesDataset

load_dotenv()

def confusion_and_balanced_acc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    # per-class recall; avoid divide-by-zero
    recalls = []
    for c in range(num_classes):
        denom = cm[c, :].sum()
        if denom > 0:
            recalls.append(cm[c, c] / denom)
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return cm, acc, bal_acc


def save_confusion_matrix_png(cm: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Site Probe Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Paths
    manifest_path = "data/abide_manifest.csv"
    splits_path = "data/splits.json"

    # Output run dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/site_probe") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparams (CPU-friendly defaults)
    batch_size = 32
    epochs = 3
    lr = 3e-4
    steps_per_epoch = 50 # controls how many batches we draw (sampling w/ replacement)
    val_batches = 30

    # MLflow local tracking (falls back to ./mlflow.db if env var not set)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("site_probe")

    with mlflow.start_run(run_name=run_id):
        # log key params
        mlflow.log_params({
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "steps_per_epoch": steps_per_epoch,
            "val_batches": val_batches,
            "out_hw": "256x256",
            "slice_mode_train": "random",
            "slice_mode_val": "fixed",
            "valid_nonzero_frac": 0.02,
            "volume_cache_size": 12,
        })
        mlflow.set_tag("run_dir", str(out_dir))
        print("MLflow tracking URI:", mlflow.get_tracking_uri())

        # Datasets (note: slice_mode=random already returns random valid slice per subject)
        train_ds = AbideSlicesDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="train",
            out_hw=(256, 256),
            slice_mode="random",
            valid_nonzero_frac=0.02,
            seed=42,
            volume_cache_size=12,
        )
        val_ds = AbideSlicesDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="val",
            out_hw=(256, 256),
            slice_mode="fixed",  # deterministic validation
            valid_nonzero_frac=0.02,
            seed=123,
            volume_cache_size=12,
        )

        # Number of classes
        df = __import__("pandas").read_csv(manifest_path)
        num_classes = int(df["site_id"].max()) + 1 if "site_id" in df.columns else int(df["site"].nunique())
        print("Num classes (sites):", num_classes)
        with open(out_dir / "config.json", "w") as f:
            json.dump(
                {
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "steps_per_epoch": steps_per_epoch,
                    "val_batches": val_batches,
                    "num_classes": num_classes,
                },
                f,
                indent=2,
            )
        # log config file as artifact (after writing)
        mlflow.log_artifact(str(out_dir / "config.json"))

        # Sampler with replacement lets us train longer than dataset length
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=RandomSampler(train_ds, replacement=True, num_samples=batch_size * steps_per_epoch),
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Model: ResNet18 adapted to 1-channel input
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(device)

        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_bal = -1.0

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            model.train()
            running_loss = 0.0

            for step, batch in enumerate(train_loader, start=1):
                x, y, _, _ = batch
                x = x.to(device)              # [B,1,256,256]
                y = y.to(device).long()

                optim.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optim.step()

                running_loss += float(loss.item())

                if step == 1 or step % 10 == 0:
                    elapsed = time.time() - epoch_start
                    print(f"Epoch {epoch} | step {step}/{steps_per_epoch} | loss {running_loss/step:.4f} | elapsed {elapsed:.1f}s")

                if step >= steps_per_epoch:
                    break

            # ---- Validation ----
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for i, batch in enumerate(val_loader, start=1):
                    x, y, _, _ = batch
                    x = x.to(device)
                    logits = model(x)
                    pred = torch.argmax(logits, dim=1).cpu().numpy()

                    y_true.append(y.numpy())
                    y_pred.append(pred)

                    if i >= val_batches:
                        break

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            cm, acc, bal = confusion_and_balanced_acc(y_true, y_pred, num_classes)

            epoch_elapsed = time.time() - epoch_start
            print(f"[VAL] Epoch {epoch} | acc={acc:.4f} | bal_acc={bal:.4f} | epoch_time={epoch_elapsed:.1f}s")
            mlflow.log_metric("val_acc", acc, step=epoch)
            mlflow.log_metric("val_bal_acc", bal, step=epoch)

            # Save artifacts
            torch.save(model.state_dict(), out_dir / f"model_epoch{epoch}.pt")
            np.save(out_dir / f"cm_epoch{epoch}.npy", cm)
            save_confusion_matrix_png(cm, out_dir / f"cm_epoch{epoch}.png")
            mlflow.log_artifact(str(out_dir / f"cm_epoch{epoch}.png"))

            # Track best
            if bal > best_val_bal:
                best_val_bal = bal
                torch.save(model.state_dict(), out_dir / "model_best.pt")
                mlflow.log_artifact(str(out_dir / "model_best.pt"))

        print("Done. Best val balanced accuracy:", best_val_bal)
        print("Run dir:", out_dir)


if __name__ == "__main__":
    main()