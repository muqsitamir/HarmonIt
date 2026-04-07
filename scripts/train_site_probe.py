import json
from pathlib import Path
from datetime import datetime
import os
import hashlib
import subprocess
from dotenv import load_dotenv
import mlflow
import mlflow.pytorch
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from harmonit.data.abide_slices_dataset import AbideSlicesDataset

load_dotenv()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_info() -> dict:
    """Best-effort git fingerprinting. Returns commit SHA and dirty flag."""
    info = {"git_commit": "unknown", "git_dirty": "unknown"}
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["git_commit"] = sha
        code = subprocess.call(["git", "diff", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        info["git_dirty"] = (code != 0)
    except Exception:
        pass
    return info

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

    preproc_cfg = {
        "preproc_version": "v0.1",
        "canonical_orientation": True,
        "intensity_clip_pcts": "1,99",
        "zscore_mask": "nonzero",
        "slice_plane": "axial",
        "slice_range_frac": "0.15,0.85",
        "slice_fg_thr": 0.05,
        "valid_fg_frac": 0.02,
        "out_hw": "256x256",
        "volume_cache_size": 12,
    }

    # Data + code fingerprints for reproducibility
    manifest_hash = sha256_file(Path(manifest_path))
    splits_hash = sha256_file(Path(splits_path))
    ginfo = git_info()

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
            "valid_nonzero_frac": float(preproc_cfg["valid_fg_frac"]),
            "volume_cache_size": int(preproc_cfg["volume_cache_size"]),
        })
        mlflow.set_tag("run_dir", str(out_dir))
        # Log preprocessing config and fingerprints
        mlflow.log_params({f"preproc_{k}": (str(v) if isinstance(v, bool) else v) for k, v in preproc_cfg.items()})
        mlflow.set_tag("abide_manifest_sha256", manifest_hash)
        mlflow.set_tag("splits_sha256", splits_hash)
        mlflow.set_tag("git_commit", ginfo.get("git_commit", "unknown"))
        mlflow.set_tag("git_dirty", str(ginfo.get("git_dirty", "unknown")))
        print("MLflow tracking URI:", mlflow.get_tracking_uri())

        # Datasets (note: slice_mode=random already returns random valid slice per subject)
        train_ds = AbideSlicesDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="train",
            out_hw=(256, 256),
            slice_mode="random",
            valid_nonzero_frac=float(preproc_cfg["valid_fg_frac"]),
            seed=42,
            volume_cache_size=int(preproc_cfg["volume_cache_size"]),
        )
        val_ds = AbideSlicesDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="val",
            out_hw=(256, 256),
            slice_mode="fixed",  # deterministic validation
            valid_nonzero_frac=float(preproc_cfg["valid_fg_frac"]),
            seed=123,
            volume_cache_size=int(preproc_cfg["volume_cache_size"]),
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

        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "hyperparams": {
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "steps_per_epoch": steps_per_epoch,
                "val_batches": val_batches,
            },
            "preprocessing": preproc_cfg,
            "paths": {
                "manifest_path": manifest_path,
                "splits_path": splits_path,
            },
            "hashes": {
                "abide_manifest_sha256": manifest_hash,
                "splits_sha256": splits_hash,
            },
            "git": ginfo,
        }
        meta_path = out_dir / "run_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(meta_path))

        preproc_path = out_dir / "preprocessing.json"
        preproc_path.write_text(json.dumps(preproc_cfg, indent=2))
        mlflow.log_artifact(str(preproc_path))

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
                x, y, subject_ids, _ = batch
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
            from collections import defaultdict

            model.eval()

            # store predictions grouped by subject
            subject_preds = defaultdict(list)
            subject_targets = {}

            with torch.no_grad():
                for i, batch in enumerate(val_loader, start=1):
                    x, y, subject_ids, _ = batch

                    x = x.to(device)
                    logits = model(x)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()

                    y = y.numpy()

                    # group predictions by subject
                    for pred, target, sid in zip(preds, y, subject_ids):
                        subject_preds[sid].append(pred)
                        subject_targets[sid] = target

                    if i >= val_batches:
                        break

            # ---- Aggregate per subject ----
            final_preds = []
            final_targets = []

            for sid in subject_preds:
                preds = subject_preds[sid]

                # majority vote
                counts = np.bincount(preds, minlength=num_classes)
                final_pred = np.argmax(counts)

                final_preds.append(final_pred)
                final_targets.append(subject_targets[sid])

            y_true = np.array(final_targets)
            y_pred = np.array(final_preds)

            print("Validation subjects:", len(subject_preds))
            print("Total slices used:", sum(len(v) for v in subject_preds.values()))

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