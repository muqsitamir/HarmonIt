import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

import mlflow

from harmonit.data.abide_slices_dataset import AbideSlicesDataset
from harmonit.utils.plotting import save_confusion_matrix_png
from harmonit.utils.metrics import confusion_and_balanced_acc
from harmonit.utils.system import load_class_names_from_manifest


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    manifest_path = os.getenv("MANIFEST_PATH", "data/abide_manifest.csv")
    splits_path = os.getenv("SPLITS_PATH", "data/splits.json")

    ckpt_path = Path(os.getenv("CKPT_PATH", "checkpoints/site_probe/model_best.pt"))
    assert ckpt_path.exists(), f"Missing CKPT_PATH: {ckpt_path}"

    # Make eval output dir unique per checkpoint + timestamp (avoid overwriting)
    run_tag = f"{ckpt_path.parent.name}__{ckpt_path.stem}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(os.getenv("EVAL_OUT_DIR", "runs/eval_site_probe")) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    num_workers = int(os.getenv("NUM_WORKERS", "4"))
    seed = int(os.getenv("SEED", "123"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use SAME env-driven preprocessing switches as training:
    mask_mode = os.getenv("MASK_MODE", "none")
    bg_suppress = os.getenv("BG_SUPPRESS", "1") == "1"
    head_mask_thr = float(os.getenv("HEAD_MASK_THR", "0.02"))
    head_mask_dilate = int(os.getenv("HEAD_MASK_DILATE", "3"))
    input_mode = os.getenv("INPUT_MODE", "image")

    # --- Evaluation mode ---
    # det: deterministic validation (slice_mode=fixed)
    # ms:  multi-slice evaluation (K random slices per subject; slice_mode=random) aggregated across runs
    eval_mode = os.getenv("EVAL_MODE", "det").lower()  # det | ms
    ms_k = int(os.getenv("MS_K", "1"))
    ms_seed_stride = int(os.getenv("MS_SEED_STRIDE", "1000"))
    if eval_mode not in {"det", "ms"}:
        raise ValueError(f"EVAL_MODE must be 'det' or 'ms', got: {eval_mode}")
    if eval_mode == "ms" and ms_k < 1:
        raise ValueError(f"MS_K must be >= 1, got: {ms_k}")

    def build_val_dataset(ds_seed: int, slice_mode: str):
        return AbideSlicesDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="val",
            out_hw=(256, 256),
            slice_mode=slice_mode,
            valid_nonzero_frac=float(os.getenv("VALID_FG_FRAC", "0.02")),
            fg_bbox_thr=float(os.getenv("FG_BBOX_THR", "0.02")),
            seed=ds_seed,
            mask_mode=mask_mode,
            bg_suppress=bg_suppress,
            head_mask_thr=head_mask_thr,
            head_mask_dilate=head_mask_dilate,
            input_mode=input_mode,
        )

    # Load checkpoint once (support plain and DataParallel keys)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Infer num_classes from checkpoint head to prevent shape mismatches
    fc_key = "fc.weight"
    if fc_key not in state and "module.fc.weight" in state:
        fc_key = "module.fc.weight"
    num_classes = int(state[fc_key].shape[0])

    # Class names from manifest (must align with training mapping)
    class_names = load_class_names_from_manifest(manifest_path)
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names={len(class_names)} != ckpt num_classes={num_classes}. "
            f"This usually means the manifest site_id mapping changed or a different cohort was used."
        )

    print("Num classes:", num_classes)
    print("Class names:", class_names)

    # Model: ResNet18 1-channel
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Strip DataParallel 'module.' prefix if present
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    def run_once(ds_seed: int, slice_mode: str):
        ds_local = build_val_dataset(ds_seed=ds_seed, slice_mode=slice_mode)
        loader_local = DataLoader(
            ds_local,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        yt, yp = [], []
        with torch.no_grad():
            for x, y, _, _ in loader_local:
                x = x.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                yt.append(y.numpy())
                yp.append(pred)

        return np.concatenate(yt), np.concatenate(yp)

    if eval_mode == "det":
        y_true, y_pred = run_once(ds_seed=seed, slice_mode="fixed")
        tag = "DET"
    else:
        ys_t, ys_p = [], []
        for r in range(ms_k):
            ds_seed = seed + r * ms_seed_stride
            yt, yp = run_once(ds_seed=ds_seed, slice_mode="random")
            ys_t.append(yt)
            ys_p.append(yp)
        y_true = np.concatenate(ys_t)
        y_pred = np.concatenate(ys_p)
        tag = f"MS(K={ms_k})"

    cm, acc, bal = confusion_and_balanced_acc(y_true, y_pred, num_classes)
    print(f"[{tag} FULL VAL] acc={acc:.4f} | bal_acc={bal:.4f} | N={len(y_true)}")

    # Save artifacts
    np.save(out_dir / "cm_fullval.npy", cm)
    save_confusion_matrix_png(cm, out_dir / "cm_fullval.png", class_names=class_names)

    metrics = {
        "val_acc_full": acc,
        "val_bal_acc_full": bal,
        "n_val_samples": int(len(y_true)),
        "ckpt_path": str(ckpt_path),
        "run_tag": run_tag,
        "eval_mode": eval_mode,
        "ms_k": int(ms_k),
        "ms_seed_stride": int(ms_seed_stride),
        "mask_mode": mask_mode,
        "bg_suppress": bg_suppress,
        "head_mask_thr": head_mask_thr,
        "head_mask_dilate": head_mask_dilate,
        "input_mode": input_mode,
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Optional MLflow logging (if you want it)
    if os.getenv("MLFLOW_LOG", "0") == "1":
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("site_probe_eval")
        with mlflow.start_run(run_name=f"eval__{ckpt_path.stem}"):
            mlflow.log_params({
                "ckpt_path": str(ckpt_path),
                "run_tag": run_tag,
                "eval_mode": eval_mode,
                "ms_k": int(ms_k),
                "ms_seed_stride": int(ms_seed_stride),
                "mask_mode": mask_mode,
                "bg_suppress": int(bg_suppress),
                "head_mask_thr": head_mask_thr,
                "head_mask_dilate": head_mask_dilate,
                "input_mode": input_mode,
                "batch_size": batch_size,
            })
            mlflow.log_metric("val_acc_full", acc)
            mlflow.log_metric("val_bal_acc_full", bal)
            mlflow.log_artifact(str(out_dir / "cm_fullval.png"))
            mlflow.log_artifact(str(out_dir / "eval_metrics.json"))


if __name__ == "__main__":
    main()