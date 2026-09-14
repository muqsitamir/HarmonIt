"""Train a site probe on exported fixed slices (raw or harmonized) for ISBI 2027.

Same architecture, optimizer, budget and affine augmentation as train_site_probe.py,
but images come from NPZ exports held in memory, so a probe trained on a method's own
outputs can be compared with a probe trained on the matching raw slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harmonit.data.label_controls import shuffled_subject_labels  # noqa: E402
from harmonit.utils.metrics import confusion_and_balanced_acc  # noqa: E402

NUM_CLASSES = 17


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(path, key, split, splits):
    with np.load(path, allow_pickle=False) as data:
        images = np.asarray(data[key], dtype=np.float32)
        subjects = data["subject_ids"].astype(str)
        sites = data["site_ids"].astype(np.int64)
        slices = data["slice_indices"].astype(np.int64)
        stored_split = str(np.asarray(data["split"]).item()) if "split" in data.files else None
    if stored_split != split:
        raise ValueError(f"{path}: split {stored_split!r}, expected {split!r}")
    if images.ndim != 4 or images.shape[1:] != (1, 256, 256) or not np.isfinite(images).all():
        raise ValueError(f"{path}: expected finite [N,1,256,256] {key}")
    if len(set(subjects)) != len(subjects) or set(subjects) != set(splits[split]):
        raise ValueError(f"{path}: subjects differ from the {split} split")
    return images, sites, subjects, slices


def augment(batch, rng, prob=0.9, rot=12.0, trans=32, scale_jitter=0.2):
    """Per-sample affine matching AbideSlicesDataset._maybe_apply_affine."""
    out = batch.clone()
    for i in range(batch.shape[0]):
        if prob < 1.0 and rng.rand() > prob:
            continue
        angle = float(rng.uniform(-rot, rot))
        tx, ty = int(rng.randint(-trans, trans + 1)), int(rng.randint(-trans, trans + 1))
        scale = float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter))
        out[i] = TF.affine(batch[i], angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                           interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    return out


def predict(model, images, device, batch_size=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start:start + batch_size]).to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-npz", required=True)
    p.add_argument("--val-npz", required=True)
    p.add_argument("--image-key", choices=("images", "raw_images"), required=True,
                   help="images = method output; raw_images = embedded raw slices")
    p.add_argument("--splits-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--label-shuffle", action="store_true", help="Subject-level shuffled-label control")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        p.error("Output directory exists; choose a new one")
    splits = json.loads(Path(args.splits_path).read_text())
    x_train, y_train, s_train, k_train = load_split(args.train_npz, args.image_key, "train", splits)
    x_val, y_val, s_val, k_val = load_split(args.val_npz, args.image_key, "val", splits)
    if args.label_shuffle:
        y_train = shuffled_subject_labels(y_train, 12345)
        y_val = shuffled_subject_labels(y_val, 12346)

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out.mkdir(parents=True)
    config = dict(vars(args), device=str(device), n_train=len(y_train), n_val=len(y_val),
                  train_npz_sha256=sha256(args.train_npz), val_npz_sha256=sha256(args.val_npz),
                  augmentation=dict(prob=0.9, rot_deg=12.0, trans_px=32, scale_jitter=0.2),
                  sampling="uniform with replacement over subjects, one fixed slice each")
    if args.label_shuffle:
        (out / "shuffled_subject_labels.json").write_text(json.dumps(
            {"train": dict(zip(s_train, map(int, y_train))), "val": dict(zip(s_val, map(int, y_val)))}, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            idx = rng.randint(0, len(y_train), args.batch_size)
            x = augment(torch.from_numpy(x_train[idx]).to(device), rng)
            y = torch.from_numpy(y_train[idx]).to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
        cm, acc, bal = confusion_and_balanced_acc(y_val, predict(model, x_val, device), NUM_CLASSES)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_acc=acc, val_bal_acc=bal))
        print(f"epoch {epoch} loss {np.mean(losses):.4f} val_acc {acc:.4f} val_bal_acc {bal:.4f}", flush=True)
        np.save(out / f"cm_epoch{epoch}.npy", cm)
        if bal > best:
            best = bal
            torch.save(model.state_dict(), out / "model_best.pt")
    torch.save(model.state_dict(), out / "model_last.pt")
    (out / "history.json").write_text(json.dumps(dict(history=history, best_val_bal_acc=best), indent=2) + "\n")
    print(f"Done. best val BA {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
