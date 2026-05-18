"""2D StarGAN-style multi-domain baseline for ABIDE harmonization.

This baseline learns one generator conditioned on a target site label. Training
uses adversarial, domain-classification, reconstruction, identity, and gradient
preservation terms. Export translates deterministic fixed-slice benchmark
artifacts to a requested target site and writes the standard HarmonIt NPZ.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


METHOD = "stargan_nyu"


def parse_out_hw(values: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(v) for v in values)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--out-hw expects one integer or two integers")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/export a 2D StarGAN-style baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest-path", default="data/abide_manifest.csv")
    common.add_argument("--splits-path", default="data/splits.json")
    common.add_argument("--out-hw", nargs="+", type=int, default=(256, 256))
    common.add_argument("--num-classes", type=int, default=17)
    common.add_argument("--target-site-id", type=int, default=5, help="Default 5 = NYU.")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--valid-nonzero-frac", type=float, default=float(os.getenv("VALID_FG_FRAC", "0.02")))
    common.add_argument("--fg-bbox-thr", type=float, default=float(os.getenv("FG_BBOX_THR", "0.02")))
    common.add_argument("--foreground-thr", type=float, default=0.02)
    common.add_argument("--volume-cache-size", type=int, default=12)

    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--out-dir", default="outputs/harmonized/stargan_nyu")
    train.add_argument("--train-split", default="train", choices=("train", "val", "test"))
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--beta1", type=float, default=0.5)
    train.add_argument("--base-channels", type=int, default=32)
    train.add_argument("--res-blocks", type=int, default=4)
    train.add_argument("--lambda-cls", type=float, default=1.0)
    train.add_argument("--lambda-rec", type=float, default=10.0)
    train.add_argument("--lambda-id", type=float, default=1.0)
    train.add_argument("--lambda-grad", type=float, default=1.0)
    train.add_argument("--target-prob", type=float, default=0.35)
    train.add_argument("--log-every", type=int, default=25)
    train.add_argument("--save-every", type=int, default=500)

    export = subparsers.add_parser("export", parents=[common])
    export.add_argument("--split", default="test", choices=("train", "val", "test"))
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out-dir", default="outputs/harmonized/stargan_nyu")
    export.add_argument("--batch-size", type=int, default=16)
    export.add_argument("--num-workers", type=int, default=2)
    export.add_argument("--identity-target", action="store_true", default=True)
    export.add_argument("--translate-target", dest="identity_target", action="store_false")
    export.add_argument("--no-qc", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(args: argparse.Namespace, split: str, slice_mode: str, out_hw: tuple[int, int]) -> Any:
    from harmonit.data.abide_slices_dataset import AbideSlicesDataset

    dataset = AbideSlicesDataset(
        manifest_path=args.manifest_path,
        splits_path=args.splits_path,
        split=split,
        out_hw=out_hw,
        slice_mode=slice_mode,
        valid_nonzero_frac=args.valid_nonzero_frac,
        fg_bbox_thr=args.fg_bbox_thr,
        seed=args.seed,
        volume_cache_size=args.volume_cache_size,
        mask_mode="none",
        bg_suppress=True,
        input_mode="image",
    )
    dataset.aug_affine = False
    return dataset


def site_names_from_manifest(manifest_path: str, num_classes: int) -> list[str]:
    df = pd.read_csv(manifest_path)
    if "site_id" not in df.columns:
        site_map = {site: idx for idx, site in enumerate(sorted(df["site"].unique()))}
        df["site_id"] = df["site"].map(site_map)
    rows = df[["site_id", "site"]].drop_duplicates().sort_values("site_id")
    names = [str(i) for i in range(num_classes)]
    for _, row in rows.iterrows():
        site_id = int(row["site_id"])
        if 0 <= site_id < num_classes:
            names[site_id] = str(row["site"])
    return names


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class StarGenerator(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32, res_blocks: int = 4) -> None:
        super().__init__()
        c = base_channels
        in_channels = 1 + num_classes
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, c, kernel_size=7, bias=False),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.ReLU(inplace=True),
        ]
        layers.extend(ResBlock(c * 4) for _ in range(res_blocks))
        layers.extend(
            [
                nn.ConvTranspose2d(c * 4, c * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(c * 2, affine=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c * 2, c, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(c, affine=True),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(c, 1, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(labels.shape[0], labels.shape[1], 1, 1)
        labels = labels.expand(labels.shape[0], labels.shape[1], x.shape[2], x.shape[3])
        return self.net(torch.cat([x, labels], dim=1))


class StarDiscriminator(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        def block(in_c: int, out_c: int, stride: int, norm: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *block(1, c, stride=2, norm=False),
            *block(c, c * 2, stride=2),
            *block(c * 2, c * 4, stride=2),
            *block(c * 4, c * 8, stride=2),
            *block(c * 8, c * 16, stride=2),
        )
        self.src_head = nn.Conv2d(c * 16, 1, kernel_size=3, padding=1)
        self.cls_head = nn.Linear(c * 16, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        src = self.src_head(feat)
        pooled = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        cls = self.cls_head(pooled)
        return src, cls


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm2d) and module.weight is not None:
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


def to_model_range(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).float()


def sample_target_labels(
    source_labels: torch.Tensor,
    num_classes: int,
    target_site_id: int,
    target_prob: float,
) -> torch.Tensor:
    device = source_labels.device
    random_labels = torch.randint(0, num_classes, size=source_labels.shape, device=device)
    same = random_labels == source_labels
    random_labels[same] = (random_labels[same] + 1) % num_classes
    target_labels = torch.full_like(source_labels, fill_value=target_site_id)
    choose_target = torch.rand(source_labels.shape, device=device) < target_prob
    return torch.where(choose_target, target_labels, random_labels)


def foreground_mask(x01: torch.Tensor, foreground_thr: float) -> torch.Tensor:
    return (x01 > foreground_thr).float()


def masked_output(fake_m11: torch.Tensor, source_m11: torch.Tensor, foreground_thr: float) -> torch.Tensor:
    source01 = to_image_range(source_m11)
    mask = foreground_mask(source01, foreground_thr)
    fake01 = to_image_range(fake_m11) * mask
    return to_model_range(fake01)


def gradient_map(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.cat([dx, dy], dim=1)


@dataclass
class TrainConfig:
    method: str
    num_classes: int
    target_site_id: int
    target_site_name: str
    out_hw: list[int]
    steps: int
    batch_size: int
    lr: float
    beta1: float
    base_channels: int
    res_blocks: int
    lambda_cls: float
    lambda_rec: float
    lambda_id: float
    lambda_grad: float
    target_prob: float
    train_split: str
    seed: int


def save_qc_grid(path: Path, tensors: dict[str, torch.Tensor], max_items: int = 4) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping QC image because matplotlib could not be imported: {exc}")
        return

    keys = list(tensors)
    n = min(max_items, next(iter(tensors.values())).shape[0])
    fig, axes = plt.subplots(len(keys), n, figsize=(2.2 * n, 2.2 * len(keys)), squeeze=False)
    for row, key in enumerate(keys):
        arr = to_image_range(tensors[key].detach().cpu()).numpy()
        for col in range(n):
            axes[row, col].imshow(arr[col, 0], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(key, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    step: int,
    cfg: TrainConfig,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
) -> None:
    torch.save(
        {
            "step": step,
            "config": asdict(cfg),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    out_hw = parse_out_hw(args.out_hw)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / "train"
    ckpt_dir = out_dir / "checkpoints"
    qc_dir = out_dir / "qc"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    site_names = site_names_from_manifest(args.manifest_path, args.num_classes)
    cfg = TrainConfig(
        method=METHOD,
        num_classes=args.num_classes,
        target_site_id=args.target_site_id,
        target_site_name=site_names[args.target_site_id],
        out_hw=list(out_hw),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        base_channels=args.base_channels,
        res_blocks=args.res_blocks,
        lambda_cls=args.lambda_cls,
        lambda_rec=args.lambda_rec,
        lambda_id=args.lambda_id,
        lambda_grad=args.lambda_grad,
        target_prob=args.target_prob,
        train_split=args.train_split,
        seed=args.seed,
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    dataset = build_dataset(args, split=args.train_split, slice_mode="random", out_hw=out_hw)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset, replacement=True, num_samples=args.batch_size * args.steps),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    print(f"Device: {device}")
    print(f"Training subjects: {len(dataset)}")
    print(f"Target export site: {site_names[args.target_site_id]} (site_id={args.target_site_id})")

    generator = StarGenerator(args.num_classes, args.base_channels, args.res_blocks).to(device)
    discriminator = StarDiscriminator(args.num_classes, args.base_channels).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    gan_loss = nn.MSELoss()
    cls_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()

    iterator = iter(loader)
    start = time.time()
    latest: dict[str, torch.Tensor] = {}

    for step in range(1, args.steps + 1):
        x, labels, _, _ = next(iterator)
        x = x.to(device).float()
        labels = labels.to(device).long()
        target_labels = sample_target_labels(labels, args.num_classes, args.target_site_id, args.target_prob)

        real = to_model_range(x)
        target_onehot = one_hot(target_labels, args.num_classes)
        source_onehot = one_hot(labels, args.num_classes)

        pred_real_src, pred_real_cls = discriminator(real)
        fake = masked_output(generator(real, target_onehot), real, args.foreground_thr)
        pred_fake_src, _ = discriminator(fake.detach())
        loss_d_adv = 0.5 * (
            gan_loss(pred_real_src, torch.ones_like(pred_real_src))
            + gan_loss(pred_fake_src, torch.zeros_like(pred_fake_src))
        )
        loss_d_cls = cls_loss(pred_real_cls, labels)
        loss_d = loss_d_adv + args.lambda_cls * loss_d_cls

        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        fake = masked_output(generator(real, target_onehot), real, args.foreground_thr)
        rec = masked_output(generator(fake, source_onehot), real, args.foreground_thr)
        ident = masked_output(generator(real, source_onehot), real, args.foreground_thr)
        pred_fake_src, pred_fake_cls = discriminator(fake)

        loss_g_adv = gan_loss(pred_fake_src, torch.ones_like(pred_fake_src))
        loss_g_cls = cls_loss(pred_fake_cls, target_labels)
        loss_rec = l1_loss(rec, real)
        loss_id = l1_loss(ident, real)
        loss_grad = l1_loss(gradient_map(fake), gradient_map(real))
        loss_g = (
            loss_g_adv
            + args.lambda_cls * loss_g_cls
            + args.lambda_rec * loss_rec
            + args.lambda_id * loss_id
            + args.lambda_grad * loss_grad
        )

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        latest = {"raw": real, "target": fake, "recon": rec, "identity": ident}

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"step {step:05d}/{args.steps} "
                f"G={float(loss_g.item()):.4f} D={float(loss_d.item()):.4f} "
                f"g_adv={float(loss_g_adv.item()):.4f} g_cls={float(loss_g_cls.item()):.4f} "
                f"rec={float(loss_rec.item()):.4f} id={float(loss_id.item()):.4f} "
                f"grad={float(loss_grad.item()):.4f} d_cls={float(loss_d_cls.item()):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        if step == 1 or step % args.save_every == 0 or step == args.steps:
            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            save_checkpoint(ckpt_path, step, cfg, generator, discriminator, opt_g, opt_d)
            save_checkpoint(ckpt_dir / "latest.pt", step, cfg, generator, discriminator, opt_g, opt_d)
            if latest:
                save_qc_grid(qc_dir / f"step_{step:06d}.png", latest)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)


def load_generator(checkpoint_path: Path, device: torch.device) -> tuple[StarGenerator, dict[str, Any]]:
    state = torch.load(str(checkpoint_path), map_location=device)
    cfg = state.get("config", {})
    generator = StarGenerator(
        num_classes=int(cfg.get("num_classes", 17)),
        base_channels=int(cfg.get("base_channels", 32)),
        res_blocks=int(cfg.get("res_blocks", 4)),
    ).to(device)
    generator.load_state_dict(state["generator"], strict=True)
    generator.eval()
    return generator, cfg


def save_manifest(
    path: Path,
    subject_ids: np.ndarray,
    site_ids: np.ndarray,
    slice_indices: np.ndarray,
    split: str,
) -> None:
    manifest = pd.DataFrame(
        {
            "row_idx": np.arange(len(subject_ids), dtype=np.int64),
            "subject_id": subject_ids,
            "site_id": site_ids,
            "slice_idx": slice_indices,
            "method": METHOD,
            "split": split,
        }
    )
    manifest.to_csv(path, index=False)


def save_export_qc(
    raw_images: np.ndarray,
    harmonized_images: np.ndarray,
    subject_ids: np.ndarray,
    site_ids: np.ndarray,
    path: Path,
    max_subjects: int = 6,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping QC image because matplotlib could not be imported: {exc}")
        return

    n = min(max_subjects, raw_images.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 4.8), squeeze=False)
    for col in range(n):
        axes[0, col].imshow(raw_images[col, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"{subject_ids[col]}\nsite {site_ids[col]}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(harmonized_images[col, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("raw", fontsize=10)
    axes[1, 0].set_ylabel(METHOD, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export(args: argparse.Namespace) -> None:
    out_hw = parse_out_hw(args.out_hw)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, train_cfg = load_generator(Path(args.checkpoint), device=device)
    num_classes = int(train_cfg.get("num_classes", args.num_classes))

    dataset = build_dataset(args, split=args.split, slice_mode="fixed", out_hw=out_hw)
    dataset.aug_affine = False
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    harmonized_batches: list[np.ndarray] = []
    raw_batches: list[np.ndarray] = []
    subject_ids: list[str] = []
    site_ids: list[int] = []
    slice_indices: list[int] = []

    with torch.no_grad():
        for images, sites, subjects, slices in loader:
            images = images.to(device).float()
            sites_t = sites.to(device).long()
            raw_m11 = to_model_range(images)
            target_labels = torch.full_like(sites_t, fill_value=args.target_site_id)
            fake01 = to_image_range(generator(raw_m11, one_hot(target_labels, num_classes)))
            fake01 = fake01 * foreground_mask(images, args.foreground_thr)
            if args.identity_target:
                target_mask = (sites_t == args.target_site_id).view(-1, 1, 1, 1)
                fake01 = torch.where(target_mask, images, fake01)

            raw_batches.append(images.cpu().numpy().astype(np.float32, copy=False))
            harmonized_batches.append(fake01.cpu().numpy().astype(np.float32, copy=False))
            subject_ids.extend(str(subject_id) for subject_id in subjects)
            site_ids.extend(int(site_id) for site_id in sites.cpu().numpy().tolist())
            slice_indices.extend(int(slice_idx) for slice_idx in slices.cpu().numpy().tolist())

    raw_images = np.concatenate(raw_batches, axis=0)
    harmonized_images = np.concatenate(harmonized_batches, axis=0)
    subject_ids_arr = np.asarray(subject_ids, dtype=str)
    site_ids_arr = np.asarray(site_ids, dtype=np.int64)
    slice_indices_arr = np.asarray(slice_indices, dtype=np.int64)
    if not np.isfinite(harmonized_images).all():
        raise AssertionError("StarGAN output contains NaN or inf values")

    output_dir = Path(args.out_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "stargan_nyu_slices.npz"
    np.savez_compressed(
        npz_path,
        images=harmonized_images,
        raw_images=raw_images,
        subject_ids=subject_ids_arr,
        site_ids=site_ids_arr,
        slice_indices=slice_indices_arr,
        split=np.asarray(args.split),
        method=np.asarray(METHOD),
    )
    save_manifest(output_dir / "manifest.csv", subject_ids_arr, site_ids_arr, slice_indices_arr, args.split)
    if not args.no_qc:
        save_export_qc(
            raw_images,
            harmonized_images,
            subject_ids_arr,
            site_ids_arr,
            output_dir / "qc_raw_vs_stargan_nyu.png",
        )

    site_names = site_names_from_manifest(args.manifest_path, num_classes)
    config = {
        "method": METHOD,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "target_site_id": args.target_site_id,
        "target_site_name": site_names[args.target_site_id],
        "identity_target": bool(args.identity_target),
        "out_hw": list(out_hw),
        "train_config": train_cfg,
        "n_subjects": int(raw_images.shape[0]),
        "n_sites": int(len(np.unique(site_ids_arr))),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved: {npz_path}")
    print(f"Saved: {output_dir / 'manifest.csv'}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "export":
        export(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
