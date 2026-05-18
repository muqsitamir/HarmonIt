"""Many-to-one 2D CycleGAN baseline for ABIDE fixed-slice harmonization.

The training setup treats all non-reference sites as domain A and one reference
site as domain B. At export time, non-reference test slices are translated with
G_A2B, while reference-site slices can be left unchanged. The resulting NPZ
uses the same artifact contract as the NeuroCombat and histogram baselines.
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


METHOD = "cyclegan_nyu"


def parse_out_hw(values: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(v) for v in values)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--out-hw expects one integer or two integers")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/export a many-to-one 2D CycleGAN baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest-path", default="data/abide_manifest.csv")
    common.add_argument("--splits-path", default="data/splits.json")
    common.add_argument("--out-hw", nargs="+", type=int, default=(256, 256))
    common.add_argument("--target-site-id", type=int, default=5, help="Default 5 = NYU.")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--valid-nonzero-frac", type=float, default=float(os.getenv("VALID_FG_FRAC", "0.02")))
    common.add_argument("--fg-bbox-thr", type=float, default=float(os.getenv("FG_BBOX_THR", "0.02")))
    common.add_argument("--foreground-thr", type=float, default=0.02)
    common.add_argument("--volume-cache-size", type=int, default=12)

    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--out-dir", default="outputs/harmonized/cyclegan_nyu")
    train.add_argument("--train-split", default="train", choices=("train", "val", "test"))
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--steps", type=int, default=1500)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--beta1", type=float, default=0.5)
    train.add_argument("--base-channels", type=int, default=32)
    train.add_argument("--res-blocks", type=int, default=4)
    train.add_argument("--lambda-cycle", type=float, default=10.0)
    train.add_argument("--lambda-identity", type=float, default=5.0)
    train.add_argument("--lambda-gradient", type=float, default=1.0)
    train.add_argument("--log-every", type=int, default=25)
    train.add_argument("--save-every", type=int, default=250)

    export = subparsers.add_parser("export", parents=[common])
    export.add_argument("--split", default="test", choices=("train", "val", "test"))
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out-dir", default="outputs/harmonized/cyclegan_nyu")
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


def filter_dataset_by_domain(dataset: Any, target_site_id: int, domain: str) -> None:
    if domain == "target":
        dataset.samples = [sample for sample in dataset.samples if int(sample.site_id) == target_site_id]
    elif domain == "source":
        dataset.samples = [sample for sample in dataset.samples if int(sample.site_id) != target_site_id]
    else:
        raise ValueError(f"Unknown domain: {domain}")
    dataset._valid_slices.clear()
    dataset._vol_cache.clear()
    if not dataset.samples:
        raise ValueError(f"No samples left after filtering domain={domain}")


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


class ResnetGenerator(nn.Module):
    def __init__(self, base_channels: int = 32, res_blocks: int = 4) -> None:
        super().__init__()
        c = base_channels
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, c, kernel_size=7, bias=False),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        def block(in_c: int, out_c: int, stride: int, norm: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(1, c, stride=2, norm=False),
            *block(c, c * 2, stride=2),
            *block(c * 2, c * 4, stride=2),
            *block(c * 4, c * 8, stride=1),
            nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
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


def gradient_map(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.cat([dx, dy], dim=1)


def foreground_mask(x01: torch.Tensor, foreground_thr: float) -> torch.Tensor:
    return (x01 > foreground_thr).float()


def masked_output(fake_m11: torch.Tensor, source_m11: torch.Tensor, foreground_thr: float) -> torch.Tensor:
    source01 = to_image_range(source_m11)
    mask = foreground_mask(source01, foreground_thr)
    fake01 = to_image_range(fake_m11) * mask
    return to_model_range(fake01)


@dataclass
class TrainConfig:
    method: str
    target_site_id: int
    target_site_name: str
    out_hw: list[int]
    steps: int
    batch_size: int
    lr: float
    beta1: float
    base_channels: int
    res_blocks: int
    lambda_cycle: float
    lambda_identity: float
    lambda_gradient: float
    train_split: str
    seed: int


def site_name_from_manifest(manifest_path: str, target_site_id: int) -> str:
    df = pd.read_csv(manifest_path)
    if "site_id" not in df.columns:
        site_map = {site: idx for idx, site in enumerate(sorted(df["site"].unique()))}
        df["site_id"] = df["site"].map(site_map)
    rows = df[df["site_id"] == target_site_id]
    if rows.empty:
        return str(target_site_id)
    return str(rows["site"].iloc[0])


def make_loaders(args: argparse.Namespace, out_hw: tuple[int, int]) -> tuple[DataLoader, DataLoader, Any, Any]:
    source_ds = build_dataset(args, split=args.train_split, slice_mode="random", out_hw=out_hw)
    target_ds = build_dataset(args, split=args.train_split, slice_mode="random", out_hw=out_hw)
    filter_dataset_by_domain(source_ds, target_site_id=args.target_site_id, domain="source")
    filter_dataset_by_domain(target_ds, target_site_id=args.target_site_id, domain="target")

    source_loader = DataLoader(
        source_ds,
        batch_size=args.batch_size,
        sampler=RandomSampler(source_ds, replacement=True, num_samples=args.batch_size * args.steps),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=args.batch_size,
        sampler=RandomSampler(target_ds, replacement=True, num_samples=args.batch_size * args.steps),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    return source_loader, target_loader, source_ds, target_ds


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
    g_a2b: nn.Module,
    g_b2a: nn.Module,
    d_a: nn.Module,
    d_b: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
) -> None:
    torch.save(
        {
            "step": step,
            "config": asdict(cfg),
            "g_a2b": g_a2b.state_dict(),
            "g_b2a": g_b2a.state_dict(),
            "d_a": d_a.state_dict(),
            "d_b": d_b.state_dict(),
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

    target_site_name = site_name_from_manifest(args.manifest_path, args.target_site_id)
    cfg = TrainConfig(
        method=METHOD,
        target_site_id=args.target_site_id,
        target_site_name=target_site_name,
        out_hw=list(out_hw),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        base_channels=args.base_channels,
        res_blocks=args.res_blocks,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        lambda_gradient=args.lambda_gradient,
        train_split=args.train_split,
        seed=args.seed,
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    source_loader, target_loader, source_ds, target_ds = make_loaders(args, out_hw)
    print(f"Device: {device}")
    print(f"Domain A source subjects: {len(source_ds)}")
    print(f"Domain B target subjects: {len(target_ds)} ({target_site_name}, site_id={args.target_site_id})")

    g_a2b = ResnetGenerator(args.base_channels, args.res_blocks).to(device)
    g_b2a = ResnetGenerator(args.base_channels, args.res_blocks).to(device)
    d_a = PatchDiscriminator(args.base_channels).to(device)
    d_b = PatchDiscriminator(args.base_channels).to(device)
    for model in (g_a2b, g_b2a, d_a, d_b):
        model.apply(init_weights)

    opt_g = torch.optim.Adam(
        list(g_a2b.parameters()) + list(g_b2a.parameters()),
        lr=args.lr,
        betas=(args.beta1, 0.999),
    )
    opt_d = torch.optim.Adam(
        list(d_a.parameters()) + list(d_b.parameters()),
        lr=args.lr,
        betas=(args.beta1, 0.999),
    )
    gan_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    start = time.time()
    latest: dict[str, torch.Tensor] = {}

    for step in range(1, args.steps + 1):
        x_a = next(source_iter)[0].to(device)
        x_b = next(target_iter)[0].to(device)
        a = to_model_range(x_a)
        b = to_model_range(x_b)

        fake_b = masked_output(g_a2b(a), a, args.foreground_thr)
        rec_a = masked_output(g_b2a(fake_b), a, args.foreground_thr)
        fake_a = masked_output(g_b2a(b), b, args.foreground_thr)
        rec_b = masked_output(g_a2b(fake_a), b, args.foreground_thr)
        idt_b = masked_output(g_a2b(b), b, args.foreground_thr)
        idt_a = masked_output(g_b2a(a), a, args.foreground_thr)

        pred_fake_b = d_b(fake_b)
        pred_fake_a = d_a(fake_a)
        valid_b = torch.ones_like(pred_fake_b)
        valid_a = torch.ones_like(pred_fake_a)

        loss_gan = gan_loss(pred_fake_b, valid_b) + gan_loss(pred_fake_a, valid_a)
        loss_cycle = l1_loss(rec_a, a) + l1_loss(rec_b, b)
        loss_identity = l1_loss(idt_a, a) + l1_loss(idt_b, b)
        loss_gradient = l1_loss(gradient_map(fake_b), gradient_map(a)) + l1_loss(gradient_map(fake_a), gradient_map(b))
        loss_g = (
            loss_gan
            + args.lambda_cycle * loss_cycle
            + args.lambda_identity * loss_identity
            + args.lambda_gradient * loss_gradient
        )

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        with torch.no_grad():
            fake_b_det = fake_b.detach()
            fake_a_det = fake_a.detach()

        pred_real_a = d_a(a)
        pred_fake_a = d_a(fake_a_det)
        pred_real_b = d_b(b)
        pred_fake_b = d_b(fake_b_det)
        loss_d_a = 0.5 * (
            gan_loss(pred_real_a, torch.ones_like(pred_real_a))
            + gan_loss(pred_fake_a, torch.zeros_like(pred_fake_a))
        )
        loss_d_b = 0.5 * (
            gan_loss(pred_real_b, torch.ones_like(pred_real_b))
            + gan_loss(pred_fake_b, torch.zeros_like(pred_fake_b))
        )
        loss_d = loss_d_a + loss_d_b

        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        latest = {"A": a, "A2B": fake_b, "A2B2A": rec_a, "B": b, "B2A": fake_a, "B2A2B": rec_b}

        if step == 1 or step % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"step {step:05d}/{args.steps} "
                f"G={float(loss_g.item()):.4f} D={float(loss_d.item()):.4f} "
                f"gan={float(loss_gan.item()):.4f} cyc={float(loss_cycle.item()):.4f} "
                f"id={float(loss_identity.item()):.4f} grad={float(loss_gradient.item()):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        if step == 1 or step % args.save_every == 0 or step == args.steps:
            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            save_checkpoint(ckpt_path, step, cfg, g_a2b, g_b2a, d_a, d_b, opt_g, opt_d)
            save_checkpoint(ckpt_dir / "latest.pt", step, cfg, g_a2b, g_b2a, d_a, d_b, opt_g, opt_d)
            if latest:
                save_qc_grid(qc_dir / f"step_{step:06d}.png", latest)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)


def load_generator(checkpoint_path: Path, device: torch.device) -> tuple[ResnetGenerator, dict[str, Any]]:
    state = torch.load(str(checkpoint_path), map_location=device)
    cfg = state.get("config", {})
    g = ResnetGenerator(
        base_channels=int(cfg.get("base_channels", 32)),
        res_blocks=int(cfg.get("res_blocks", 4)),
    ).to(device)
    g.load_state_dict(state["g_a2b"], strict=True)
    g.eval()
    return g, cfg


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
        raw = raw_images[col, 0]
        harm = harmonized_images[col, 0]
        axes[0, col].imshow(raw, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"{subject_ids[col]}\nsite {site_ids[col]}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(harm, cmap="gray", vmin=0.0, vmax=1.0)
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
            images = images.to(device)
            sites_t = sites.to(device)
            raw01 = images.float()
            input_m11 = to_model_range(raw01)
            fake01 = to_image_range(generator(input_m11))
            mask = foreground_mask(raw01, args.foreground_thr)
            fake01 = fake01 * mask
            if args.identity_target:
                target_mask = (sites_t == args.target_site_id).view(-1, 1, 1, 1)
                fake01 = torch.where(target_mask, raw01, fake01)

            raw_batches.append(raw01.cpu().numpy().astype(np.float32, copy=False))
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
        raise AssertionError("CycleGAN output contains NaN or inf values")

    output_dir = Path(args.out_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "cyclegan_nyu_slices.npz"
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
            output_dir / "qc_raw_vs_cyclegan_nyu.png",
        )

    config = {
        "method": METHOD,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "target_site_id": args.target_site_id,
        "target_site_name": site_name_from_manifest(args.manifest_path, args.target_site_id),
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
