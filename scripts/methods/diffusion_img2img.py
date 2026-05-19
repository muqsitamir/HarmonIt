"""Site-conditional diffusion img2img baseline for ABIDE harmonization.

This trains an unpaired site-conditional DDPM on raw fixed/augmented slices.
At export time it uses an SDEdit/DDIM-style image-to-image path: partially
noise each source slice, then denoise it while conditioning on the NYU site.
Moderate noise strength keeps anatomy anchored while allowing site appearance
to move toward the reference domain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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


METHOD = "diffusion_img2img_nyu"


def parse_out_hw(values: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(v) for v in values)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--out-hw expects one integer or two integers")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/export a site-conditional diffusion img2img baseline.")
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
    common.add_argument("--num-timesteps", type=int, default=1000)
    common.add_argument("--beta-start", type=float, default=1e-4)
    common.add_argument("--beta-end", type=float, default=0.02)
    common.add_argument("--base-channels", type=int, default=64)
    common.add_argument("--channel-mults", nargs="+", type=int, default=(1, 2, 4, 4))

    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--out-dir", default="outputs/harmonized/diffusion_img2img_nyu")
    train.add_argument("--train-split", default="train", choices=("train", "val", "test"))
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--steps", type=int, default=20000)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--ema-decay", type=float, default=0.999)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--log-every", type=int, default=100)
    train.add_argument("--save-every", type=int, default=2000)
    train.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional checkpoint to continue training from. Loads model/EMA and optimizer when present.",
    )

    export = subparsers.add_parser("export", parents=[common])
    export.add_argument("--split", default="test", choices=("train", "val", "test"))
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out-dir", default="outputs/harmonized/diffusion_img2img_nyu")
    export.add_argument("--batch-size", type=int, default=8)
    export.add_argument("--num-workers", type=int, default=2)
    export.add_argument("--ddim-steps", type=int, default=50)
    export.add_argument("--strength", type=float, default=0.35, help="Fraction of the diffusion chain used for img2img.")
    export.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity; 0 is deterministic.")
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
    dataset.aug_affine = slice_mode != "fixed"
    return dataset


def site_count_from_manifest(manifest_path: str | Path) -> int:
    frame = pd.read_csv(manifest_path)
    if "site_id" not in frame.columns:
        site_map = {site: idx for idx, site in enumerate(sorted(frame["site"].unique()))}
        frame["site_id"] = frame["site"].map(site_map).astype(int)
    return int(frame["site_id"].max()) + 1


def site_name_from_manifest(manifest_path: str | Path, site_id: int) -> str:
    frame = pd.read_csv(manifest_path)
    if "site_id" not in frame.columns:
        site_map = {site: idx for idx, site in enumerate(sorted(frame["site"].unique()))}
        frame["site_id"] = frame["site"].map(site_map).astype(int)
    rows = frame.loc[frame["site_id"] == int(site_id), "site"]
    return str(rows.iloc[0]) if len(rows) else str(site_id)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int) -> None:
        super().__init__()
        groups1 = min(8, in_channels)
        groups2 = min(8, out_channels)
        self.norm1 = nn.GroupNorm(groups1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb)).view(emb.shape[0], -1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class ConditionalUNet(nn.Module):
    def __init__(self, num_sites: int, base_channels: int = 64, channel_mults: tuple[int, ...] = (1, 2, 4, 4)) -> None:
        super().__init__()
        emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.site_emb = nn.Embedding(num_sites, emb_dim)
        self.in_conv = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * mult for mult in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_channels
        skips: list[int] = []
        for level, out_ch in enumerate(channels):
            block = ResBlock(in_ch, out_ch, emb_dim)
            self.down_blocks.append(block)
            skips.append(out_ch)
            self.downsamples.append(Downsample(out_ch) if level < len(channels) - 1 else nn.Identity())
            in_ch = out_ch

        self.mid1 = ResBlock(in_ch, in_ch, emb_dim)
        self.mid2 = ResBlock(in_ch, in_ch, emb_dim)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level, skip_ch in reversed(list(enumerate(skips))):
            self.up_blocks.append(ResBlock(in_ch + skip_ch, skip_ch, emb_dim))
            self.upsamples.append(Upsample(skip_ch) if level > 0 else nn.Identity())
            in_ch = skip_ch

        self.out_norm = nn.GroupNorm(min(8, in_ch), in_ch)
        self.out_conv = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t) + self.site_emb(site.long())
        h = self.in_conv(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid1(h, emb), emb)
        for block, up in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), emb)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


class DiffusionSchedule:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise


def to_model_range(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def foreground_mask(x01: torch.Tensor, thr: float) -> torch.Tensor:
    return (x01 > thr).float()


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)


def train(args: argparse.Namespace) -> None:
    out_hw = parse_out_hw(args.out_hw)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available; diffusion training will be slow.", flush=True)

    dataset = build_dataset(args, split=args.train_split, slice_mode="random", out_hw=out_hw)
    sampler = RandomSampler(dataset, replacement=True, num_samples=args.steps * args.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    num_sites = site_count_from_manifest(args.manifest_path)
    channel_mults = tuple(int(v) for v in args.channel_mults)
    model = ConditionalUNet(num_sites=num_sites, base_channels=args.base_channels, channel_mults=channel_mults).to(device)
    ema_model = ConditionalUNet(num_sites=num_sites, base_channels=args.base_channels, channel_mults=channel_mults).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = DiffusionSchedule(args.num_timesteps, args.beta_start, args.beta_end, device)
    start_step = 0
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if resume_path.exists():
            payload = torch.load(resume_path, map_location=device)
            model.load_state_dict(payload["model"])
            ema_model.load_state_dict(payload.get("ema_model", payload["model"]))
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload.get("step", 0))
            print(f"Resumed from {resume_path} at step={start_step}", flush=True)
        else:
            print(f"Resume checkpoint not found, starting fresh: {resume_path}", flush=True)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": METHOD,
        "train_split": args.train_split,
        "target_site_id": args.target_site_id,
        "target_site_name": site_name_from_manifest(args.manifest_path, args.target_site_id),
        "num_sites": num_sites,
        "out_hw": list(out_hw),
        "num_timesteps": args.num_timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "base_channels": args.base_channels,
        "channel_mults": list(channel_mults),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.time()
    running_loss = 0.0
    model.train()
    for local_step, (images, sites, _subjects, _slices) in enumerate(loader, start=1):
        step = start_step + local_step
        if step > args.steps:
            break
        images = to_model_range(images.to(device, non_blocking=True).float())
        sites = sites.to(device, non_blocking=True).long()
        t = torch.randint(0, args.num_timesteps, (images.shape[0],), device=device)
        noise = torch.randn_like(images)
        noisy = schedule.q_sample(images, t, noise)
        pred = model(noisy, t, sites)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        update_ema(model, ema_model, args.ema_decay)

        running_loss += float(loss.detach().cpu())
        if step % args.log_every == 0 or step == 1:
            avg = running_loss / (args.log_every if step % args.log_every == 0 else 1)
            running_loss = 0.0
            elapsed = time.time() - start
            print(f"step={step} loss={avg:.6f} elapsed_min={elapsed / 60.0:.1f}", flush=True)

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = ckpt_dir / f"model_step_{step:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                },
                ckpt_path,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                },
                ckpt_dir / "model_latest.pt",
            )
            print(f"saved={ckpt_path}", flush=True)


def load_model(checkpoint: Path, device: torch.device) -> tuple[ConditionalUNet, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device)
    config = dict(payload["config"])
    model = ConditionalUNet(
        num_sites=int(config["num_sites"]),
        base_channels=int(config["base_channels"]),
        channel_mults=tuple(int(v) for v in config["channel_mults"]),
    ).to(device)
    state = payload.get("ema_model") or payload["model"]
    model.load_state_dict(state)
    model.eval()
    return model, config


@torch.no_grad()
def ddim_img2img(
    model: ConditionalUNet,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    target_sites: torch.Tensor,
    strength: float,
    steps: int,
    eta: float,
) -> torch.Tensor:
    strength = float(np.clip(strength, 0.01, 1.0))
    t_start = max(1, min(schedule.timesteps - 1, int(round(strength * (schedule.timesteps - 1)))))
    inference_steps = max(2, min(int(steps), t_start + 1))
    times = torch.linspace(t_start, 0, inference_steps, device=x0.device).round().long()
    times = torch.unique_consecutive(times)
    if times[-1].item() != 0:
        times = torch.cat([times, torch.zeros(1, device=x0.device, dtype=torch.long)])

    noise = torch.randn_like(x0)
    current = schedule.q_sample(x0, torch.full((x0.shape[0],), int(times[0].item()), device=x0.device, dtype=torch.long), noise)
    for i in range(len(times) - 1):
        t = times[i].repeat(x0.shape[0])
        t_next = times[i + 1].repeat(x0.shape[0])
        eps = model(current, t, target_sites)
        a_t = schedule.alpha_bars[t].view(-1, 1, 1, 1)
        a_next = schedule.alpha_bars[t_next].view(-1, 1, 1, 1)
        pred_x0 = (current - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if eta > 0 and int(t_next[0].item()) > 0:
            sigma = eta * torch.sqrt((1.0 - a_next) / (1.0 - a_t) * (1.0 - a_t / a_next))
            direction_scale = torch.sqrt((1.0 - a_next - sigma**2).clamp_min(0.0))
            current = torch.sqrt(a_next) * pred_x0 + direction_scale * eps + sigma * torch.randn_like(current)
        else:
            current = torch.sqrt(a_next) * pred_x0 + torch.sqrt(1.0 - a_next) * eps
    return current.clamp(-1.0, 1.0)


def save_manifest(path: Path, subject_ids: np.ndarray, site_ids: np.ndarray, slice_indices: np.ndarray, split: str) -> None:
    frame = pd.DataFrame(
        {
            "row_idx": np.arange(len(subject_ids), dtype=np.int64),
            "subject_id": subject_ids,
            "site_id": site_ids,
            "slice_idx": slice_indices,
            "split": split,
        }
    )
    frame.to_csv(path, index=False)


def save_export_qc(
    raw_images: np.ndarray,
    harmonized_images: np.ndarray,
    subject_ids: np.ndarray,
    site_ids: np.ndarray,
    path: Path,
    max_cols: int = 8,
) -> None:
    import matplotlib.pyplot as plt

    n = min(max_cols, raw_images.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.2))
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col in range(n):
        axes[0, col].imshow(raw_images[col, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{subject_ids[col]}\nsite {site_ids[col]}", fontsize=8)
        axes[1, col].imshow(harmonized_images[col, 0], cmap="gray", vmin=0, vmax=1)
        for row in range(2):
            axes[row, col].axis("off")
    axes[0, 0].set_ylabel("raw", fontsize=9)
    axes[1, 0].set_ylabel("diffusion", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export(args: argparse.Namespace) -> None:
    out_hw = parse_out_hw(args.out_hw)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg = load_model(Path(args.checkpoint), device=device)
    schedule = DiffusionSchedule(
        int(train_cfg["num_timesteps"]),
        float(train_cfg["beta_start"]),
        float(train_cfg["beta_end"]),
        device,
    )

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
            raw01 = images.to(device, non_blocking=True).float()
            sites_t = sites.to(device, non_blocking=True).long()
            target_sites = torch.full_like(sites_t, int(args.target_site_id))
            harmonized = ddim_img2img(
                model,
                schedule,
                to_model_range(raw01),
                target_sites,
                strength=args.strength,
                steps=args.ddim_steps,
                eta=args.eta,
            )
            fake01 = to_image_range(harmonized)
            fake01 = fake01 * foreground_mask(raw01, args.foreground_thr)
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
        raise AssertionError("Diffusion output contains NaN or inf values")

    output_dir = Path(args.out_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "diffusion_img2img_nyu_slices.npz"
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
            output_dir / "qc_raw_vs_diffusion_img2img_nyu.png",
        )

    config = {
        "method": METHOD,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "target_site_id": args.target_site_id,
        "target_site_name": site_name_from_manifest(args.manifest_path, args.target_site_id),
        "identity_target": bool(args.identity_target),
        "strength": args.strength,
        "ddim_steps": args.ddim_steps,
        "eta": args.eta,
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
