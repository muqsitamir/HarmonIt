"""Train the adapted HCLD conditional latent diffusion module on ABIDE volumes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


class HcldVolumeDataset(Dataset):
    def __init__(self, data_dir: str | Path, labels_path: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.volume_dir = self.data_dir / "volumes"
        self.labels = pd.read_csv(self.data_dir / labels_path, sep="\t")
        if len(self.labels) == 0:
            raise ValueError(f"No rows in labels: {self.data_dir / labels_path}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.labels.iloc[idx]
        filename = str(row["filename"])
        volume = np.load(self.volume_dir / f"{filename}.npy").astype(np.float32, copy=False)
        if volume.ndim == 3:
            volume = volume[None]
        return {
            "image": torch.from_numpy(volume),
            "fn": filename,
            "site": torch.tensor(int(row["site"]), dtype=torch.long),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train adapted HCLD cLDM on HarmonIt ABIDE volumes.")
    parser.add_argument("--hcld-root", default="/home/muqsitamir/repos/HCLD")
    parser.add_argument("--data-dir", default="outputs/hcld_abide")
    parser.add_argument("--out-dir", default="outputs/hcld_abide/adapted_hcld/ldm")
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--train-src-labels", default="labels/train_src.tsv")
    parser.add_argument("--train-tar-labels", default="labels/train_tar.tsv")
    parser.add_argument("--val-src-labels", default="labels/val_src.tsv")
    parser.add_argument("--val-tar-labels", default="labels/val_tar.tsv")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--style-d-lr", type=float, default=1e-5)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--val-every-steps", type=int, default=1000)
    parser.add_argument("--diff-loss", choices=("l1", "l2"), default="l2")
    parser.add_argument("--diff-weight", type=float, default=1.0)
    parser.add_argument("--content-weight", type=float, default=10.0)
    parser.add_argument("--style-weight", type=float, default=10.0)
    parser.add_argument("--adv-style-weight", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=10000.0)
    parser.add_argument("--burn-in-steps", type=int, default=4000)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--use-flash-attention", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_hcld_to_path(hcld_root: str | Path) -> None:
    hcld_root = Path(hcld_root)
    if str(hcld_root) not in sys.path:
        sys.path.insert(0, str(hcld_root))


def amp_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16


def cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "cuda_mem=NA"
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    return (
        f"cuda_mem_alloc_gb={allocated:.2f} cuda_mem_reserved_gb={reserved:.2f} "
        f"cuda_mem_peak_alloc_gb={peak_allocated:.2f} cuda_mem_peak_reserved_gb={peak_reserved:.2f}"
    )


def build_autoencoder(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    from generative.networks.nets import AutoencoderKL

    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64),
        latent_channels=6,
        num_res_blocks=2,
        norm_num_groups=8,
        attention_levels=(False, False, True),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_flash_attention=False,
        use_checkpointing=True,
    ).to(device)
    payload = torch.load(args.ae_checkpoint, map_location=device)
    autoencoder.load_state_dict(payload["model_state_dict"])
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad_(False)
    return autoencoder


def build_unet(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    from generative.networks.nets import DiffusionModelUNet

    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=12,
        out_channels=6,
        num_res_blocks=1,
        num_channels=(32, 64, 64),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 64),
        use_flash_attention=bool(args.use_flash_attention),
    ).to(device)


def get_mean_std(input_tensor: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels = input_tensor.shape[:2]
    flat = input_tensor.view(batch, channels, -1)
    mean = torch.mean(flat, dim=2).view(batch, channels, 1, 1, 1)
    std = torch.sqrt(torch.var(flat, dim=2) + eps).view(batch, channels, 1, 1, 1)
    return mean, std


def instance_norm(content: torch.Tensor) -> torch.Tensor:
    mean, std = get_mean_std(content)
    return (content - mean) / std


def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width, depth = input_tensor.size()
    features = input_tensor.view(batch * channels, height * width * depth)
    return torch.mm(features, features.t()).div(batch * channels * height * width * depth)


def style_loss_gram(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(gram_matrix(input_tensor), gram_matrix(target_tensor))


def torch_gradmap(img: torch.Tensor) -> torch.Tensor:
    dh = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]
    dw = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
    dz = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]
    grad_map = (dh[:, :, 1:, 1:, :] + dw[:, :, 1:, :, 1:] + dz[:, :, :, 1:, 1:]) / 3.0
    return torch.nn.functional.pad(grad_map, (1, 0, 1, 0, 1, 0), "constant", 0)


class StyleDiscriminator3d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv3d(6, 64, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv3d(64, 128, 4, 2, 1, bias=False),
            torch.nn.InstanceNorm3d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv3d(128, 256, 4, 2, 1, bias=False),
            torch.nn.InstanceNorm3d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.AdaptiveAvgPool3d((6, 6, 2)),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 6 * 6 * 2, 1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.main(input_tensor)


def save_checkpoint(
    path: Path,
    unet: torch.nn.Module,
    style_discriminator: torch.nn.Module,
    optimizer_diff: torch.optim.Optimizer,
    optimizer_style_d: torch.optim.Optimizer,
    scaler: GradScaler,
    config: dict[str, Any],
    step: int,
    epoch: int,
    scale_factor: float,
) -> None:
    torch.save(
        {
            "unet_state_dict": unet.state_dict(),
            "s_D_state_dict": style_discriminator.state_dict(),
            "optimizer_diff": optimizer_diff.state_dict(),
            "optimizer_s_D": optimizer_style_d.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "step": step,
            "epoch": epoch,
            "scale_factor": float(scale_factor),
        },
        path,
    )


@torch.no_grad()
def compute_scale_factor(autoencoder: torch.nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype) -> float:
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True).float()
    with autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
        latents = autoencoder.encode_stage_2_inputs(images)
    scale = 1.0 / torch.std(latents.float()).clamp_min(1e-6)
    return float(scale.detach().cpu())


@torch.no_grad()
def validate(
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scheduler: Any,
    inferer: Any,
    loader_src: DataLoader,
    loader_tar: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 8,
) -> float:
    unet.eval()
    losses: list[float] = []
    for idx, (batch_src, batch_tar) in enumerate(zip(loader_src, cycle(loader_tar))):
        if idx >= max_batches:
            break
        images = batch_src["image"].to(device, non_blocking=True).float()
        conditions_img = batch_tar["image"].to(device, non_blocking=True).float()
        with autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            conditions = inferer.scale_factor * autoencoder.encode_stage_2_inputs(conditions_img)
            noise = torch.randn_like(conditions)
            timesteps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (images.shape[0],),
                device=device,
            ).long()
            noise_pred, _noisy_image, _latent, _ = inferer(
                inputs=images,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=timesteps,
                condition=conditions,
                mode="concat",
            )
            losses.append(float(F.mse_loss(noise_pred.float(), noise.float()).detach().cpu()))
    unet.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = build_arg_parser().parse_args()
    add_hcld_to_path(args.hcld_root)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(args)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    from generative.inferers import LatentDiffusionInferer
    from generative.networks.schedulers import DDPMScheduler

    train_src = HcldVolumeDataset(args.data_dir, args.train_src_labels)
    train_tar = HcldVolumeDataset(args.data_dir, args.train_tar_labels)
    val_src = HcldVolumeDataset(args.data_dir, args.val_src_labels)
    val_tar = HcldVolumeDataset(args.data_dir, args.val_tar_labels)
    train_loader_src = DataLoader(
        train_src,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    train_loader_tar = DataLoader(
        train_tar,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader_src = DataLoader(val_src, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader_tar = DataLoader(val_tar, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    autoencoder = build_autoencoder(args, device)
    scale_factor = compute_scale_factor(autoencoder, train_loader_src, device, dtype)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor, Adain="AdaIN_reverse")
    unet = build_unet(args, device)
    style_discriminator = StyleDiscriminator3d().to(device)

    optimizer_diff = torch.optim.Adam(unet.parameters(), lr=args.lr)
    optimizer_style_d = torch.optim.Adam(style_discriminator.parameters(), lr=args.style_d_lr, betas=(0.5, 0.999))
    scaler_enabled = device.type == "cuda" and args.amp_dtype == "fp16"
    scaler = GradScaler(device.type, enabled=scaler_enabled)
    style_criterion = torch.nn.BCEWithLogitsLoss()

    start_step = 0
    start_epoch = 0
    if args.resume_checkpoint:
        ckpt = Path(args.resume_checkpoint)
        if ckpt.exists():
            payload = torch.load(ckpt, map_location=device)
            unet.load_state_dict(payload["unet_state_dict"])
            style_discriminator.load_state_dict(payload["s_D_state_dict"])
            optimizer_diff.load_state_dict(payload["optimizer_diff"])
            optimizer_style_d.load_state_dict(payload["optimizer_s_D"])
            scaler.load_state_dict(payload["scaler_state_dict"])
            start_step = int(payload.get("step", 0))
            start_epoch = int(payload.get("epoch", 0))
            scale_factor = float(payload.get("scale_factor", scale_factor))
            inferer.scale_factor = scale_factor
            print(f"Resumed from {ckpt} at step={start_step} epoch={start_epoch}", flush=True)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "train_src_size": len(train_src),
            "train_tar_size": len(train_tar),
            "val_src_size": len(val_src),
            "val_tar_size": len(val_tar),
            "device": str(device),
            "amp_torch_dtype": str(dtype),
            "scale_factor": scale_factor,
            "started_at": dt.datetime.now().isoformat(),
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, sort_keys=True), flush=True)

    step = start_step
    started = time.time()
    unet.train()
    style_discriminator.train()
    optimizer_diff.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, (batch_src, batch_tar) in enumerate(zip(train_loader_src, cycle(train_loader_tar))):
            step += 1
            images = batch_src["image"].to(device, non_blocking=True).float()
            conditions_img = batch_tar["image"].to(device, non_blocking=True).float()

            with autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    conditions = autoencoder.encode_stage_2_inputs(conditions_img) * scale_factor
                noise = torch.randn_like(conditions)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
                noise_pred, noisy_image, latent, _ = inferer(
                    inputs=images,
                    autoencoder_model=autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                    condition=conditions,
                    mode="concat",
                )
                loss_diff = (
                    F.mse_loss(noise_pred.float(), noise.float())
                    if args.diff_loss == "l2"
                    else F.l1_loss(noise_pred.float(), noise.float())
                )
                noisy_latent = noisy_image[:, : noisy_image.shape[1] // 2]
                x0_pred = torch.zeros_like(noise_pred)
                for item_idx in range(noise_pred.shape[0]):
                    _prev, pred_original = scheduler.step(
                        noise_pred[item_idx].unsqueeze(0),
                        timesteps[item_idx],
                        noisy_latent[item_idx].unsqueeze(0),
                    )
                    x0_pred[item_idx] = pred_original[0]

                loss_content = F.mse_loss(instance_norm(latent.float()), instance_norm(x0_pred.float()))
                decoded = autoencoder.decode_stage_2_outputs((x0_pred / scale_factor).float())
                loss_grad = F.mse_loss(torch_gradmap(images.float()), torch_gradmap(decoded.float()))
                loss_style = style_loss_gram(x0_pred.float(), conditions.float())

                s_real = style_discriminator(conditions.float().detach()).view(-1)
                s_fake = style_discriminator(latent.float().detach()).view(-1)
                loss_style_d = style_criterion(s_real, torch.ones_like(s_real)) + style_criterion(
                    s_fake, torch.zeros_like(s_fake)
                )

            optimizer_style_d.zero_grad(set_to_none=True)
            scaler.scale(loss_style_d).backward()
            scaler.step(optimizer_style_d)

            with autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                if step > args.burn_in_steps:
                    s_recon = style_discriminator(x0_pred.float()).view(-1)
                    loss_style_adv = style_criterion(s_recon, torch.ones_like(s_recon))
                else:
                    loss_style_adv = torch.zeros((), device=device)
                loss = (
                    args.diff_weight * loss_diff
                    + args.content_weight * loss_content
                    + args.style_weight * loss_style
                    + args.gradient_weight * loss_grad
                    + args.adv_style_weight * loss_style_adv
                ) / max(1, args.accumulation_steps)

            scaler.scale(loss).backward()
            if step % max(1, args.accumulation_steps) == 0:
                scaler.step(optimizer_diff)
                scaler.update()
                optimizer_diff.zero_grad(set_to_none=True)
            else:
                scaler.update()

            finite_loss = torch.isfinite(loss.detach()) and torch.isfinite(loss_style_d.detach())
            if not bool(finite_loss):
                raise RuntimeError(f"Non-finite loss at step={step}: loss={loss.item()} style_d={loss_style_d.item()}")

            if step == 1 or step % 25 == 0:
                elapsed = (time.time() - started) / 60.0
                print(
                    f"step={step} epoch={epoch} diff={float(loss_diff.detach().cpu()):.6f} "
                    f"content={float(loss_content.detach().cpu()):.6f} style={float(loss_style.detach().cpu()):.6f} "
                    f"grad={float(loss_grad.detach().cpu()):.6f} adv_style={float(loss_style_adv.detach().cpu()):.6f} "
                    f"style_d={float(loss_style_d.detach().cpu()):.6f} elapsed_min={elapsed:.1f} "
                    f"{cuda_memory_summary(device)}",
                    flush=True,
                )

            if step % args.val_every_steps == 0:
                val_diff = validate(autoencoder, unet, scheduler, inferer, val_loader_src, val_loader_tar, device, dtype)
                print(f"validation step={step} val_diff={val_diff:.6f} {cuda_memory_summary(device)}", flush=True)

            if step % args.save_every_steps == 0:
                save_checkpoint(
                    ckpt_dir / f"model_step_{step:06d}.pt",
                    unet,
                    style_discriminator,
                    optimizer_diff,
                    optimizer_style_d,
                    scaler,
                    config,
                    step,
                    epoch,
                    scale_factor,
                )
                save_checkpoint(
                    ckpt_dir / "model_latest.pt",
                    unet,
                    style_discriminator,
                    optimizer_diff,
                    optimizer_style_d,
                    scaler,
                    config,
                    step,
                    epoch,
                    scale_factor,
                )
                print(f"saved={ckpt_dir / 'model_latest.pt'}", flush=True)

            del images, conditions_img, conditions, noise, noise_pred, noisy_image, latent, x0_pred, decoded
            if args.max_steps is not None and step >= args.max_steps:
                save_checkpoint(
                    ckpt_dir / "model_latest.pt",
                    unet,
                    style_discriminator,
                    optimizer_diff,
                    optimizer_style_d,
                    scaler,
                    config,
                    step,
                    epoch,
                    scale_factor,
                )
                return

    save_checkpoint(
        ckpt_dir / "model_latest.pt",
        unet,
        style_discriminator,
        optimizer_diff,
        optimizer_style_d,
        scaler,
        config,
        step,
        args.epochs,
        scale_factor,
    )


if __name__ == "__main__":
    main()
