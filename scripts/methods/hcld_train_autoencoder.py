"""Train the HCLD 3D AutoencoderKL on exported ABIDE volumes.

This is a thin, configurable wrapper around the official HCLD/MONAI
AutoencoderKL architecture. It keeps checkpoint keys compatible with the
official HCLD latent-diffusion script (``model_state_dict``) while avoiding the
hard-coded OpenBHB/IXI paths in the upstream training file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HCLD AutoencoderKL on HarmonIt ABIDE volumes.")
    parser.add_argument("--hcld-root", default="/home/muqsitamir/repos/HCLD")
    parser.add_argument("--data-dir", default="outputs/hcld_abide")
    parser.add_argument("--out-dir", default="outputs/hcld_abide/aekl")
    parser.add_argument("--train-labels", default="labels/train.tsv")
    parser.add_argument("--val-labels", default="labels/val.tsv")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard stop for smoke tests.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adv-lr", type=float, default=5e-4)
    parser.add_argument("--adv-weight", type=float, default=0.01)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--base-channels", nargs="+", type=int, default=(32, 64, 64))
    parser.add_argument("--latent-channels", type=int, default=6)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention", action="store_true", help="Enable attention at the deepest AE level.")
    parser.add_argument("--nonlocal-attention", action="store_true", help="Enable encoder/decoder non-local attention.")
    parser.add_argument("--encoder-nonlocal-attention", action="store_true", help="Enable encoder non-local attention.")
    parser.add_argument("--decoder-nonlocal-attention", action="store_true", help="Enable decoder non-local attention.")
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Checkpoint encoder/decoder activations.")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--val-every-steps", type=int, default=1000)
    parser.add_argument("--export-recon-every-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser


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
        path = self.volume_dir / f"{filename}.npy"
        volume = np.load(path).astype(np.float32, copy=False)
        if volume.ndim == 3:
            volume = volume[None]
        return {
            "image": torch.from_numpy(volume),
            "fn": filename,
            "site": torch.tensor(int(row["site"]), dtype=torch.long),
        }


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


def build_models(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    from monai.networks.layers import Act
    from generative.networks.nets import AutoencoderKL, PatchDiscriminator

    channels = tuple(int(v) for v in args.base_channels)
    encoder_nonlocal = bool(args.nonlocal_attention or args.encoder_nonlocal_attention)
    decoder_nonlocal = bool(args.nonlocal_attention or args.decoder_nonlocal_attention)
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=channels,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_res_blocks,
        norm_num_groups=8,
        attention_levels=tuple(bool(args.attention) and i == len(channels) - 1 for i in range(len(channels))),
        with_encoder_nonlocal_attn=encoder_nonlocal,
        with_decoder_nonlocal_attn=decoder_nonlocal,
        use_flash_attention=bool(args.use_flash_attention),
        use_checkpointing=bool(args.gradient_checkpointing),
    ).to(device)
    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
        kernel_size=4,
        activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm="BATCH",
        bias=False,
        padding=1,
    ).to(device)
    return autoencoder, discriminator


def amp_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16


def cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "cuda_mem=NA"
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    return (
        f"cuda_mem_alloc_gb={allocated:.2f} cuda_mem_reserved_gb={reserved:.2f} "
        f"cuda_mem_peak_alloc_gb={max_allocated:.2f} cuda_mem_peak_reserved_gb={max_reserved:.2f}"
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
    config: dict[str, Any],
    step: int,
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "scaler_g_state_dict": scaler_g.state_dict(),
            "scaler_d_state_dict": scaler_d.state_dict(),
            "config": config,
            "step": step,
            "epoch": epoch,
        },
        path,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 8,
) -> float:
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True).float()
        with autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            reconstruction, _z_mu, _z_sigma = model(images)
        losses.append(float(F.l1_loss(reconstruction.float(), images.float()).detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def export_reconstructions(
    out_dir: Path,
    step: int,
    images: torch.Tensor,
    reconstruction: torch.Tensor,
    batch: dict[str, Any],
) -> None:
    recon_dir = out_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        recon_dir / f"recon_step_{step:06d}.npz",
        raw=images.detach().float().cpu().numpy(),
        recon=reconstruction.detach().float().cpu().numpy(),
        fn=np.asarray(batch["fn"]),
        site=batch["site"].detach().cpu().numpy(),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    add_hcld_to_path(args.hcld_root)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available; HCLD AE training is intended for GPU.", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_ds = HcldVolumeDataset(args.data_dir, args.train_labels)
    val_ds = HcldVolumeDataset(args.data_dir, args.val_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model, discriminator = build_models(args, device)
    from generative.losses import PatchAdversarialLoss

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    optimizer_g = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.adv_lr)
    dtype = amp_dtype(args)
    scaler_enabled = device.type == "cuda" and args.amp_dtype == "fp16"
    scaler_g = GradScaler(device.type, enabled=scaler_enabled)
    scaler_d = GradScaler(device.type, enabled=scaler_enabled)

    start_step = 0
    start_epoch = 0
    if args.resume_checkpoint:
        ckpt = Path(args.resume_checkpoint)
        if ckpt.exists():
            payload = torch.load(ckpt, map_location=device)
            model.load_state_dict(payload["model_state_dict"])
            discriminator.load_state_dict(payload["discriminator_state_dict"])
            optimizer_g.load_state_dict(payload["optimizer_g_state_dict"])
            optimizer_d.load_state_dict(payload["optimizer_d_state_dict"])
            scaler_g.load_state_dict(payload["scaler_g_state_dict"])
            scaler_d.load_state_dict(payload["scaler_d_state_dict"])
            start_step = int(payload.get("step", 0))
            start_epoch = int(payload.get("epoch", 0))
            print(f"Resumed from {ckpt} at step={start_step} epoch={start_epoch}", flush=True)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["effective_encoder_nonlocal_attention"] = bool(args.nonlocal_attention or args.encoder_nonlocal_attention)
    config["effective_decoder_nonlocal_attention"] = bool(args.nonlocal_attention or args.decoder_nonlocal_attention)
    config["train_size"] = len(train_ds)
    config["val_size"] = len(val_ds)
    config["device"] = str(device)
    config["amp_torch_dtype"] = str(dtype)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, sort_keys=True), flush=True)

    step = start_step
    started = time.time()
    model.train()
    discriminator.train()
    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            step += 1
            images = batch["image"].to(device, non_blocking=True).float()

            optimizer_g.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                reconstruction, z_mu, z_sigma = model(images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                kl_loss = 0.5 * torch.sum(
                    z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2).clamp_min(1e-8)) - 1,
                    dim=[1, 2, 3, 4],
                )
                kl_loss = torch.mean(kl_loss)
                loss_g = recons_loss + args.kl_weight * kl_loss + args.adv_weight * generator_loss
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            optimizer_d.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = args.adv_weight * 0.5 * (loss_d_fake + loss_d_real)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            if not torch.isfinite(loss_g.detach()) or not torch.isfinite(loss_d.detach()):
                raise RuntimeError(f"Non-finite loss at step={step}: loss_g={loss_g.item()} loss_d={loss_d.item()}")

            if step == 1 or step % 25 == 0:
                elapsed = (time.time() - started) / 60.0
                print(
                    f"step={step} epoch={epoch} recon={float(recons_loss.detach().cpu()):.6f} "
                    f"kl={float(kl_loss.detach().cpu()):.6f} g_adv={float(generator_loss.detach().cpu()):.6f} "
                    f"d={float(loss_d.detach().cpu()):.6f} elapsed_min={elapsed:.1f} "
                    f"{cuda_memory_summary(device)}",
                    flush=True,
                )
            if step % args.val_every_steps == 0:
                val_l1 = validate(model, val_loader, device, dtype)
                print(f"validation step={step} val_l1={val_l1:.6f} {cuda_memory_summary(device)}", flush=True)
            if args.export_recon_every_steps > 0 and step % args.export_recon_every_steps == 0:
                export_reconstructions(out_dir, step, images, reconstruction, batch)
                print(f"exported_recon={out_dir / 'reconstructions' / f'recon_step_{step:06d}.npz'}", flush=True)
            if step % args.save_every_steps == 0:
                save_checkpoint(
                    ckpt_dir / f"model_step_{step:06d}.pt",
                    model,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    scaler_g,
                    scaler_d,
                    config,
                    step,
                    epoch,
                )
                save_checkpoint(
                    ckpt_dir / "model_latest.pt",
                    model,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    scaler_g,
                    scaler_d,
                    config,
                    step,
                    epoch,
                )
                print(f"saved={ckpt_dir / 'model_latest.pt'}", flush=True)
            if args.max_steps is not None and step >= args.max_steps:
                save_checkpoint(
                    ckpt_dir / "model_latest.pt",
                    model,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    scaler_g,
                    scaler_d,
                    config,
                    step,
                    epoch,
                )
                del images, reconstruction, z_mu, z_sigma, logits_fake, logits_real
                return

            del images, reconstruction, z_mu, z_sigma, logits_fake, logits_real

    save_checkpoint(
        ckpt_dir / "model_latest.pt",
        model,
        discriminator,
        optimizer_g,
        optimizer_d,
        scaler_g,
        scaler_d,
        config,
        step,
        args.epochs,
    )


if __name__ == "__main__":
    main()
