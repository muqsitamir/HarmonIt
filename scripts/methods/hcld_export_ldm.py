"""Export adapted HCLD cLDM harmonized test slices as a HarmonIt NPZ artifact."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from harmonit.data.abide_slices_dataset import (  # noqa: E402
    AbideSlicesDataset,
    bbox_from_mask,
    crop_with_bbox,
    make_head_mask,
    resize_to_hw,
)


METHOD = "adapted_hcld"


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
            "subject_id": str(row["subject_id"]),
            "site": torch.tensor(int(row["site"]), dtype=torch.long),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export adapted HCLD cLDM test harmonizations.")
    parser.add_argument("--hcld-root", default="/home/muqsitamir/repos/HCLD")
    parser.add_argument("--manifest-path", default="data/abide_manifest.csv")
    parser.add_argument("--splits-path", default="data/splits.json")
    parser.add_argument("--data-dir", default="outputs/hcld_abide")
    parser.add_argument("--out-dir", default="outputs/harmonized/adapted_hcld")
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--ldm-checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--target-site-id", type=int, default=5)
    parser.add_argument("--condition-labels", default="labels/train_tar.tsv")
    parser.add_argument("--out-hw", nargs=2, type=int, default=(256, 256))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-train-ddim", type=int, default=50)
    parser.add_argument("--num-inference-fdp", type=int, default=30)
    parser.add_argument("--num-inference-rdp", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-nonzero-frac", type=float, default=0.02)
    parser.add_argument("--fg-bbox-thr", type=float, default=0.02)
    parser.add_argument("--volume-cache-size", type=int, default=12)
    parser.add_argument(
        "--slice-index-map",
        default=None,
        help="Optional JSON mapping subject_id to evaluator-compatible fixed slice index.",
    )
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--no-qc", action="store_true")
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
    return autoencoder


def build_unet(device: torch.device) -> torch.nn.Module:
    from generative.networks.nets import DiffusionModelUNet

    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=12,
        out_channels=6,
        num_res_blocks=1,
        num_channels=(32, 64, 64),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 64),
        use_flash_attention=False,
    ).to(device)
    return unet


def hcld_dhw_to_original_xyz(volume_dhw: np.ndarray, original_shape_xyz: tuple[int, int, int]) -> np.ndarray:
    tensor = torch.from_numpy(volume_dhw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    target_dhw = (int(original_shape_xyz[2]), int(original_shape_xyz[0]), int(original_shape_xyz[1]))
    resized = F.interpolate(tensor, size=target_dhw, mode="trilinear", align_corners=False)[0, 0].cpu().numpy()
    return np.transpose(resized, (1, 2, 0)).astype(np.float32, copy=False)


def fixed_slice_from_volume(
    dataset: AbideSlicesDataset,
    row_idx: int,
    volume_xyz: np.ndarray,
    slice_idx: int,
) -> np.ndarray:
    sample = dataset.samples[row_idx]
    raw_volume = dataset._load_volume(sample)

    raw_slice_full = raw_volume[:, :, slice_idx]
    head_mask_full = make_head_mask(raw_slice_full, thr=dataset.head_mask_thr, dilate_iters=dataset.head_mask_dilate)
    bbox = bbox_from_mask(head_mask_full, margin=dataset.fg_bbox_margin)
    head_mask_pre = crop_with_bbox(head_mask_full.astype(np.uint8), bbox).astype(bool)

    image_slice = crop_with_bbox(volume_xyz[:, :, slice_idx], bbox)
    image_slice = image_slice.copy()
    image_slice[~head_mask_pre] = 0.0
    image_slice = resize_to_hw(image_slice, dataset.out_hw)
    return image_slice.astype(np.float32)


@torch.no_grad()
def harmonize_volume(
    image: torch.Tensor,
    condition: torch.Tensor,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    inferer: Any,
    scheduler_ddim: Any,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    image = image.to(device, non_blocking=True).float()
    condition = condition.to(device, non_blocking=True).float()
    with autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
        conditioning = autoencoder.encode_stage_2_inputs(condition) * inferer.scale_factor
        z_x = autoencoder.encode_stage_2_inputs(image) * inferer.scale_factor
        scheduler_ddim.set_timesteps(num_inference_steps=args.num_inference_fdp)
        _img_noisy, latent_noisy = inferer.reverse_sample(
            input_noise=z_x,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler_ddim,
            conditioning=conditioning,
            mode="concat",
            verbose=False,
            return_latent=True,
        )
        scheduler_ddim.set_timesteps(num_inference_steps=args.num_inference_rdp)
        recon = inferer.sample(
            input_noise=latent_noisy,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler_ddim,
            conditioning=conditioning,
            mode="concat",
            verbose=False,
        )
    recon = recon.float().clamp_min(0)
    lo = recon.amin(dim=(2, 3, 4), keepdim=True)
    hi = recon.amax(dim=(2, 3, 4), keepdim=True)
    recon = (recon - lo) / (hi - lo).clamp_min(1e-6)
    return recon.detach().cpu()


def save_qc(path: Path, raw_images: np.ndarray, harmonized_images: np.ndarray, labels: list[str], n: int = 6) -> None:
    import matplotlib.pyplot as plt

    n = min(n, raw_images.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 3.6), constrained_layout=True)
    for idx in range(n):
        axes[0, idx].imshow(raw_images[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, idx].set_title(labels[idx], fontsize=7)
        axes[1, idx].imshow(harmonized_images[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, idx].set_axis_off()
        axes[1, idx].set_axis_off()
    axes[0, 0].set_ylabel("raw", fontsize=8)
    axes[1, 0].set_ylabel(METHOD, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    add_hcld_to_path(args.hcld_root)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(args)

    from generative.inferers import LatentDiffusionInferer
    from generative.networks.schedulers import DDIMScheduler

    autoencoder = build_autoencoder(args, device)
    unet = build_unet(device)
    ldm_payload = torch.load(args.ldm_checkpoint, map_location=device)
    unet.load_state_dict(ldm_payload["unet_state_dict"])
    unet.eval()
    scale_factor = float(ldm_payload.get("scale_factor", ldm_payload.get("config", {}).get("scale_factor", 1.0)))

    scheduler_ddim = DDIMScheduler(
        num_train_timesteps=args.num_train_ddim,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    inferer = LatentDiffusionInferer(scheduler_ddim, scale_factor=scale_factor, Adain="AdaIN_reverse")

    raw_dataset = AbideSlicesDataset(
        manifest_path=args.manifest_path,
        splits_path=args.splits_path,
        split=args.split,
        out_hw=tuple(args.out_hw),
        slice_mode="fixed",
        seed=args.seed,
        valid_nonzero_frac=args.valid_nonzero_frac,
        fg_bbox_thr=args.fg_bbox_thr,
        volume_cache_size=args.volume_cache_size,
        mask_mode="none",
        bg_suppress=True,
        input_mode="image",
    )
    raw_dataset.aug_affine = False

    hcld_all = HcldVolumeDataset(args.data_dir, f"labels/{args.split}.tsv")
    all_by_subject = {str(row.subject_id): idx for idx, row in hcld_all.labels.iterrows()}
    cond_ds = HcldVolumeDataset(args.data_dir, args.condition_labels)
    cond_loader = DataLoader(cond_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    cond_iter = cycle(cond_loader)

    output_dir = Path(args.out_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_index_map: dict[str, int] = {}
    if args.slice_index_map:
        slice_index_map = {str(k): int(v) for k, v in json.loads(Path(args.slice_index_map).read_text()).items()}

    raw_images: list[np.ndarray] = []
    harmonized_images: list[np.ndarray] = []
    subject_ids: list[str] = []
    site_ids: list[int] = []
    slice_indices: list[int] = []

    manifest_rows: list[dict[str, object]] = []
    for row_idx, sample in enumerate(raw_dataset.samples):
        subject_id = str(sample.subject_id)
        site_id = int(sample.site_id)
        if subject_id in slice_index_map:
            slice_idx = int(slice_index_map[subject_id])
            raw_volume = raw_dataset._load_volume(sample)
            raw_slice = fixed_slice_from_volume(raw_dataset, row_idx, volume_xyz=raw_volume, slice_idx=slice_idx)
        else:
            raw_item = raw_dataset[row_idx]
            raw_slice = raw_item[0].squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            site_id = int(raw_item[1])
            subject_id = str(raw_item[2])
            slice_idx = int(raw_item[3])
        raw_images.append(raw_slice[None])
        subject_ids.append(subject_id)
        site_ids.append(site_id)
        slice_indices.append(slice_idx)

        if site_id == args.target_site_id:
            harm_slice = raw_slice
        else:
            hcld_idx = all_by_subject[subject_id]
            source = hcld_all[hcld_idx]
            cond = next(cond_iter)
            recon = harmonize_volume(
                image=source["image"].unsqueeze(0),
                condition=cond["image"],
                autoencoder=autoencoder,
                unet=unet,
                inferer=inferer,
                scheduler_ddim=scheduler_ddim,
                args=args,
                device=device,
                dtype=dtype,
            )
            hcld_volume = recon[0, 0].numpy().astype(np.float32)
            raw_volume = raw_dataset._load_volume(sample)
            recon_xyz = hcld_dhw_to_original_xyz(hcld_volume, raw_volume.shape)
            harm_slice = fixed_slice_from_volume(raw_dataset, row_idx, volume_xyz=recon_xyz, slice_idx=slice_idx)

        harmonized_images.append(np.clip(harm_slice, 0.0, 1.0)[None].astype(np.float32))
        manifest_rows.append(
            {
                "row": row_idx,
                "subject_id": subject_id,
                "site_id": site_id,
                "slice_idx": slice_idx,
                "identity_target": bool(site_id == args.target_site_id),
            }
        )
        if (row_idx + 1) % 10 == 0 or row_idx + 1 == len(raw_dataset):
            print(f"exported={row_idx + 1}/{len(raw_dataset)} subject={subject_id}", flush=True)

    raw_array = np.stack(raw_images).astype(np.float32)
    harm_array = np.stack(harmonized_images).astype(np.float32)
    subject_array = np.asarray(subject_ids, dtype=str)
    site_array = np.asarray(site_ids, dtype=np.int64)
    slice_array = np.asarray(slice_indices, dtype=np.int64)

    npz_path = output_dir / f"{METHOD}_slices.npz"
    np.savez_compressed(
        npz_path,
        images=harm_array,
        raw_images=raw_array,
        subject_ids=subject_array,
        site_ids=site_array,
        slice_indices=slice_array,
        split=np.asarray(args.split),
        method=np.asarray(METHOD),
    )
    with (output_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row", "subject_id", "site_id", "slice_idx", "identity_target"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "method": METHOD,
        "split": args.split,
        "npz_path": str(npz_path),
        "ae_checkpoint": args.ae_checkpoint,
        "ldm_checkpoint": args.ldm_checkpoint,
        "scale_factor": scale_factor,
        "num_inference_fdp": args.num_inference_fdp,
        "num_inference_rdp": args.num_inference_rdp,
        "target_site_id": args.target_site_id,
        "valid_nonzero_frac": args.valid_nonzero_frac,
        "fg_bbox_thr": args.fg_bbox_thr,
        "slice_index_map": args.slice_index_map,
        "n_subjects": len(subject_ids),
    }
    (output_dir / "export_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.no_qc:
        save_qc(
            output_dir / f"qc_raw_vs_{METHOD}.png",
            raw_array,
            harm_array,
            [f"{sid} s{sid_site}" for sid, sid_site in zip(subject_ids, site_ids)],
        )
    print(f"saved_npz={npz_path}", flush=True)


if __name__ == "__main__":
    main()
