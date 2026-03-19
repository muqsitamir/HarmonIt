from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset


def robust_normalize(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip p1-p99 then z-score using nonzero voxels (fallback to all voxels)."""
    v = vol.astype(np.float32)
    mask = v > 0
    vals = v[mask] if mask.any() else v.reshape(-1)

    p1, p99 = np.percentile(vals, [1, 99])
    v = np.clip(v, p1, p99)

    vals2 = v[mask] if mask.any() else v.reshape(-1)
    mu, sigma = float(vals2.mean()), float(vals2.std())
    v = (v - mu) / (sigma + eps)
    return v


def center_crop_or_pad(img2d: np.ndarray, out_hw: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Center crop/pad a 2D array to out_hw."""
    h, w = img2d.shape
    oh, ow = out_hw
    out = np.zeros((oh, ow), dtype=img2d.dtype)

    y0 = max(0, (h - oh) // 2)
    x0 = max(0, (w - ow) // 2)
    y1 = min(h, y0 + oh)
    x1 = min(w, x0 + ow)

    cropped = img2d[y0:y1, x0:x1]

    py0 = max(0, (oh - cropped.shape[0]) // 2)
    px0 = max(0, (ow - cropped.shape[1]) // 2)
    out[py0:py0 + cropped.shape[0], px0:px0 + cropped.shape[1]] = cropped
    return out


@dataclass
class AbideSample:
    subject_id: str
    site: str
    site_id: int
    t1_path: str


class AbideSlicesDataset(Dataset):
    """
    ABIDE 2D axial slice dataset.

    Modes:
      - slice_mode="random": length = num subjects; each __getitem__ returns a random valid slice from that subject.
      - slice_mode="fixed": deterministic slice selection per subject (middle slice among valid ones).

    Returns:
      image: torch.FloatTensor [1, H, W]
      site_id: int
      subject_id: str
      slice_idx: int
    """

    def __init__(
        self,
        manifest_path: str = "data/abide_manifest.csv",
        splits_path: str = "data/splits.json",
        split: str = "train",
        out_hw: Tuple[int, int] = (256, 256),
        slice_mode: str = "random",
        valid_nonzero_frac: float = 0.10,
        seed: int = 42,
        volume_cache_size: int = 12,
    ):
        self.manifest_path = Path(manifest_path)
        self.splits_path = Path(splits_path)
        self.split = split
        self.out_hw = out_hw
        self.slice_mode = slice_mode
        self.valid_nonzero_frac = valid_nonzero_frac
        self.rng = np.random.RandomState(seed)

        df = pd.read_csv(self.manifest_path)

        # build site->id mapping deterministically if not present
        if "site_id" not in df.columns:
            sites = sorted(df["site"].unique())
            site_map = {s: i for i, s in enumerate(sites)}
            df["site_id"] = df["site"].map(site_map).astype(int)

        # load split subject IDs
        split_obj = json.loads(self.splits_path.read_text())
        split_ids = set(split_obj[split])

        df = df[df["subject_id"].isin(split_ids)].copy()
        df = df.sort_values(["site", "subject_id"]).reset_index(drop=True)

        self.samples: List[AbideSample] = [
            AbideSample(
                subject_id=row["subject_id"],
                site=row["site"],
                site_id=int(row["site_id"]),
                t1_path=row["t1_path"],
            )
            for _, row in df.iterrows()
        ]

        # cache valid slice indices per subject to avoid recomputation
        self._valid_slices: Dict[str, List[int]] = {}

        # LRU cache for normalized volumes to avoid repeated NIfTI load + normalization on CPU
        self._vol_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.volume_cache_size = volume_cache_size

    def __len__(self) -> int:
        return len(self.samples)

    def _load_volume(self, sample: AbideSample) -> np.ndarray:
        sid = sample.subject_id

        # Cache hit: refresh LRU order
        if sid in self._vol_cache:
            vol_n = self._vol_cache.pop(sid)
            self._vol_cache[sid] = vol_n
            return vol_n

        # Cache miss: load and normalize
        img = nib.load(sample.t1_path)
        img = nib.as_closest_canonical(img)
        vol = img.get_fdata(dtype=np.float32)
        vol_n = robust_normalize(vol)

        # Insert and evict LRU
        self._vol_cache[sid] = vol_n
        if len(self._vol_cache) > self.volume_cache_size:
            self._vol_cache.popitem(last=False)

        return vol_n

    def _compute_valid_slices(self, sample: AbideSample, vol_n: np.ndarray) -> List[int]:
        if sample.subject_id in self._valid_slices:
            return self._valid_slices[sample.subject_id]

        D = vol_n.shape[2]

        # Only consider middle slices (avoid empty top/bottom)
        z_min = int(0.15 * D)
        z_max = int(0.85 * D)
        z_min = max(0, min(z_min, D - 1))
        z_max = max(z_min + 1, min(z_max, D))

        valid = []
        thr = 0.05  # threshold after z-score; filters background/noise
        for k in range(z_min, z_max):
            sl = vol_n[:, :, k]
            fg = float((np.abs(sl) > thr).mean())
            if fg >= self.valid_nonzero_frac:
                valid.append(k)

        # fallback if strict filtering wipes everything
        if len(valid) == 0:
            mid = D // 2
            valid = list(range(max(0, mid - 10), min(D, mid + 10)))

        self._valid_slices[sample.subject_id] = valid
        return valid

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        vol_n = self._load_volume(sample)
        valid = self._compute_valid_slices(sample, vol_n)

        if self.slice_mode == "random":
            k = int(self.rng.choice(valid))
        elif self.slice_mode == "fixed":
            k = int(valid[len(valid) // 2])
        else:
            raise ValueError(f"Unknown slice_mode={self.slice_mode}")

        sl = vol_n[:, :, k]
        sl = center_crop_or_pad(sl, self.out_hw)

        # return as [1, H, W]
        img_t = torch.from_numpy(sl).float().unsqueeze(0)
        return img_t, sample.site_id, sample.subject_id, k