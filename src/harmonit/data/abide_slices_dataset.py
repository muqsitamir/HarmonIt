from __future__ import annotations

import os
import json
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import torchvision.transforms.functional as TF
except Exception:  # pragma: no cover
    TF = None

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover
    distance_transform_edt = None

def robust_normalize(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robust 0..1 normalization with background preserved.

    - Treat voxels with value 0 as background and keep them exactly 0.
    - Clip foreground (v>0) to [p1, p99] to reduce outliers.
    - Min-max scale foreground to (0,1].

    This yields a stable intensity range across sites while maintaining the
    interpretation that background equals the minimum value (0).
    """
    v = vol.astype(np.float32)

    bg = v == 0
    fg = ~bg

    if fg.any():
        vals = v[fg]
        p1, p99 = np.percentile(vals, [1, 99])
        # Guard against degenerate percentiles
        if p99 <= p1 + eps:
            p1 = float(vals.min())
            p99 = float(vals.max())

        v_fg = np.clip(v[fg], p1, p99)
        denom = (p99 - p1) + eps
        v_fg = (v_fg - p1) / denom
        # Ensure foreground stays within [0,1]
        v_fg = np.clip(v_fg, 0.0, 1.0)
        v[fg] = v_fg

    # Keep true background at exactly 0
    v[bg] = 0.0
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


def crop_to_foreground_bbox(img2d: np.ndarray, thr: float = 0.02, margin: int = 10) -> np.ndarray:
    """Crop a 2D slice to the bounding box of foreground pixels (cheap brain proxy).

    Foreground is defined as |img| > thr (after 0..1 normalization). If no foreground is found,
    returns the original image.
    """
    m = np.abs(img2d) > thr
    if not m.any():
        return img2d

    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(img2d.shape[0] - 1, y1 + margin)
    x1 = min(img2d.shape[1] - 1, x1 + margin)

    return img2d[y0 : y1 + 1, x0 : x1 + 1]


def resize_to_hw(img2d: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D numpy array to (H,W) using torch bilinear interpolation."""
    oh, ow = out_hw
    t = torch.from_numpy(img2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    t = F.interpolate(t, size=(oh, ow), mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)


def resize_mask_to_hw(mask2d: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to (H,W) using nearest-neighbor."""
    oh, ow = out_hw
    t = torch.from_numpy(mask2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    t = F.interpolate(t, size=(oh, ow), mode="nearest")
    return (t[0, 0].cpu().numpy() > 0.5)


def binary_dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(iters):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask using flood-fill from the border."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    q = deque()

    # enqueue all background border pixels
    for x in range(w):
        if not mask[0, x]:
            q.append((0, x))
        if not mask[h - 1, x]:
            q.append((h - 1, x))
    for y in range(h):
        if not mask[y, 0]:
            q.append((y, 0))
        if not mask[y, w - 1]:
            q.append((y, w - 1))

    while q:
        y, x = q.popleft()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if visited[y, x] or mask[y, x]:
            continue
        visited[y, x] = True
        q.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])

    holes = (~mask) & (~visited)
    return mask | holes


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return largest 4-connected component of a binary mask."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = np.zeros_like(mask, dtype=bool)
    best_size = 0

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            q = deque([(y, x)])
            comp = []
            visited[y, x] = True

            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))

            if len(comp) > best_size:
                best_size = len(comp)
                best[:] = False
                for cy, cx in comp:
                    best[cy, cx] = True

    return best

def make_head_mask(img2d: np.ndarray, thr: float = 0.02, dilate_iters: int = 3) -> np.ndarray:
    """Conservative head/foreground mask.

    1) Threshold |img| on the normalized slice.
    2) Keep the largest connected component.
    3) Fill holes.
    4) Dilate slightly for safety (avoid cutting cortex edges).
    """
    mask = np.abs(img2d) > thr
    if not mask.any():
        return np.ones_like(img2d, dtype=bool)

    mask = largest_connected_component(mask)
    mask = fill_holes(mask)
    mask = binary_dilate(mask, iters=dilate_iters)
    return mask

# --- Helper functions for bbox cropping of head mask ---
def bbox_from_mask(mask: np.ndarray, margin: int = 20) -> Tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) bbox from a binary mask, expanded by margin."""
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        h, w = mask.shape
        return 0, h, 0, w

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(mask.shape[0], y1 + margin + 1)
    x1 = min(mask.shape[1], x1 + margin + 1)
    return y0, y1, x0, x1


def crop_with_bbox(img2d: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return img2d[y0:y1, x0:x1]


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
        slice_percentiles: Tuple[float, ...] = (0.40, 0.50, 0.60),
        fg_bbox_thr: float = 0.02,
        fg_bbox_margin: int = 20,
        seed: int = 42,
        volume_cache_size: int = 12,
        mask_mode: str = "none",  # none | bg_only | brain_only
        label_permutation: Optional[list] = None,
        bg_suppress: bool = True,
        head_mask_thr: float = 0.02,
        head_mask_dilate: int = 3,
        input_mode: str = "image",  # image | mask_only
    ):
        self.manifest_path = Path(manifest_path)
        self.splits_path = Path(splits_path)
        self.split = split
        self.out_hw = out_hw
        self.slice_mode = slice_mode
        self.valid_nonzero_frac = valid_nonzero_frac
        self.slice_percentiles = slice_percentiles
        self.fg_bbox_thr = fg_bbox_thr
        self.fg_bbox_margin = fg_bbox_margin
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

        self.mask_mode = mask_mode
        self._label_perm = label_permutation

        self.bg_suppress = bg_suppress
        self.head_mask_thr = head_mask_thr
        self.head_mask_dilate = head_mask_dilate
        self.input_mode = str(input_mode)

        # For mask-only diagnostics, optionally use a smooth distance-transform instead of a hard binary mask.
        # Values: "binary" (default) or "dist".
        self.mask_only_repr = str(os.getenv("MASK_ONLY_REPR", "binary"))

        # --- Optional geometry-invariance augmentation (train only) ---
        self.aug_affine = (os.getenv("AUG_AFFINE", "0") == "1")
        self.aug_prob = float(os.getenv("AUG_PROB", "1.0"))
        self.aug_rot_deg = float(os.getenv("AUG_ROT_DEG", "7"))
        self.aug_trans_px = int(os.getenv("AUG_TRANS_PX", "16"))
        self.aug_scale_jitter = float(os.getenv("AUG_SCALE_JITTER", "0.10"))

    def __len__(self) -> int:
        return len(self.samples)

    def set_label_permutation(self, perm: list) -> None:
        """Apply a permutation to site_id labels (used for label-shuffle sanity checks)."""
        self._label_perm = perm

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
        thr = float(self.fg_bbox_thr)  # threshold in 0..1 space; filters background/noise
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

    def _maybe_apply_affine(self, img_t: torch.Tensor) -> torch.Tensor:
        """Apply random affine augmentation to a [1,H,W] tensor (train only).

        This aims to reduce site shortcut learning from geometry/alignment while
        preserving intensity-based scanner signature.
        """
        if not self.aug_affine or self.split != "train":
            return img_t
        if self.aug_prob < 1.0 and float(self.rng.rand()) > self.aug_prob:
            return img_t

        # Sample params
        angle = float(self.rng.uniform(-self.aug_rot_deg, self.aug_rot_deg))
        tx = int(self.rng.randint(-self.aug_trans_px, self.aug_trans_px + 1))
        ty = int(self.rng.randint(-self.aug_trans_px, self.aug_trans_px + 1))
        scale = float(self.rng.uniform(1.0 - self.aug_scale_jitter, 1.0 + self.aug_scale_jitter))

        if TF is not None:
            # torchvision expects [C,H,W]
            return TF.affine(
                img_t,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.0,
                center=None,
            )

        # Fallback: no torchvision
        return img_t

    def _pick_slice_index(self, vol_n: np.ndarray, valid: List[int]) -> int:
        """Pick a slice index using fixed depth percentiles but prefer the slice(s)
        with the largest foreground area (cheap mid-brain proxy).

        This reduces slice-level/coverage shortcuts for the site probe.
        """
        D = int(vol_n.shape[2])
        if len(valid) == 0:
            return D // 2

        targets = [int(p * (D - 1)) for p in self.slice_percentiles]
        base = []
        for t in targets:
            k = min(valid, key=lambda vv: abs(vv - t))
            base.append(int(k))

        valid_sorted = sorted(valid)
        idx_map = {k: i for i, k in enumerate(valid_sorted)}
        cand = []
        for k in base:
            i = idx_map.get(k, None)
            if i is None:
                continue
            for j in range(max(0, i - 3), min(len(valid_sorted), i + 4)):
                cand.append(int(valid_sorted[j]))

        cand = list(dict.fromkeys(cand))
        if len(cand) == 0:
            cand = valid_sorted

        thr = float(self.fg_bbox_thr)
        scores = []
        for k in cand:
            sl = vol_n[:, :, k]
            scores.append(float(np.mean(np.abs(sl) > thr)))

        order = np.argsort(-np.asarray(scores))
        cand_sorted = [cand[i] for i in order]

        if self.slice_mode == "fixed":
            return int(cand_sorted[0])

        topk = min(3, len(cand_sorted))
        return int(self.rng.choice(cand_sorted[:topk]))

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        vol_n = self._load_volume(sample)
        valid = self._compute_valid_slices(sample, vol_n)

        k = self._pick_slice_index(vol_n, valid)

        sl_full = vol_n[:, :, k]

        # Compute head/foreground mask on the FULL slice (before any crop)
        head_mask_full = make_head_mask(sl_full, thr=self.head_mask_thr, dilate_iters=self.head_mask_dilate)

        # Crop slice and mask together using the head-mask bbox (+ margin)
        bbox = bbox_from_mask(head_mask_full, margin=self.fg_bbox_margin)

        sl = crop_with_bbox(sl_full, bbox)
        head_mask_pre = crop_with_bbox(head_mask_full.astype(np.uint8), bbox).astype(bool)

        # Input-mode diagnostics
        if self.input_mode == "mask_only":
            # Use only the binary mask as input (tests whether mask geometry alone predicts site)
            m = resize_mask_to_hw(head_mask_pre, self.out_hw).astype(np.float32)
            if self.mask_only_repr == "dist" and distance_transform_edt is not None:
                dt = distance_transform_edt(m > 0)
                if dt.max() > 0:
                    dt = dt / (dt.max() + 1e-6)
                m = dt.astype(np.float32)
            img_t = torch.from_numpy(m).float().unsqueeze(0)
            img_t = self._maybe_apply_affine(img_t)

            site_id = sample.site_id
            if self._label_perm is not None:
                site_id = int(self._label_perm[int(site_id)])
            return img_t, site_id, sample.subject_id, k
        elif self.input_mode != "image":
            raise ValueError(f"Unknown input_mode={self.input_mode}")

        # Default: suppress background BEFORE resize so it cannot carry site signal
        if self.bg_suppress:
            sl = sl.copy()
            sl[~head_mask_pre] = 0.0

        # Apply diagnostic masking ablations BEFORE resize (avoid interpolation leakage)
        if self.mask_mode == "bg_only":
            out = np.zeros_like(sl, dtype=np.float32)
            out[~head_mask_pre] = sl[~head_mask_pre]
            sl = out
        elif self.mask_mode == "brain_only":
            out = np.zeros_like(sl, dtype=np.float32)
            out[head_mask_pre] = sl[head_mask_pre]
            sl = out
        elif self.mask_mode != "none":
            raise ValueError(f"Unknown mask_mode={self.mask_mode}")

        # Resize final slice to model input size
        sl = resize_to_hw(sl, self.out_hw)

        # return as [1, H, W]
        img_t = torch.from_numpy(sl).float().unsqueeze(0)
        img_t = self._maybe_apply_affine(img_t)

        site_id = sample.site_id
        if self._label_perm is not None:
            site_id = int(self._label_perm[int(site_id)])

        return img_t, site_id, sample.subject_id, k