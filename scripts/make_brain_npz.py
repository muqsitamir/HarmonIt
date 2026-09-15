"""Brain-only slice exports from HD-BET masks (protocol amendment 7).

For every fixed slice in the exports, the raw volume's HD-BET mask is reoriented like the
image, cut at the same slice, cropped with the same head bounding box and resized with the
dataset's nearest-neighbour mask resizer. Gate: the recomputed raw slice must equal the
export's raw slice. Writes per split:
  masks/<split>/brain_masks.npz                   masks [N,1,256,256], brain-to-head area ratio
  <method>/<split>/<npz>                          images and raw_images multiplied by the mask
  brain_shape/<split>/brain_shape_slices.npz      images = binary mask, raw_images unchanged
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harmonit.data.abide_slices_dataset import (  # noqa: E402
    AbideSlicesDataset, bbox_from_mask, crop_with_bbox, make_head_mask, resize_mask_to_hw, resize_to_hw)

NPZ = {"histogram_matching": "histogram_matching_slices.npz", "cyclegan_tuned": "cyclegan_nyu_slices.npz",
       "diffusion_20k": "diffusion_img2img_nyu_slices.npz"}
ARGS = None


def one_subject(job):
    split, subject, k, raw_export = job
    if split == "test" or not ARGS.volume_cache:
        os.environ.pop("VOLUME_CACHE_DIR", None)
    else:
        os.environ["VOLUME_CACHE_DIR"] = ARGS.volume_cache
    ds = AbideSlicesDataset(manifest_path=ARGS.manifest, splits_path=ARGS.splits, split=split,
                            slice_mode="fixed", valid_nonzero_frac=0.02, fg_bbox_thr=0.02, volume_cache_size=1)
    sample = next(s for s in ds.samples if s.subject_id == subject)
    vol = ds._load_volume(sample)
    sl_full = vol[:, :, k]
    head_full = make_head_mask(sl_full, thr=0.02, dilate_iters=3)
    bbox = bbox_from_mask(head_full, margin=20)
    head = crop_with_bbox(head_full.astype(np.uint8), bbox).astype(bool)
    sl = np.array(crop_with_bbox(sl_full, bbox), dtype=np.float32)
    sl[~head] = 0.0
    raw = resize_to_hw(sl, (256, 256))
    diff = float(np.abs(raw - raw_export[0]).max())
    mask_img = nib.as_closest_canonical(nib.load(str(Path(ARGS.mask_dir) / f"{subject}.nii.gz")))
    mask_vol = np.asarray(mask_img.dataobj) > 0.5
    if mask_vol.shape != vol.shape:
        raise ValueError(f"{subject}: mask shape {mask_vol.shape} != volume {vol.shape}")
    brain = crop_with_bbox(mask_vol[:, :, k], bbox) & head
    ratio = float(brain.sum() / max(1, head.sum()))
    return subject, resize_mask_to_hw(brain, (256, 256))[None], ratio, diff


def save(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--volume-cache", help="VOLUME_CACHE_DIR for train/val volumes")
    p.add_argument("--workers", type=int, default=6)
    ARGS = p.parse_args()
    exports, out = Path(ARGS.exports), Path(ARGS.out_dir)
    report = {}
    for split in ("test", "val", "train"):
        with np.load(exports / "histogram_matching" / split / NPZ["histogram_matching"], allow_pickle=False) as data:
            ref = {key: data[key] for key in data.files}
        subjects = ref["subject_ids"].astype(str)
        jobs = [(split, s, int(k), r) for s, k, r in zip(subjects, ref["slice_indices"], ref["raw_images"])]
        with Pool(ARGS.workers) as pool:
            results = pool.map(one_subject, jobs, chunksize=4)
        if [r[0] for r in results] != subjects.tolist():
            raise RuntimeError("subject order changed")
        masks = np.stack([r[1] for r in results])
        ratios = np.array([r[2] for r in results])
        diffs = np.array([r[3] for r in results])
        if diffs.max() > 1e-6:
            raise SystemExit(f"{split}: recomputed raw slices differ from exports (max {diffs.max():.2e})")
        if not masks.reshape(len(masks), -1).any(axis=1).all():
            raise SystemExit(f"{split}: empty brain mask on a fixed slice")
        common = {k: ref[k] for k in ("subject_ids", "site_ids", "slice_indices", "split")}
        save(out / "masks" / split / "brain_masks.npz", dict(common, masks=masks, brain_to_head_ratio=ratios))
        m = masks.astype(np.float32)
        save(out / "brain_shape" / split / "brain_shape_slices.npz",
             dict(common, images=m, raw_images=ref["raw_images"], method=np.asarray("brain_shape")))
        if split != "test":
            for method, name in NPZ.items():
                with np.load(exports / method / split / name, allow_pickle=False) as data:
                    arrays = {key: data[key] for key in data.files}
                if not np.array_equal(arrays["subject_ids"].astype(str), subjects):
                    raise ValueError(f"{method}/{split}: subject order differs")
                arrays["images"] = arrays["images"] * m
                arrays["raw_images"] = arrays["raw_images"] * m
                save(out / method / split / name, arrays)
        low = [(s, round(float(r), 3)) for s, r in zip(subjects, ratios) if r < 0.10]
        report[split] = dict(n=len(subjects), max_raw_abs_diff=float(diffs.max()),
                             ratio_quantiles=np.quantile(ratios, [0, .05, .5, .95, 1]).round(3).tolist(),
                             ratio_below_0_10=low)
        print(split, json.dumps(report[split]), flush=True)
    (out / "brain_mask_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
