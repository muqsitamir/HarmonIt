"""Cache normalized ABIDE volumes on local disk for site-probe training (protocol amendment 6).

Volumes come from AbideSlicesDataset's own loader and are stored as float32 [D, H, W], so the
dataset reads one contiguous block per axial slice (VOLUME_CACHE_DIR). --verify compares the
slices returned with and without the cache, in fixed and random slice mode, and fails on any
difference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harmonit.data.abide_slices_dataset import AbideSlicesDataset  # noqa: E402

ARGS = None


def make_dataset(split, slice_mode, seed=0):
    return AbideSlicesDataset(
        manifest_path=ARGS.manifest, splits_path=ARGS.splits, split=split, out_hw=(256, 256),
        slice_mode=slice_mode, valid_nonzero_frac=0.02, fg_bbox_thr=0.02, seed=seed,
        volume_cache_size=2, mask_mode="none", bg_suppress=True, head_mask_thr=0.02,
        head_mask_dilate=3, input_mode="image")


def write_one(job):
    split, index = job
    os.environ.pop("VOLUME_CACHE_DIR", None)
    ds = make_dataset(split, "fixed")
    sample = ds.samples[index]
    target = Path(ARGS.out_dir) / f"{sample.subject_id}.npy"
    if not target.exists():
        vol = np.ascontiguousarray(ds._load_volume(sample).transpose(2, 0, 1), dtype=np.float32)
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, vol)
        tmp.rename(target)
    return sample.subject_id


def verify(split, count):
    os.environ.pop("VOLUME_CACHE_DIR", None)
    fresh = [make_dataset(split, mode, seed=7) for mode in ("fixed", "random")]
    os.environ["VOLUME_CACHE_DIR"] = ARGS.out_dir
    cached = [make_dataset(split, mode, seed=7) for mode in ("fixed", "random")]
    os.environ.pop("VOLUME_CACHE_DIR")
    indices = np.linspace(0, len(fresh[0]) - 1, count).astype(int)
    for a, b in zip(fresh, cached):
        for i in indices:
            xa, ya, sa, ka = a[i]
            xb, yb, sb, kb = b[i]
            if (ya, sa, ka) != (yb, sb, kb) or not np.array_equal(xa.numpy(), xb.numpy()):
                raise SystemExit(f"cache mismatch: {split} {a.slice_mode} {sa} slice {ka} vs {kb}")
    return len(indices) * 2


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits-to-cache", nargs="+", default=["train", "val"])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--verify", type=int, default=40, help="subjects per split to compare")
    ARGS = p.parse_args()
    Path(ARGS.out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(split, i) for split in ARGS.splits_to_cache for i in range(len(make_dataset(split, "fixed")))]
    with Pool(ARGS.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(write_one, jobs), start=1):
            if n % 100 == 0 or n == len(jobs):
                print(f"cached {n}/{len(jobs)}", flush=True)
    checked = {split: verify(split, ARGS.verify) for split in ARGS.splits_to_cache}
    report = dict(n_volumes=len(jobs), verified_items=checked, out_dir=ARGS.out_dir)
    (Path(ARGS.out_dir) / "cache_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
