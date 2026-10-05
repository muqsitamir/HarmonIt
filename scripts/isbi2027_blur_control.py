"""Synthetic destroyed control for the frozen-probe audit (protocol amendment 12).

Blurs every raw test slice of an existing test artifact with a 2D Gaussian and writes one
artifact per sigma in the evaluator's NPZ format (images, raw_images, ids, slices, metadata).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-artifact", required=True, help="Any test artifact carrying raw_images")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sigmas", type=float, nargs="+", default=[2, 4, 8], help="Gaussian sigma in pixels")
    args = p.parse_args()
    with np.load(args.source_artifact, allow_pickle=False) as data:
        raw = data["raw_images"].astype(np.float32)
        meta = {k: data[k] for k in ("subject_ids", "site_ids", "slice_indices")}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sigma in args.sigmas:
        blurred = np.stack([gaussian_filter(s[0], sigma=sigma, mode="constant", cval=0.0)[None] for s in raw])
        name = f"gaussian_blur_s{sigma:g}"
        path = out / f"{name}.npz"
        np.savez(path, images=blurred.astype(np.float32), raw_images=raw, split=np.array("test"),
                 method=np.array(name), **meta)
        print(path, float(np.abs(blurred - raw).mean()))


if __name__ == "__main__":
    main()
