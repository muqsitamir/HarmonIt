"""Replace images in an export NPZ with filled binary head silhouettes of its raw slices.

Silhouette = raw slice > threshold with interior holes filled. raw_images, identifiers
and split are kept, so the result can be used for slice-probe training and evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.02)
    args = p.parse_args()
    if Path(args.out).exists():
        p.error("Output exists")
    with np.load(args.npz, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    raw = arrays["raw_images"]
    silhouettes = np.stack([binary_fill_holes(r[0] > args.threshold)[None] for r in raw]).astype(np.float32)
    arrays["images"] = silhouettes
    arrays["method"] = np.asarray("silhouette")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"{args.out}: {len(silhouettes)} silhouettes, mean foreground {silhouettes.mean():.3f}")


if __name__ == "__main__":
    main()
