"""Extra Fig. 2 panels: a matched NYU test subject, the Gaussian-blur control (amendment 12) and
HACA3 (amendment 13, from its test artifact).

Rendered like the original panels in `paper/isbi2027/figures/qualitative_panels/`: the input's
foreground 99.5th-percentile display window, a 64-pixel zoom box at the foreground centroid,
difference maps on the blue-grey-red scale at +-0.3, nearest-neighbour upsampling to the
original panel size. The NYU subject maximizes head-mask Dice with the input times correlation
inside the zoom box: Dice keeps the input's in-plane orientation, which varies between subjects,
and the box correlation picks matching central anatomy (slice level), which Dice alone does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import gaussian_filter

DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#a8322f"])
TARGET_SITE, LIMIT, ZOOM = 5, .3, 64


def render(values, cmap, vmin, vmax, size):
    rgb = (matplotlib.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap)(
        np.clip((values - vmin) / (vmax - vmin), 0, 1))[..., :3]
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize(size, Image.NEAREST)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", required=True, help="Any test artifact with raw_images")
    p.add_argument("--subject", default="UM_50428", help="Subject shown in Fig. 2")
    p.add_argument("--sigma", type=float, default=2)
    p.add_argument("--haca3", help="HACA3 test artifact (amendment 13)")
    p.add_argument("--panels", required=True)
    args = p.parse_args()
    with np.load(args.artifact, allow_pickle=False) as data:
        ids, raw, sites = data["subject_ids"].astype(str), data["raw_images"][:, 0], data["site_ids"]
    row = int(np.flatnonzero(ids == args.subject)[0])
    inp = raw[row].astype(np.float64)
    fg = inp > .02
    vmax = float(np.percentile(inp[fg], 99.5))
    rows, cols = np.nonzero(fg)
    r0 = int(np.clip(rows.mean() - ZOOM // 2, 0, inp.shape[0] - ZOOM))
    c0 = int(np.clip(cols.mean() - ZOOM // 2, 0, inp.shape[1] - ZOOM))
    crop = lambda img: img[r0:r0 + ZOOM, c0:c0 + ZOOM]

    dice = lambda m: 2 * (fg & m).sum() / (fg.sum() + m.sum())
    centred = lambda img: (img - img.mean()) / np.linalg.norm(img - img.mean())
    box_corr = lambda img: (centred(crop(inp)) * centred(crop(img))).sum()
    nyu = np.flatnonzero(sites == TARGET_SITE)
    scores = [dice(raw[i] > .02) * box_corr(raw[i].astype(np.float64)) for i in nyu]
    match = int(nyu[np.argmax(scores)])
    blurred = gaussian_filter(raw[row], sigma=args.sigma, mode="constant", cval=0.0).astype(np.float64)
    psnr = 10 * np.log10(1 / np.mean((blurred - inp) ** 2))

    out = Path(args.panels)
    render(raw[match], "gray", 0, vmax, (459, 460)).save(out / "image_nyu_target.png")
    render(crop(raw[match]), "gray", 0, vmax, (230, 230)).save(out / "inset_nyu_target.png")
    render(blurred, "gray", 0, vmax, (459, 460)).save(out / "image_blur.png")
    render(crop(blurred), "gray", 0, vmax, (230, 230)).save(out / "inset_blur.png")
    render(blurred - inp, DIVERGING, -LIMIT, LIMIT, (460, 460)).save(out / "diff_blur.png")
    if args.haca3:
        with np.load(args.haca3, allow_pickle=False) as data:
            assert str(data["subject_ids"][row]) == args.subject
            haca3 = data["images"][row, 0].astype(np.float64)
        render(haca3, "gray", 0, vmax, (459, 460)).save(out / "image_haca3.png")
        render(crop(haca3), "gray", 0, vmax, (230, 230)).save(out / "inset_haca3.png")
        render(haca3 - inp, DIVERGING, -LIMIT, LIMIT, (460, 460)).save(out / "diff_haca3.png")
        print(f"HACA3 PSNR {10 * np.log10(1 / np.mean((haca3 - inp) ** 2)):.1f} dB")
    print(f"NYU match {ids[match]}; blur sigma {args.sigma}: PSNR {psnr:.1f} dB on {args.subject}")


if __name__ == "__main__":
    main()
