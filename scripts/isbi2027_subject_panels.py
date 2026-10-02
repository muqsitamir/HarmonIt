"""Per-subject panels for co-author review: one figure per source test subject.

Row 1: the input (no harmonization) and a subject from the target site, each with its site label.
Row 2: four harmonized outputs with the paper's metrics for this subject (PSNR and XCorr against
the input, and the change in Wasserstein-1 distance to the NYU reference). Row 3: output - input.

Subjects come from fixed rules, not appearance: the five source sites with the most test subjects
(ties broken alphabetically) and, in each, the subject whose PSNR averaged over the shown methods
is the median. The target-site subject is the NYU test subject whose slice correlates best with the
input (normalized cross-correlation), so both show a similar anatomical level. Images are 256x256 slices drawn at an
integer scale with nearest-neighbour sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

METHODS = ["histogram_matching", "cyclegan_tuned", "diffusion_20k", "adapted_hcld"]
TITLES = {"histogram_matching": "Histogram matching", "cyclegan_tuned": "CycleGAN",
          "diffusion_20k": "Diffusion img2img", "adapted_hcld": "Adapted HCLD"}
TARGET_SITE = 5  # NYU
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#a8322f"])
INK, MUTED = "#0b0b0b", "#52514e"


def site_name(subject_id):
    return subject_id.rsplit("_", 1)[0]


def load(artifacts):
    images, raw, ids = {}, None, None
    for m in METHODS:
        with np.load(Path(artifacts) / f"{m}.npz", allow_pickle=False) as data:
            if ids is None:
                ids, raw, slices = data["subject_ids"].astype(str), data["raw_images"][:, 0], data["slice_indices"]
            assert (data["subject_ids"].astype(str) == ids).all(), m
            assert np.abs(data["raw_images"][:, 0] - raw).max() < 1e-5, m
            images[m] = data["images"][:, 0]
    return ids, raw, slices, images


def select(metrics, n_sites):
    source = metrics[metrics.site_id != TARGET_SITE]
    counts = source.drop_duplicates("subject_id").subject_id.map(site_name).value_counts()
    sites = sorted(counts.index, key=lambda s: (-counts[s], s))[:n_sites]
    mean_psnr = source.groupby("subject_id").psnr.mean()
    chosen = []
    for site in sites:
        ranked = mean_psnr[mean_psnr.index.map(site_name) == site].sort_values()
        chosen.append(ranked.index[len(ranked) // 2])
    return chosen, counts


def matched_target(row, ids, raw):
    """NYU test subject whose slice has the highest normalized cross-correlation with the input."""
    centred = lambda img: (img - img.mean()) / np.linalg.norm(img - img.mean())
    x = centred(raw[row].astype(np.float64))
    nyu = [i for i, s in enumerate(ids) if site_name(s) == "NYU"]
    return nyu[int(np.argmax([(x * centred(raw[i].astype(np.float64))).sum() for i in nyu]))]


def panel(fig, box, image, **kwargs):
    ax = fig.add_axes(box)
    shown = ax.imshow(image, interpolation="nearest", resample=False, **kwargs)
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax, shown


def figure(subject, row, ids, raw, slices, images, metrics, alignment, counts, args):
    k, gap = args.scale, 28
    side = 256 * k
    left, cbar_w, cbar_gap, right = 100, 34, 40, 170
    head, title_h, metric_h, row_gap, foot = 120, 92, 120, 36, 210
    width = left + 4 * side + 3 * gap + cbar_gap + cbar_w + right
    height = head + title_h + side + row_gap + title_h + side + metric_h + side + foot
    dpi = args.dpi
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    box = lambda x, y, w, h: [x / width, 1 - (y + h) / height, w / width, h / height]  # top-left pixel origin
    col_x = [left + j * (side + gap) for j in range(4)]

    inp = raw[row]
    nyu_row = matched_target(row, ids, raw)
    foreground = inp > 0.02
    vmax = float(np.percentile(inp[foreground], 99.5))
    grey = dict(cmap="gray", vmin=0, vmax=vmax)
    site = site_name(subject)

    fig.text(left / width, 1 - 52 / height, f"Test subject {subject}  ·  source site {site}  ·  target site NYU",
             fontsize=19, fontweight="bold", color=INK, va="center")
    # Row 1: input and a target-site subject, centred over the four method columns.
    y1 = head + title_h
    for x, img, label in ((col_x[1], inp, f"Input (no harmonization)\nsite {site}  ·  axial slice {slices[row]}"),
                          (col_x[2], raw[nyu_row], f"Target-site subject\nsite NYU  ·  {ids[nyu_row]}")):
        panel(fig, box(x, y1, side, side), img, **grey)
        fig.text((x + side / 2) / width, 1 - (y1 - 14) / height, label, fontsize=14, ha="center", va="bottom",
                 color=INK, linespacing=1.3)

    # Row 2: outputs with this subject's metrics; Row 3: output - input.
    y2 = y1 + side + row_gap + title_h
    y3 = y2 + side + metric_h
    for j, m in enumerate(METHODS):
        out = images[m][row]
        panel(fig, box(col_x[j], y2, side, side), np.clip(out, 0, 1), **grey)
        fig.text((col_x[j] + side / 2) / width, 1 - (y2 - 14) / height, TITLES[m], fontsize=15, fontweight="bold",
                 ha="center", va="bottom", color=INK)
        s = metrics[(metrics.method == m) & (metrics.subject_id == subject)].iloc[0]
        a = alignment[(alignment.method == m) & (alignment.subject_id == subject)].iloc[0]
        dw = a.target_wasserstein_harmonized - a.target_wasserstein_raw
        fig.text((col_x[j] + side / 2) / width, 1 - (y2 + side + 16) / height,
                 f"PSNR {s.psnr:.1f} dB   XCorr {s.cross_correlation:.3f}\n$\\Delta W_{{\\mathrm{{NYU}}}}$ {dw:+.4f}",
                 fontsize=13.5, ha="center", va="top", color=INK, linespacing=1.45)
        _, diff = panel(fig, box(col_x[j], y3, side, side), out - inp, cmap=DIVERGING, vmin=-args.limit,
                        vmax=args.limit)
    fig.text((left - 14) / width, 1 - (y2 + side / 2) / height, "harmonized output", rotation=90, fontsize=13,
             ha="right", va="center", color=MUTED)
    fig.text((left - 14) / width, 1 - (y3 + side / 2) / height, "output − input", rotation=90, fontsize=13,
             ha="right", va="center", color=MUTED)
    cax = fig.add_axes(box(col_x[3] + side + cbar_gap, y3, cbar_w, side))
    bar = fig.colorbar(diff, cax=cax)
    bar.ax.tick_params(labelsize=12, length=4)
    bar.set_label("output − input (intensity, [0, 1] scale)", fontsize=12.5, color=MUTED)

    n_site = int(counts[site])
    note = (f"Fixed selection rules, not appearance: the five source sites with the most test subjects (ties broken "
            f"alphabetically); here {site} ({n_site} test subjects), the subject with median PSNR over the four methods.\n"
            "Target-site subject: the NYU test subject whose slice correlates best with this input. "
            "Display window [0, 99.5th percentile of the input's foreground], shared by every image above.\n"
            "PSNR and XCorr compare each output with the input; $\\Delta W_{\\mathrm{NYU}}$ is the change in "
            "Wasserstein-1 distance of foreground intensities to the NYU reference (negative = closer to NYU).\n"
            f"Native 256×256 slices drawn at {k}× with nearest-neighbour sampling.")
    fig.text(left / width, 1 - (height - foot + 30) / height, note, fontsize=11.5, va="top", color=MUTED,
             linespacing=1.5)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", required=True, help="Directory with <method>.npz test artifacts")
    p.add_argument("--eval-run", required=True, help="Evaluation run with <method>_subjects.csv")
    p.add_argument("--alignment", required=True, help="target_alignment_subjects.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sites", type=int, default=5)
    p.add_argument("--scale", type=int, default=3, help="Integer upscaling of each 256x256 slice")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--limit", type=float, default=0.3, help="Difference colour limit")
    args = p.parse_args()

    ids, raw, slices, images = load(args.artifacts)
    run = Path(args.eval_run)
    metrics = pd.concat([pd.read_csv(run / f"{m}_subjects.csv").assign(method=m) for m in METHODS])
    alignment = pd.read_csv(args.alignment)
    # The panels must show the images the paper's metrics were computed on. Source subjects only:
    # target-site subjects are returned almost unchanged, where PSNR is dominated by rounding.
    for m in METHODS:
        psnr = metrics[metrics.method == m].set_index("subject_id").psnr
        rows = [i for i, s in enumerate(ids) if site_name(s) != "NYU"]
        mse = ((images[m][rows].astype(np.float64) - raw[rows]) ** 2).mean(axis=(1, 2))
        assert np.allclose(10 * np.log10(1 / mse), psnr[ids[rows]].values, atol=1e-3), m
    chosen, counts = select(metrics, args.sites)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for n, subject in enumerate(chosen, start=1):
        row = int(np.flatnonzero(ids == subject)[0])
        fig = figure(subject, row, ids, raw, slices, images, metrics, alignment, counts, args)
        path = out / f"subject{n}_{subject}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
