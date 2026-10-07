"""Rebuild Fig. 2 (fig_qualitative_wide.pdf) from its saved panels.

The second diffusion draw shown in Fig. 2 was deleted from vpulab on 2026-10-01, so the figure
can no longer be rendered from data with `isbi2027_qualitative.py`. Its panels were extracted
pixel for pixel from the published figure into `paper/isbi2027/figures/qualitative_panels/`;
the NYU-target and blur-control panels come from `isbi2027_qualitative_extra_panels.py`. Text
sizes reproduce the original figure as printed (6 pt titles). Subject UM_50428, PSNR values and
the zoom box are those of the original figure (selection: median PSNR over the nine outputs in
`isbi2027_qualitative.py`'s ORDER); the blur and HACA3 PSNRs are from their runs (amendments 12, 13).
HACA3 replaced the second diffusion draw, which Table 1 still reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

COLUMNS = [("input", "Input"), ("nyu_target", "NYU subject"), ("histogram_matching", "Hist. match."),
           ("cyclegan", "CycleGAN"), ("diffusion_draw1", "Diff. (draw 1)"), ("haca3", "HACA3"),
           ("blur", "Blur, $\\sigma{=}2$")]
REFERENCES = {"input", "nyu_target"}  # shown without a difference map
PSNR = {"histogram_matching": "20.9 dB", "cyclegan": "17.2 dB", "diffusion_draw1": "22.7 dB",
        "diffusion_draw2": "22.9 dB", "haca3": "15.4 dB", "blur": "25.0 dB"}
ZOOM_BOX = (.3724, .3687, .2503)  # x0, y0 (axes fraction, from bottom-left) and side of the input's zoom box
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#a8322f"])
LIMIT = .3
INK, MUTED = "#0b0b0b", "#52514e"
SIZE, LABEL = 6, 5.4  # the original figure's text as printed


def bare(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panels", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    panels = Path(args.panels)
    plt.rcParams.update({"font.size": SIZE, "font.family": "DejaVu Sans", "pdf.fonttype": 42})

    width, side, gap, left, top_band, row_gap, bottom = 7.0, .83, .05, .02, .14, .04, .02
    height = top_band + side + row_gap + side + bottom
    fig = plt.figure(figsize=(width, height))
    box = lambda x, y, w, h: [x / width, 1 - (y + h) / height, w / width, h / height]  # inches from top-left
    col_x = [left + j * (side + gap) for j in range(len(COLUMNS))]
    y_top, y_bottom = top_band, top_band + side + row_gap

    for j, (key, title) in enumerate(COLUMNS):
        ax = fig.add_axes(box(col_x[j], y_top, side, side))
        ax.imshow(plt.imread(panels / f"image_{key}.png"), interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=SIZE, pad=3, color=INK)
        bare(ax)
        inset = ax.inset_axes([.5, 0, .5, .5])
        inset.imshow(plt.imread(panels / f"inset_{key}.png"), interpolation="nearest", aspect="auto")
        inset.set_xticks([]), inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor("white"), spine.set_linewidth(.6)
        if key == "input":
            x0, y0, s = ZOOM_BOX
            ax.add_patch(plt.Rectangle((x0, y0), s, s, transform=ax.transAxes, fill=False, edgecolor="white", lw=.6))
        if key in REFERENCES:
            continue
        diff = fig.add_axes(box(col_x[j], y_bottom, side, side))
        diff.imshow(plt.imread(panels / f"diff_{key}.png"), interpolation="nearest", aspect="auto")
        diff.text(.03, .04, PSNR[key], fontsize=LABEL, color=INK, transform=diff.transAxes)
        bare(diff)
    fig.text((col_x[0] + side + gap / 2) / width, 1 - (y_bottom + side / 2) / height, "output\n− input",
             fontsize=SIZE, color=MUTED, ha="center", va="center")
    cax = fig.add_axes(box(col_x[-1] + side + .06, y_bottom, .06, side))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-LIMIT, LIMIT), cmap=DIVERGING), cax=cax,
                       ticks=[-.2, 0, .2])
    bar.ax.tick_params(labelsize=SIZE, length=2, pad=1)
    bar.set_label("output − input", size=LABEL, color=MUTED, labelpad=2)
    out = Path(args.out)
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(out)


if __name__ == "__main__":
    main()
