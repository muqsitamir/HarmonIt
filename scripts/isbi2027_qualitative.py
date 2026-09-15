"""Qualitative figure: one source test subject across all evaluated outputs.

The subject is chosen by a fixed rule, not by appearance: the source (non-NYU) subject
whose PSNR averaged over all outputs is the median. Row 1 shows images, row 2 the
signed difference output - input on a diverging blue/red scale with a neutral midpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ORDER = ["neurocombat", "histogram_matching", "cyclegan_tuned", "stargan_aggressive", "stargan_conservative",
         "dlest_1000", "diffusion_20k", "diffusion_20k_redraw", "adapted_hcld"]
TITLES = {"neurocombat": "NeuroComb.", "histogram_matching": "Hist. match", "cyclegan_tuned": "CycleGAN",
          "stargan_aggressive": "StarGAN-A", "stargan_conservative": "StarGAN-C", "dlest_1000": "DLEST-1000",
          "diffusion_20k": "Diffusion", "diffusion_20k_redraw": "Diff. draw 2", "adapted_hcld": "HCLD"}
SHORT = {"histogram_matching": "Hist. match.", "diffusion_20k": "Diff. draw 1", "diffusion_20k_redraw": "Diff. draw 2"}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#a8322f"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-run", required=True, help="Complete evaluation run with raw_reference.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=float, default=0.3, help="Difference colour limit")
    p.add_argument("--show", nargs="+", help="Outputs to display (default all); selection still uses all")
    p.add_argument("--width", type=float, default=7.0, help="Figure width in inches")
    p.add_argument("--height", type=float, help="Figure height in inches (default from width)")
    args = p.parse_args()
    run = Path(args.eval_run)
    protocol = json.loads((run / "protocol.json").read_text())
    paths = dict(spec.split("=", 1) for spec in protocol["args"]["artifact"])
    methods = [m for m in ORDER if m in paths]

    frames = [pd.read_csv(run / f"{m}_subjects.csv")[["subject_id", "site_id", "psnr"]].assign(method=m) for m in methods]
    table = pd.concat(frames)
    source = table[table.site_id != protocol["args"]["target_site_id"]]
    mean_psnr = source.groupby("subject_id").psnr.mean().sort_values()
    subject = mean_psnr.index[len(mean_psnr) // 2]

    with np.load(run / "raw_reference.npz", allow_pickle=False) as data:
        row = int(np.flatnonzero(data["subject_ids"].astype(str) == subject)[0])
        raw = data["raw_images"][row, 0]
    images = {}
    for m in methods:
        with np.load(paths[m], allow_pickle=False) as data:
            assert str(data["subject_ids"][row]) == subject
            images[m] = data["images"][row, 0]

    plt.rcParams.update({"font.size": 6, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    shown = [m for m in methods if not args.show or m in args.show]
    cols = len(shown) + 1
    size = 5 if args.show else 6
    fig, axes = plt.subplots(2, cols, figsize=(args.width, args.height or args.width * 2 / cols * 1.12),
                             gridspec_kw=dict(wspace=.04, hspace=.06))
    axes[0, 0].imshow(raw, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input", fontsize=size, pad=2)
    axes[1, 0].text(.5, .5, "output\n$-$ input", ha="center", va="center", fontsize=size, color="#52514e",
                    transform=axes[1, 0].transAxes)
    psnr = table[table.subject_id == subject].set_index("method").psnr
    for j, m in enumerate(shown, start=1):
        axes[0, j].imshow(np.clip(images[m], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title((SHORT if args.show else {}).get(m, TITLES[m]), fontsize=size, pad=2)
        diff = axes[1, j].imshow(images[m] - raw, cmap=DIVERGING, vmin=-args.limit, vmax=args.limit)
        axes[1, j].text(.03, .04, f"{psnr[m]:.1f} dB", color="#0b0b0b", fontsize=size - 0.5, transform=axes[1, j].transAxes)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(diff, ax=axes[1, :].tolist(), fraction=.012, pad=.005)
    bar.ax.tick_params(labelsize=5, length=2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=.01)
    fig.savefig(Path(args.out).with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.01)
    print(f"subject {subject} (site {int(source[source.subject_id == subject].site_id.iloc[0])}); saved {args.out}")


if __name__ == "__main__":
    main()
