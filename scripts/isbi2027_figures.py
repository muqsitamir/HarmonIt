"""Paper figures from analysis/all_runs_long.csv (scripts/isbi2027_collect.py).

fig_probe_verdicts.pdf
  (a) source site BA versus source PSNR per method: frozen probe (with 95% CI) and
      retrained site probes (mean over seeds, min-max bar), joined per method.
      and converged-recipe probes (amendment 6).
  (b) removed versus hidden: probes trained on raw versus on each method's outputs, tested on
      that method's outputs; image and intensity probes, whole head and brain only (amendment 7).
Validated categorical slots 1, 2, 3 and 7 of the dataviz reference palette, plus distinct marker
shapes so the figure survives grayscale print.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#d9d8d4", "#f1f0ec"
CHANCE_SOURCE = 1 / 16  # 16 source (non-NYU) sites
DX = .3  # horizontal offset (dB) separating probe families at one output
LABELS = {
    "neurocombat": "NeuroCombat", "histogram_matching": "Hist. match", "cyclegan_tuned": "CycleGAN",
    "stargan_aggressive": "StarGAN-A", "stargan_conservative": "StarGAN-C", "dlest_1000": "DLEST-1000",
    "dlest_1500": "DLEST-1500", "diffusion_20k": "Diff. draw 1", "diffusion_20k_redraw": "Diff. draw 2",
    "adapted_hcld": "HCLD",
}
# Label offsets in points, chosen to avoid collisions at column width.
OFFSETS = {"dlest_1000": (10, -3), "dlest_1500": (-4, -9), "stargan_conservative": (-40, 3),
           "diffusion_20k_redraw": (13, -8), "diffusion_20k": (13, 3), "cyclegan_tuned": (-36, 4),
           "neurocombat": (-18, -13), "histogram_matching": (11, 3), "adapted_hcld": (9, 3),
           "stargan_aggressive": (9, -12)}
ADVERSARY = {"histogram_matching": "histogram_matching", "cyclegan_tuned": "cyclegan_tuned",
             "diffusion_20k": "diffusion_20k_redraw"}  # slice-probe source -> matched test artifact


def style():
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    })


def value(df, **query):
    sel = df
    for k, v in query.items():
        sel = sel[sel[k] == v]
    return sel


def panel_tradeoff(ax, df):
    """Dot plot: one row per output, ordered by PSNR (most changed at the top)."""
    src = df[df.group == "source_non_nyu"]
    frozen = src[src.family == "frozen"]
    psnr = value(frozen, metric="psnr").set_index("method").estimate.sort_values(ascending=False)
    f_ba = value(frozen, metric="harmonized_site_ba").set_index("method")
    rows = {method: i for i, method in enumerate(psnr.index)}
    families = (("retrained_site_probe", "best", ORANGE, "s", -.27, "benchmark recipe"),
                ("converged_site_probe", "last", VIOLET, "D", .27, "converged recipe"))
    for family, ckpt, color, marker, offset, label in families:
        sel = src[(src.family == family) & (src.source == "raw") & (src.ckpt == ckpt) & (src.metric == "harmonized_site_ba")]
        agg = sel.groupby("method").estimate.agg(["mean", "min", "max"])
        y = np.array([rows[m] for m in agg.index]) + offset
        ax.hlines(y, agg["min"], agg["max"], color=color, lw=1.1, zorder=2)
        ax.scatter(agg["mean"], y, s=11, marker=marker, color=color, edgecolor="white", lw=.4, zorder=3,
                   label=f"{label} (mean, range, 5 seeds)")
    y = np.array([rows[m] for m in f_ba.index])
    ax.hlines(y, f_ba.ci_low, f_ba.ci_high, color=BLUE, lw=1.1, zorder=2)
    ax.scatter(f_ba.estimate, y, s=14, marker="o", color=BLUE, edgecolor="white", lw=.4, zorder=3,
               label="frozen probe (95% CI)")
    ax.set_yticks(range(len(psnr)), [f"{LABELS.get(m, m)}  {psnr[m]:.1f}" for m in psnr.index])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(psnr) - .45, -.55)
    ax.axvline(CHANCE_SOURCE, color=MUTED, lw=.6, ls=":", zorder=0)
    ax.text(CHANCE_SOURCE + .01, -.35, "chance", fontsize=5.3, color=MUTED, va="center")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Source site BA on outputs")
    ax.text(-.02, 1.005, "Output, PSNR (dB)", transform=ax.transAxes, fontsize=5.8, color=MUTED, ha="right", va="bottom")
    ax.grid(axis="x", color=GRID, lw=.4)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in sorted(labels, key=lambda s: not s.startswith("frozen"))]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left", bbox_to_anchor=(-.02, 1.04),
              ncol=1, frameon=False, handletextpad=.2, borderaxespad=0, labelspacing=.15)


def panel_adversary(ax, df, hist, hist_brain):
    """Dumbbells: probe trained on raw (blue) -> on the method's outputs (aqua), tested on those outputs."""
    src = df[(df.group == "source_non_nyu") & (df.ckpt == "best")]
    kinds = (("slice_probe", None, -.33, "o"), ("slice_probe", hist, -.11, "s"),
             ("brain_slice_probe", None, .11, "o"), ("brain_slice_probe", hist_brain, .33, "s"))
    controls = {"slice_probe": ("silhouette", "silhouette"), "brain_slice_probe": ("brain_shape", "brain_shape")}
    for i, (source, artifact) in enumerate(ADVERSARY.items()):
        ax.axvspan(i + .005, i + .45, color=SHADE, lw=0, zorder=0)
        for family, h, offset, marker in kinds:
            x = i + offset
            fam = src[src.family == family]
            if h is None:  # image probes: mean over seeds, min-max whiskers
                c_src, c_art = controls[family]
                ctrl = value(fam, source=c_src, method=c_art, metric="harmonized_site_ba").estimate
                if len(ctrl):
                    ax.fill_between([x - .09, x + .09], ctrl.min(), ctrl.max(), color=GRID, lw=0, zorder=.5)
                raw_test = value(fam, source="raw", method="neurocombat", metric="raw_site_ba").estimate.mean()
                pts = []
                for train_src in ("raw", source):
                    v = value(fam, source=train_src, method=artifact, metric="harmonized_site_ba").estimate
                    pts.append((v.mean(), v.min(), v.max()))
            else:  # intensity probes: estimate with 95% interval
                own = next(e["source_ba"] for e in h["own_trained"].values() if e["test_artifact"] == artifact)
                raw_test = h["raw_trained"]["raw"]["estimate"]
                pts = [(e["estimate"], *e["ci95"]) for e in (h["raw_trained"][artifact], own)]
            ax.plot([x, x], [pts[0][0], pts[1][0]], color=MUTED, lw=.7, zorder=1)
            ax.plot([x - .06, x + .06], [raw_test] * 2, color=INK, lw=.9, zorder=1.5)
            for (est, lo, hi), color in zip(pts, (BLUE, AQUA)):
                ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt=marker, ms=3.4, color=color, mec="white",
                            mew=.4, ecolor=color, elinewidth=.7, zorder=3)
    ax.axhline(CHANCE_SOURCE, color=MUTED, lw=.6, ls=":", zorder=0)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=3.4, color=BLUE, label="trained on raw"),
        plt.Line2D([], [], ls="", marker="o", ms=3.4, color=AQUA, label="trained on outputs"),
        plt.Line2D([], [], ls="", marker="o", ms=3.4, color=MUTED, label="image probe"),
        plt.Line2D([], [], ls="", marker="s", ms=3.4, color=MUTED, label="intensity probe"),
        plt.Line2D([], [], ls="-", lw=.9, color=INK, label="raw test images"),
        plt.Rectangle((0, 0), 1, 1, color=GRID, label="geometry only"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-.02, 1.04), ncol=3, frameon=False,
              handletextpad=.2, columnspacing=.8, borderaxespad=0, labelspacing=.15, fontsize=6)
    ax.set_xticks(range(len(ADVERSARY)), ["Hist. match", "CycleGAN", "Diff. draw 2"])
    for i in range(len(ADVERSARY)):
        ax.text(i - .22, 1.03, "head", fontsize=5.3, color=MUTED, ha="center", va="top")
        ax.text(i + .22, 1.03, "brain", fontsize=5.3, color=MUTED, ha="center", va="top")
    ax.set_xlim(-.5, len(ADVERSARY) - .5)
    ax.set_ylabel("Source site BA on outputs")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, lw=.4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--histogram-probe", required=True)
    p.add_argument("--histogram-probe-brain", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    style()
    df = pd.read_csv(args.csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(3.39, 4.45))
    axes = [fig.add_axes([.295, .565, .675, .345]), fig.add_axes([.12, .065, .85, .335])]
    panel_tradeoff(axes[0], df)
    load = lambda path: json.loads(Path(path).read_text())
    panel_adversary(axes[1], df, load(args.histogram_probe), load(args.histogram_probe_brain))
    fig.text(.01, .99, "(a)", fontsize=7.5, fontweight="bold", va="top")
    fig.text(.01, .49, "(b)", fontsize=7.5, fontweight="bold", va="top")
    fig.savefig(out / "fig_probe_verdicts.pdf")
    fig.savefig(out / "fig_probe_verdicts.png", dpi=300)
    print(out / "fig_probe_verdicts.pdf")


if __name__ == "__main__":
    main()
