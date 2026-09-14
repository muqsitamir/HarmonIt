"""Paper figures from analysis/all_runs_long.csv (scripts/isbi2027_collect.py).

fig_probe_verdicts.pdf
  (a) source site BA versus source PSNR per method: frozen probe (with 95% CI) and
      retrained site probes (mean over seeds, min-max bar), joined per method.
  (b) harmonized-probe test: raw-trained versus method-trained slice probes on each
      method's own test outputs, one marker per seed.
Validated categorical slots 1-3 of the dataviz reference palette, plus distinct marker
shapes so the figure survives grayscale print.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
CHANCE_SOURCE = 1 / 16  # 16 source (non-NYU) sites
LABELS = {
    "neurocombat": "NeuroCombat$^\\dagger$", "histogram_matching": "Hist. match", "cyclegan_tuned": "CycleGAN",
    "stargan_aggressive": "StarGAN-A", "stargan_conservative": "StarGAN-C", "dlest_1000": "DLEST-1000",
    "dlest_1500": "DLEST-1500", "diffusion_20k": "Diffusion", "diffusion_20k_redraw": "Diffusion (redraw)",
    "adapted_hcld": "HCLD",
}
# Label offsets in points, chosen to avoid collisions at column width.
OFFSETS = {"dlest_1000": (4, 3), "dlest_1500": (-4, -9), "stargan_conservative": (-40, 3),
           "diffusion_20k_redraw": (4, -8), "diffusion_20k": (4, 3), "cyclegan_tuned": (-36, 4),
           "neurocombat": (-18, -13)}
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
    src = df[df.group == "source_non_nyu"]
    frozen = src[src.family == "frozen"]
    retrained = src[(src.family == "retrained_site_probe") & (src.source == "raw") & (src.ckpt == "best")]
    psnr = value(frozen, metric="psnr").set_index("method").estimate
    f_ba = value(frozen, metric="harmonized_site_ba").set_index("method")
    r_ba = value(retrained, metric="harmonized_site_ba").groupby("method").estimate.agg(["mean", "min", "max", "count"])
    for method in f_ba.index:
        x = psnr[method]
        if method in r_ba.index:
            ax.plot([x, x], [f_ba.estimate[method], r_ba["mean"][method]], color=GRID, lw=1, zorder=1)
            ax.plot([x, x], [r_ba["min"][method], r_ba["max"][method]], color=ORANGE, lw=1.2, zorder=2,
                    solid_capstyle="round")
        ax.errorbar(x, f_ba.estimate[method], yerr=[[f_ba.estimate[method] - f_ba.ci_low[method]],
                    [f_ba.ci_high[method] - f_ba.estimate[method]]], fmt="none", ecolor=BLUE, elinewidth=.8, zorder=2)
        dx, dy = OFFSETS.get(method, (4, 3))
        ax.annotate(LABELS.get(method, method), (x, f_ba.estimate[method]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=5.8, color=INK)
    ax.scatter(psnr[f_ba.index], f_ba.estimate, s=22, marker="o", color=BLUE, edgecolor="white", lw=.6, zorder=3,
               label="Frozen probe (95% CI)")
    if len(r_ba):
        n = int(r_ba["count"].max())
        ax.scatter(psnr[r_ba.index], r_ba["mean"], s=20, marker="s", color=ORANGE, edgecolor="white", lw=.6, zorder=2.5,
                   label=f"Retrained probes (mean, range; {n} seed{'s' if n > 1 else ''})")
    raw_frozen = value(frozen, metric="raw_site_ba").estimate.iloc[0]
    ax.axhline(raw_frozen, color=MUTED, lw=.6, ls="--", zorder=0)
    ax.text(psnr.max() + .3, raw_frozen - .05, "raw images (frozen probe)", fontsize=5.5, color=MUTED, ha="right")
    ax.axhline(CHANCE_SOURCE, color=MUTED, lw=.6, ls=":", zorder=0)
    ax.text(psnr.max() - 1.5, CHANCE_SOURCE + .015, "chance", fontsize=5.5, color=MUTED)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.set_xlabel("Source PSNR vs. raw (dB)  [higher = less changed]")
    ax.set_ylabel("Source site balanced accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, handletextpad=.3, borderaxespad=.1)


def panel_adversary(ax, df, hist):
    src = df[(df.group == "source_non_nyu") & (df.family == "slice_probe") & (df.ckpt == "best")]
    if src.empty:
        ax.text(.5, .5, "harmonized-probe runs pending", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return
    rng = np.random.default_rng(0)
    sil = value(src, source="silhouette", method="silhouette", metric="harmonized_site_ba").estimate
    if len(sil):
        ax.axhspan(sil.min(), sil.max(), color=GRID, alpha=.6, lw=0, zorder=0)
        ax.text(.45, sil.min() + .07, "head silhouette only", fontsize=5.5, color=MUTED,
                ha="center", va="center")
    raw_img = value(src, source="raw", method="neurocombat", metric="raw_site_ba").estimate.mean()
    ax.axhline(raw_img, color=MUTED, lw=.6, ls="--", zorder=0)
    ax.axhline(CHANCE_SOURCE, color=MUTED, lw=.6, ls=":", zorder=0)
    ax.text(len(ADVERSARY) - .52, CHANCE_SOURCE + .015, "chance", fontsize=5.5, color=MUTED, ha="right")
    own_hist = {v["test_artifact"]: v["source_ba"] for v in hist["own_trained"].values()}
    for i, (source, artifact) in enumerate(ADVERSARY.items()):
        # Image probes: one marker per seed, filled.
        for offset, train_src, color, marker in ((-.27, "raw", BLUE, "o"), (-.09, source, AQUA, "^")):
            values = value(src, source=train_src, method=artifact, metric="harmonized_site_ba").estimate.to_numpy()
            if len(values):
                ax.scatter(i + offset + rng.uniform(-.025, .025, len(values)), values, s=13, marker=marker,
                           color=color, edgecolor="white", lw=.4, zorder=3)
                ax.plot([i + offset - .07, i + offset + .07], [values.mean()] * 2, color=INK, lw=.9, zorder=4)
        # Intensity-only probes: deterministic, hollow markers with 95% interval.
        for offset, entry, color, marker in ((.09, hist["raw_trained"][artifact], BLUE, "o"),
                                             (.27, own_hist.get(artifact), AQUA, "^")):
            if entry:
                est, (lo, hi) = entry["estimate"], entry["ci95"]
                ax.errorbar(i + offset, est, yerr=[[est - lo], [hi - est]], fmt=marker, ms=4, mfc="white",
                            mec=color, mew=1, ecolor=color, elinewidth=.8, zorder=3)
    raw_int = hist["raw_trained"]["raw"]["estimate"]
    ax.plot([-.5, len(ADVERSARY) - .5], [raw_int] * 2, color=MUTED, lw=.6, ls="-.", zorder=0)
    ax.text(-.48, raw_img + .015, "raw slices: image probe", fontsize=5.5, color=MUTED)
    ax.text(-.48, raw_int - .075, "raw slices: intensity probe", fontsize=5.5, color=MUTED)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=4, color=BLUE, label="trained on raw"),
        plt.Line2D([], [], ls="", marker="^", ms=4, color=AQUA, label="trained on outputs"),
        plt.Line2D([], [], ls="", marker="o", ms=4, color=MUTED, label="image probe (per seed)"),
        plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=MUTED, label="intensity probe (95% CI)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              handletextpad=.2, columnspacing=.8, borderaxespad=.1)
    ax.set_xticks(range(len(ADVERSARY)), ["Hist. match", "CycleGAN", "Diffusion\n(2nd draw)"])
    ax.set_xlim(-.5, len(ADVERSARY) - .5)
    ax.set_ylabel("Source site BA on outputs")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=.4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--histogram-probe", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    style()
    df = pd.read_csv(args.csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(3.39, 4.6), gridspec_kw=dict(height_ratios=[1.45, 1]))
    panel_tradeoff(axes[0], df)
    import json
    panel_adversary(axes[1], df, json.loads(Path(args.histogram_probe).read_text()))
    for ax, tag in zip(axes, "ab"):
        ax.text(-.26, 1.2, f"({tag})", transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top")
    fig.tight_layout(h_pad=.8)
    fig.savefig(out / "fig_probe_verdicts.pdf")
    fig.savefig(out / "fig_probe_verdicts.png", dpi=300)
    print(out / "fig_probe_verdicts.pdf")


if __name__ == "__main__":
    main()
