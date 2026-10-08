"""Paper figure fig_probe_verdicts.pdf from analysis/all_runs_long.csv (scripts/isbi2027_collect.py).

Is site still decodable after harmonization? Slice probes trained on raw slices versus on each
method's outputs, all tested on that method's outputs: (a) whole head, (b) brain only
(amendment 7). Image probes show the mean and range over seeds, intensity-histogram probes their
95% interval, and each image-probe pair is annotated with the paired difference in three-seed
mean BA (amendment 9). Per-output site BA of the raw-trained probes is in Table 1. HACA3
(amendments 13, 14 and 17) is a fourth group in both panels, from its own runs.
Neutral grey and validated aqua from the dataviz reference palette, plus distinct marker shapes
so the figure survives grayscale print.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

AQUA = "#1baf7a"
RAW_TRAINED = "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
OUTLINE = "#a9a79f"  # hatched band for outline-only probes, distinct from grey (raw-trained) markers
CHANCE_SOURCE = 1 / 17  # uniform guess over the 17 predicted classes; BA averages the 16 source classes
ADVERSARY = {"histogram_matching": "histogram_matching", "cyclegan_tuned": "cyclegan_tuned",
             "diffusion_20k": "diffusion_20k_redraw"}  # slice-probe source -> matched test artifact


def style():
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "hatch.linewidth": .6,
    })


def value(df, **query):
    sel = df
    for k, v in query.items():
        sel = sel[sel[k] == v]
    return sel


def draw_pair(ax, x, pts, raw_test, marker):
    ax.plot([x, x], [pts[0][0], pts[1][0]], color=MUTED, lw=.7, zorder=1)
    ax.plot([x - .08, x + .08], [raw_test] * 2, color=INK, lw=.9, zorder=1.5)
    for (est, lo, hi), color in zip(pts, (RAW_TRAINED, AQUA)):
        ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt=marker, ms=3.8, color=color, mec="white",
                    mew=.4, ecolor=color, elinewidth=.7, zorder=3)


def draw_haca3(ax, haca3, ctrl):
    """Fourth group: raw-trained vs HACA3-trained image and histogram probes, tested on HACA3."""
    x = len(ADVERSARY) - .17
    if haca3 is None:
        ax.text(len(ADVERSARY), .5, "not run", fontsize=5.5, color=MUTED, ha="center", va="center")
        return
    src = haca3["csv"][(haca3["csv"].group == "source_non_nyu") & (haca3["csv"].ckpt == "best")
                       & (haca3["csv"].family == haca3["family"])]
    if len(ctrl):
        ax.fill_between([x - .12, x + .12], ctrl.min(), ctrl.max(), facecolor="#f4f3ef", edgecolor=OUTLINE, hatch="//////", lw=.4, zorder=.5)
    pts = []
    for train_src in ("raw", "haca3"):
        v = value(src, source=train_src, method="haca3", metric="harmonized_site_ba").estimate
        pts.append((v.mean(), v.min(), v.max()))
    raw_test = value(src, source="raw", method="haca3", metric="raw_site_ba").estimate.mean()
    draw_pair(ax, x, pts, raw_test, "o")
    ax.text(x, 1.1, f"{haca3['difference']:+.2f}", fontsize=6, color=INK, ha="center", va="center")
    h = haca3["histogram"]
    pts = [(e["estimate"], *e["ci95"]) for e in (h["raw_trained"]["haca3"], h["own_trained"]["haca3"]["source_ba"])]
    draw_pair(ax, len(ADVERSARY) + .17, pts, h["raw_trained"]["raw"]["estimate"], "s")


def panel_adversary(ax, df, hist, family, control, title, differences, haca3=None):
    """One input restriction (whole head or brain only): raw-trained -> output-trained probes.

    Grey = probe trained on raw images, aqua = probe trained on that method's outputs; circles are
    image probes (mean, min-max over 3 seeds), squares intensity-histogram probes (95% interval).
    The number above each image-probe pair is the paired difference in three-seed mean BA.
    """
    src = df[(df.group == "source_non_nyu") & (df.ckpt == "best") & (df.family == family)]
    for i, (source, artifact) in enumerate(ADVERSARY.items()):
        for h, offset, marker in ((None, -.17, "o"), (hist, .17, "s")):
            x = i + offset
            if h is None:  # image probes
                ctrl = value(src, source=control, method=control, metric="harmonized_site_ba").estimate
                if len(ctrl):
                    ax.fill_between([x - .12, x + .12], ctrl.min(), ctrl.max(), facecolor="#f4f3ef", edgecolor=OUTLINE, hatch="//////", lw=.4, zorder=.5)
                raw_test = value(src, source="raw", method="neurocombat", metric="raw_site_ba").estimate.mean()
                pts = []
                for train_src in ("raw", source):
                    v = value(src, source=train_src, method=artifact, metric="harmonized_site_ba").estimate
                    pts.append((v.mean(), v.min(), v.max()))
                diff = differences[source]["difference"]
                ax.text(x, 1.1, f"{diff:+.2f}", fontsize=6, color=INK, ha="center", va="center")
            else:  # intensity-histogram probes
                own = next(e["source_ba"] for e in h["own_trained"].values() if e["test_artifact"] == artifact)
                raw_test = h["raw_trained"]["raw"]["estimate"]
                pts = [(e["estimate"], *e["ci95"]) for e in (h["raw_trained"][artifact], own)]
            draw_pair(ax, x, pts, raw_test, marker)
    draw_haca3(ax, haca3, value(src, source=control, method=control, metric="harmonized_site_ba").estimate)
    ax.axhline(CHANCE_SOURCE, color=MUTED, lw=.6, ls=":", zorder=0)
    ax.text(-.47, .07, "chance", fontsize=5.5, color=MUTED, va="bottom")
    ax.set_title(title, loc="left", fontsize=7, fontweight="bold", pad=3)
    ax.set_xticks(range(len(ADVERSARY) + 1), ["Hist. match", "CycleGAN", "Diff. (draw 2)", "HACA3"])
    ax.set_xlim(-.5, len(ADVERSARY) + .5)
    ax.set_ylim(0, 1.17)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel("Source site BA on outputs", fontsize=6.5)
    ax.grid(axis="y", color=GRID, lw=.4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--histogram-probe", required=True)
    p.add_argument("--histogram-probe-brain", required=True)
    p.add_argument("--differences", required=True, help="analysis/probe_difference.json (amendment 9)")
    p.add_argument("--haca3-csv", required=True, help="analysis/haca3_runs_long.csv (amendment 13)")
    p.add_argument("--haca3-differences", required=True, help="analysis/probe_difference_haca3_best.json")
    p.add_argument("--haca3-histogram", required=True, help="analysis/histogram_probe_haca3.json (amendment 14)")
    p.add_argument("--haca3-brain-csv", required=True, help="analysis/haca3_brain_runs_long.csv (amendment 17)")
    p.add_argument("--haca3-brain-differences", required=True, help="analysis/probe_difference_haca3_brain_best.json")
    p.add_argument("--haca3-brain-histogram", required=True, help="analysis/histogram_probe_haca3_brain.json")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    style()
    df = pd.read_csv(args.csv)
    load = lambda path: json.loads(Path(path).read_text())
    diffs = load(args.differences)["results"]
    by_family = lambda family: {key.split("/")[1]: r for key, r in diffs.items() if key.startswith(family)}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(3.39, 3.2))
    ax_a = fig.add_axes([.15, .615, .83, .33])
    ax_b = fig.add_axes([.15, .165, .83, .32])
    haca3 = {"csv": pd.read_csv(args.haca3_csv), "histogram": load(args.haca3_histogram), "family": "slice_probe",
             "difference": load(args.haca3_differences)["results"]["whole_head/haca3"]["difference"]}
    haca3_brain = {"csv": pd.read_csv(args.haca3_brain_csv), "histogram": load(args.haca3_brain_histogram),
                   "family": "brain_slice_probe",
                   "difference": load(args.haca3_brain_differences)["results"]["brain_only/haca3"]["difference"]}
    panel_adversary(ax_a, df, load(args.histogram_probe), "slice_probe", "silhouette", "(a) whole head",
                    by_family("whole_head"), haca3)
    panel_adversary(ax_b, df, load(args.histogram_probe_brain), "brain_slice_probe", "brain_shape", "(b) brain only",
                    by_family("brain_only"), haca3_brain)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=3.8, color=RAW_TRAINED, label="trained on raw"),
        plt.Line2D([], [], ls="", marker="o", ms=3.8, color=AQUA, label="trained on outputs"),
        plt.Line2D([], [], ls="", marker="o", ms=3.8, color=MUTED, label="image probe"),
        plt.Line2D([], [], ls="", marker="s", ms=3.8, color=MUTED, label="intensity probe"),
        plt.Line2D([], [], ls="-", lw=.9, color=INK, label="raw test images"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#f4f3ef", edgecolor=OUTLINE, hatch="//////", lw=.4, label="outline only"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, 0), ncol=3, frameon=False,
               handletextpad=.2, columnspacing=.9, labelspacing=.2, fontsize=6)
    fig.savefig(out / "fig_probe_verdicts.pdf")
    fig.savefig(out / "fig_probe_verdicts.png", dpi=300)
    print(out / "fig_probe_verdicts.pdf")


if __name__ == "__main__":
    main()
