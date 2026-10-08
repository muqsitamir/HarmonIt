"""LaTeX results table for the ISBI 2027 paper from collected runs and target alignment.

All quantities are for the 90 source (non-NYU) test subjects, as means over subjects. Change:
PSNR, XCorr and the Wasserstein-1 distance W between input and output intensities. Target
alignment: change in Wasserstein-1 distance to the NYU reference and KL divergence to it. Frozen
probe: source BA with its paired bootstrap 95% interval and the share of outputs it labels NYU.
Probes retrained on raw images (benchmark recipe, best-validation checkpoints; converged recipe
of amendment 6, final-epoch checkpoints) are summarized over seeds as mean (min--max). HACA3
(amendment 13) comes from its own runs: the frozen probe and the benchmark-recipe and converged
probes trained with the original recipes and seeds; it replaces the DLEST-style 1500 row. The Gaussian-blur control (amendment 12) was scored by the
frozen probe only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Table 1 shows eight of the nine original outputs: DLEST-style 1500 gave way to HACA3 (amendment 13) and
# conservative StarGAN to space (like DLEST-style 1000, little change and high site accuracy); both
# stay in every statistic and in the results files.
ORDER = ["neurocombat", "histogram_matching", "cyclegan_tuned", "stargan_aggressive", "dlest_1000", "diffusion_20k",
         "diffusion_20k_redraw"]
NAMES = {"neurocombat": "NeuroCombat$^\\dagger$", "histogram_matching": "Histogram matching$^\\ddagger$",
         "cyclegan_tuned": "CycleGAN", "stargan_aggressive": "StarGAN (aggr.)", "stargan_conservative": "StarGAN (cons.)",
         "dlest_1500": "DLEST-style 1500", "dlest_1000": "DLEST-style 1000", "diffusion_20k": "Diffusion, draw 1",
         "diffusion_20k_redraw": "Diffusion, draw 2", "haca3": "HACA3 (pretrained)$^\\S$"}
BLUR = ("gaussian_blur_s2", "Blur control, $\\sigma{=}2$")
TARGET_SITE = 5


def seeds(values, digits=2):
    if not len(values):
        return "--"
    if len(values) == 1:
        return f"{values.iloc[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ({values.min():.{digits}f}--{values.max():.{digits}f})"


def nyu_share(run, method, column="harmonized_prediction"):
    frame = pd.read_csv(Path(run) / f"{method}_subjects.csv")
    source = frame[frame.site_id != TARGET_SITE]
    return f"{100 * (source[column] == TARGET_SITE).mean():.0f}\\%"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--alignment", required=True, help="target_alignment_summary.json of the ten outputs")
    p.add_argument("--frozen-run", required=True, help="Canonical frozen-probe run (per-subject predictions)")
    p.add_argument("--blur-run", required=True, help="Frozen-probe run of the blur control (amendment 12)")
    p.add_argument("--blur-alignment", required=True, help="target_alignment_summary.json of the blur control")
    p.add_argument("--haca3-csv", required=True, help="Collected HACA3 runs (amendment 13)")
    p.add_argument("--haca3-alignment", required=True, help="target_alignment_summary.json of the HACA3 frozen run")
    p.add_argument("--haca3-frozen-run", required=True, help="Frozen-probe run that includes HACA3")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    src = df[df.group == "source_non_nyu"]
    frozen = src[src.family == "frozen"].set_index(["method", "metric"])
    retrained = src[(src.family == "retrained_site_probe") & (src.source == "raw") & (src.ckpt == "best")]
    converged = src[(src.family == "converged_site_probe") & (src.source == "raw") & (src.ckpt == "last")]
    align = json.loads(Path(args.alignment).read_text())["methods"]
    blur_align = json.loads(Path(args.blur_alignment).read_text())["methods"]
    blur = json.loads((Path(args.blur_run) / f"{BLUR[0]}_summary.json").read_text())["groups"]["source_non_nyu"]["metrics"]

    lines = [
        "\\begin{tabular}{@{}lccccccccc@{}}", "\\toprule",
        " & \\multicolumn{3}{c}{Change} & \\multicolumn{2}{c}{Target alignment} & \\multicolumn{2}{c}{Frozen probe}"
        " & \\multicolumn{2}{c}{Probes retrained on raw images} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(l){9-10}",
        # Arrows: the direction usually read as better (the paper questions this reading for site BA).
        "Method & PSNR$\\uparrow$ & XCorr$\\uparrow$ & $W$ & $\\Delta W_{\\mathrm{NYU}}\\!\\downarrow$"
        " & KL$_{\\mathrm{NYU}}\\!\\downarrow$ & Source BA$\\downarrow$ & NYU$\\uparrow$"
        " & Benchmark recipe$\\downarrow$ & Converged recipe$\\downarrow$ \\\\",
        "\\midrule",
    ]
    kl_raw = align["neurocombat"]["kl"]["raw"]["estimate"]
    raw_row = ["Raw input", "--", "--", "0", "0", f"{kl_raw:.2f}",
               f"{frozen.loc[('neurocombat', 'raw_site_ba'), 'estimate']:.2f}",
               nyu_share(args.frozen_run, "neurocombat", "raw_prediction"),
               seeds(retrained[(retrained.method == 'neurocombat') & (retrained.metric == 'raw_site_ba')].estimate),
               seeds(converged[(converged.method == 'neurocombat') & (converged.metric == 'raw_site_ba')].estimate)]
    lines.append(" & ".join(raw_row) + " \\\\")

    def method_row(method, frozen, align, frozen_run, retrained, converged):
        f = lambda m: frozen.loc[(method, m)]
        ba = f("harmonized_site_ba")
        a = align[method]
        return [
            NAMES[method], f"{f('psnr').estimate:.1f}", f"{f('cross_correlation').estimate:.3f}",
            f"{f('subject_wasserstein_raw_harm').estimate:.3f}",
            f"{a['wasserstein']['harmonized_minus_raw']['estimate']:+.3f}", f"{a['kl']['harmonized']['estimate']:.2f}",
            f"{ba.estimate:.2f} [{ba.ci_low:.2f}, {ba.ci_high:.2f}]", nyu_share(frozen_run, method),
            seeds(retrained[(retrained.method == method) & (retrained.metric == "harmonized_site_ba")].estimate),
            seeds(converged[(converged.method == method) & (converged.metric == "harmonized_site_ba")].estimate),
        ]

    for method in ORDER:
        lines.append(" & ".join(method_row(method, frozen, align, args.frozen_run, retrained, converged)) + " \\\\")
    h = pd.read_csv(args.haca3_csv)
    h = h[h.group == "source_non_nyu"]
    lines.append(" & ".join(method_row(
        "haca3", h[h.family == "frozen"].set_index(["method", "metric"]),
        json.loads(Path(args.haca3_alignment).read_text())["methods"], args.haca3_frozen_run,
        h[(h.family == "retrained_site_probe") & (h.source == "raw") & (h.ckpt == "best")],
        h[(h.family == "converged_site_probe") & (h.source == "raw") & (h.ckpt == "last")])) + " \\\\")
    b, ba = blur_align[BLUR[0]], blur["harmonized_site_ba"]
    lines.append("\\midrule")
    lines.append(" & ".join([
        BLUR[1], f"{blur['psnr']['estimate']:.1f}", f"{blur['cross_correlation']['estimate']:.3f}",
        f"{blur['subject_wasserstein_raw_harm']['estimate']:.3f}",
        f"{b['wasserstein']['harmonized_minus_raw']['estimate']:+.3f}", f"{b['kl']['harmonized']['estimate']:.2f}",
        f"{ba['estimate']:.2f} [{ba['ci95'][0]:.2f}, {ba['ci95'][1]:.2f}]", nyu_share(args.blur_run, BLUR[0]), "--", "--",
    ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
