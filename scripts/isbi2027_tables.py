"""LaTeX results table for the ISBI 2027 paper from collected runs and target alignment.

All quantities are for the 90 source (non-NYU) test subjects. Retrained and slice-probe
columns summarize seeds as mean [min, max]; frozen-probe and preservation columns show
point estimates with paired subject-bootstrap 95% intervals where space allows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ORDER = ["neurocombat", "histogram_matching", "cyclegan_tuned", "stargan_aggressive", "stargan_conservative",
         "dlest_1500", "dlest_1000", "diffusion_20k", "diffusion_20k_redraw", "adapted_hcld"]
NAMES = {"neurocombat": "NeuroCombat$^\\dagger$", "histogram_matching": "Histogram matching$^\\ddagger$",
         "cyclegan_tuned": "CycleGAN", "stargan_aggressive": "StarGAN (aggr.)", "stargan_conservative": "StarGAN (cons.)",
         "dlest_1500": "DLEST 1500", "dlest_1000": "DLEST 1000", "diffusion_20k": "Diffusion img2img",
         "diffusion_20k_redraw": "\\quad second draw", "adapted_hcld": "Adapted HCLD"}
OWN_PROBE = {"histogram_matching": "histogram_matching", "cyclegan_tuned": "cyclegan_tuned",
             "diffusion_20k_redraw": "diffusion_20k"}  # test artifact -> slice-probe source trained on it


def seeds(values, digits=2):
    if not len(values):
        return "--"
    if len(values) == 1:
        return f"{values.iloc[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} [{values.min():.{digits}f}, {values.max():.{digits}f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--alignment", required=True, help="target_alignment_summary.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    src = df[df.group == "source_non_nyu"]
    frozen = src[src.family == "frozen"].set_index(["method", "metric"])
    retrained = src[(src.family == "retrained_site_probe") & (src.source == "raw") & (src.ckpt == "best")]
    slice_best = src[(src.family == "slice_probe") & (src.ckpt == "best")]
    align = json.loads(Path(args.alignment).read_text())["methods"]

    lines = [
        "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
        " & \\multicolumn{3}{c}{Change and target alignment} & \\multicolumn{4}{c}{Source-site balanced accuracy} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(l){5-8}",
        "Method & PSNR & XCorr & $\\Delta W_{\\mathrm{NYU}}$ & Frozen & Retrained & Slice (raw) & Slice (own) \\\\",
        "\\midrule",
    ]
    raw_row = ["Raw input", "--", "--", "0", f"{frozen.loc[('neurocombat', 'raw_site_ba'), 'estimate']:.2f}",
               seeds(retrained[(retrained.method == 'neurocombat') & (retrained.metric == 'raw_site_ba')].estimate),
               seeds(slice_best[(slice_best.source == 'raw') & (slice_best.method == 'neurocombat')
                                & (slice_best.metric == 'raw_site_ba')].estimate), "--"]
    lines.append(" & ".join(raw_row) + " \\\\")
    for method in ORDER:
        if (method, "psnr") not in frozen.index:
            continue
        f = lambda m: frozen.loc[(method, m)]
        ba = f("harmonized_site_ba")
        dw = align.get(method, {}).get("wasserstein", {}).get("harmonized_minus_raw")
        own = OWN_PROBE.get(method)
        row = [
            NAMES[method], f"{f('psnr').estimate:.1f}", f"{f('cross_correlation').estimate:.3f}",
            f"{dw['estimate']:+.3f}" if dw else "--",
            f"{ba.estimate:.2f} [{ba.ci_low:.2f}, {ba.ci_high:.2f}]",
            seeds(retrained[(retrained.method == method) & (retrained.metric == "harmonized_site_ba")].estimate),
            seeds(slice_best[(slice_best.source == "raw") & (slice_best.method == method)
                             & (slice_best.metric == "harmonized_site_ba")].estimate) if own else "--",
            seeds(slice_best[(slice_best.source == own) & (slice_best.method == method)
                             & (slice_best.metric == "harmonized_site_ba")].estimate) if own else "--",
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
