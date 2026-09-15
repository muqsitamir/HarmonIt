"""Converged-probe outcomes (protocol amendment 6).

Per output: source harmonized site BA range over the five converged probes (final-epoch
checkpoints, primary; best-validation also reported) next to the five production-recipe
probes (best-validation, as reported in the paper). Kendall tau between seeds. Convergence:
range of validation BA over the last five epochs per seed, parsed from training logs.
Decision rule fixed in the amendment: converged ranges for the four named outputs no wider
than the widest 95% bootstrap interval of a single converged probe on any output.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

NAMED = ["cyclegan_tuned", "diffusion_20k", "histogram_matching", "stargan_aggressive"]


def spread(sel):
    wide = sel.pivot_table(index="method", columns="seed", values="estimate")
    taus = [kendalltau(wide[a], wide[b])[0] for a, b in itertools.combinations(wide.columns, 2)]
    frame = pd.DataFrame({"mean": wide.mean(1), "min": wide.min(1), "max": wide.max(1)})
    frame["range"] = frame["max"] - frame["min"]
    return wide, frame, dict(min=min(taus), max=max(taus), mean=sum(taus) / len(taus), n_pairs=len(taus))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--logs", required=True, help="Directory with converged_raw_seed<S>.log training logs")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    src = df[(df.group == "source_non_nyu") & (df.source == "raw")]
    ba = src[src.metric == "harmonized_site_ba"]
    report = {}
    for label, family, ckpt in (("converged_last", "converged_site_probe", "last"),
                                ("converged_best", "converged_site_probe", "best"),
                                ("production_best", "retrained_site_probe", "best")):
        sel = ba[(ba.family == family) & (ba.ckpt == ckpt)]
        wide, frame, tau = spread(sel)
        raw = src[(src.family == family) & (src.ckpt == ckpt) & (src.metric == "raw_site_ba") & (src.method == "neurocombat")]
        widths = (sel.ci_high - sel.ci_low)
        report[label] = dict(seeds=sorted(int(s) for s in wide.columns), kendall_tau=tau,
                             widest_single_probe_ci=float(widths.max()),
                             raw_ba={int(r.seed): r.estimate for r in raw.itertuples()},
                             per_output=frame.round(4).to_dict(orient="index"))
    primary = report["converged_last"]
    primary["decision"] = dict(
        threshold=primary["widest_single_probe_ci"],
        named_ranges={m: primary["per_output"][m]["range"] for m in NAMED},
        all_within=all(primary["per_output"][m]["range"] <= primary["widest_single_probe_ci"] for m in NAMED))
    sh = df[(df.group == "source_non_nyu") & (df.family == "converged_site_probe") & (df.source == "shuffle")
            & (df.metric == "raw_site_ba") & (df.method == "neurocombat")]
    report["shuffle_control_raw_ba"] = {r.ckpt: [r.estimate, r.ci_low, r.ci_high] for r in sh.itertuples()}
    convergence = {}
    for log in sorted(Path(args.logs).glob("converged_*_seed*.log")):
        vals = [float(v) for v in re.findall(r"\[VAL\] Epoch \d+ \| acc=[\d.]+ \| bal_acc=([\d.]+)", log.read_text())]
        if vals:
            last5 = vals[-5:]
            convergence[log.stem] = dict(epochs=len(vals), last5=last5, last5_range=max(last5) - min(last5),
                                         best=max(vals), final=vals[-1])
    raw_logs = [v for k, v in convergence.items() if "_raw_" in k]
    report["convergence"] = dict(per_seed=convergence,
                                 stable=bool(raw_logs) and all(v["last5_range"] <= 0.05 for v in raw_logs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    for label in ("production_best", "converged_last", "converged_best"):
        r = report[label]
        print(label, "tau", {k: round(v, 2) for k, v in r["kendall_tau"].items()}, "widest CI", round(r["widest_single_probe_ci"], 3))
        print(pd.DataFrame(r["per_output"]).T.round(2).to_string())
    print("decision", primary["decision"], "stable", report["convergence"]["stable"])


if __name__ == "__main__":
    main()
