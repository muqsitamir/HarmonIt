"""Probe seed variability and ranking agreement (protocol amendment 2).

For source (non-NYU) harmonized site BA: per-output spread across retrained site-probe
seeds, and Kendall tau between the method orderings produced by every pair of probes
(frozen, each retrained seed's best and final checkpoints).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    sel = df[(df.group == "source_non_nyu") & (df.metric == "harmonized_site_ba")
             & (df.family.isin(["frozen", "retrained_site_probe"])) & (df.source == "raw")]
    sel = sel.assign(probe=np.where(sel.family == "frozen", "frozen",
                                    "seed" + sel.seed.fillna(-1).astype(int).astype(str) + "_" + sel.ckpt))
    wide = sel.pivot_table(index="method", columns="probe", values="estimate")
    wide = wide.dropna(axis=0, how="any")  # outputs evaluated by every probe

    pairs = []
    for a, b in itertools.combinations(wide.columns, 2):
        tau, pval = kendalltau(wide[a], wide[b])
        pairs.append(dict(probe_a=a, probe_b=b, kendall_tau=float(tau), p_value=float(pval)))
    pairs = pd.DataFrame(pairs)

    best = [c for c in wide.columns if c.endswith("_best")]
    spread = pd.DataFrame({
        "frozen": wide.get("frozen"),
        "retrained_best_mean": wide[best].mean(axis=1), "retrained_best_min": wide[best].min(axis=1),
        "retrained_best_max": wide[best].max(axis=1), "n_seeds": len(best),
    })
    raw = df[(df.group == "source_non_nyu") & (df.metric == "raw_site_ba") & (df.method == "neurocombat")
             & (df.family.isin(["frozen", "retrained_site_probe"])) & (df.source == "raw")]
    summary = {
        "outputs": wide.index.tolist(), "probes": wide.columns.tolist(),
        "kendall_tau_best_vs_best": pairs[pairs.probe_a.isin(best) & pairs.probe_b.isin(best)].kendall_tau.describe().to_dict(),
        "kendall_tau_frozen_vs_best": pairs[(pairs.probe_a == "frozen") & pairs.probe_b.isin(best)].kendall_tau.describe().to_dict(),
        "raw_ba_by_probe": {f"{r.family}_{r.seed}_{r.ckpt}": r.estimate for r in raw.itertuples()},
        "spread": spread.round(4).to_dict(orient="index"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out / "source_ba_by_probe.csv")
    pairs.to_csv(out / "kendall_tau_pairs.csv", index=False)
    (out / "probe_agreement.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(spread.round(3).to_string())
    print(json.dumps({k: summary[k] for k in ("kendall_tau_best_vs_best", "kendall_tau_frozen_vs_best")}, indent=2))


if __name__ == "__main__":
    main()
