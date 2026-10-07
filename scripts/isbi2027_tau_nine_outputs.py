"""Probe agreement over the nine outputs reported in the paper (after amendment 12).

The adapted HCLD output was replaced by the blur control (amendment 12), so the paper's ranking
statistics use the nine remaining outputs scored by the original probes: Kendall tau between the
method orderings of benchmark-recipe probes (best-validation checkpoints), of converged probes
(final epoch), and between the frozen probe and each converged probe; per-output seed ranges; and
which outputs every converged probe scores above the frozen probe's 95% interval.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

EXCLUDED = ["adapted_hcld"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="analysis/all_runs_long.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    src = df[(df.group == "source_non_nyu") & (df.metric == "harmonized_site_ba") & ~df.method.isin(EXCLUDED)]
    probes = lambda family, ckpt: src[(src.family == family) & (src.source == "raw") & (src.ckpt == ckpt)].pivot_table(
        index="method", columns="seed", values="estimate")
    bench, conv = probes("retrained_site_probe", "best"), probes("converged_site_probe", "last")
    frozen = src[src.family == "frozen"].set_index("method")
    taus = lambda table: [kendalltau(table[a], table[b])[0] for a, b in itertools.combinations(table.columns, 2)]
    vs_frozen = [kendalltau(frozen.estimate.reindex(conv.index), conv[s])[0] for s in conv.columns]
    above = conv.gt(frozen.ci_high.reindex(conv.index), axis=0).all(axis=1)
    rng = lambda table: (table.max(axis=1) - table.min(axis=1))
    report = {
        "outputs": conv.index.tolist(), "excluded": EXCLUDED,
        "kendall_tau_benchmark_recipe": [round(min(taus(bench)), 3), round(max(taus(bench)), 3)],
        "kendall_tau_converged": [round(min(taus(conv)), 3), round(max(taus(conv)), 3)],
        "kendall_tau_frozen_vs_converged": [round(min(vs_frozen), 3), round(max(vs_frozen), 3)],
        "max_seed_range_benchmark_recipe": round(float(rng(bench).max()), 3),
        "max_seed_range_converged": round(float(rng(conv).max()), 3),
        "converged_all_above_frozen_interval": {m: bool(v) for m, v in above.items()},
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
