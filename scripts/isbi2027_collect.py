"""Collect every complete ISBI 2027 evaluation run into one long table for the paper.

Probe identity is parsed from the run directory name:
  frozen_probe_v1_9methods_*                      -> frozen production probe
  retrained_{raw|shuffle}_seed{S}[_{ckpt}]_*      -> site probe retrained on raw volumes
  sliceprobe_{source}_seed{S}_{ckpt}_*            -> fixed-slice probe trained on raw or method outputs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PATTERNS = [
    (re.compile(r"^frozen_probe_v1_9methods_"), lambda m: dict(family="frozen", source="raw", seed=None, ckpt="best")),
    (re.compile(r"^retrained_(raw|shuffle)_seed(\d+)(?:_(model_best|model_last))?_9methods_"),
     lambda m: dict(family="retrained_site_probe", source=m[1], seed=int(m[2]),
                    ckpt=(m[3] or "model_best").replace("model_", ""))),
    (re.compile(r"^sliceprobe_(.+)_seed(\d+)_(model_best|model_last)_9methods"),
     lambda m: dict(family="slice_probe", source=m[1], seed=int(m[2]), ckpt=m[3].replace("model_", ""))),
]


def parse(name):
    for pattern, build in PATTERNS:
        match = pattern.match(name)
        if match:
            return build(match)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = []
    for run in sorted(Path(args.runs).iterdir()):
        info = parse(run.name)
        if info is None or not (run / "COMPLETE.json").is_file():
            continue
        for method in json.loads((run / "COMPLETE.json").read_text())["methods"]:
            summary = json.loads((run / f"{method}_summary.json").read_text())
            for group, entry in summary["groups"].items():
                for metric, value in entry["metrics"].items():
                    ci = value.get("ci95") or [None, None]
                    rows.append(dict(run=run.name, **info, method=method, group=group, n=entry["n"],
                                     exact_identities=entry["exact_identities"], metric=metric,
                                     estimate=value["estimate"], ci_low=ci[0], ci_high=ci[1],
                                     finite_only_mean=value.get("finite_only_mean")))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    print(frame.groupby(["family", "source", "ckpt"]).run.nunique().to_string())


if __name__ == "__main__":
    main()
