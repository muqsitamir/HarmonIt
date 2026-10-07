"""Paired difference: slice probes trained on a method's outputs minus probes trained on raw slices.

Both are tested on the same outputs of that method (diffusion: second draw), on the source subjects.
Balanced accuracy is averaged over the probe seeds; the interval resamples subjects within site with
the evaluator's indices (2,000 replicates, seed 20260913), the same draws for every probe, so it is
paired and conditional on the trained checkpoints. Protocol amendment 9 (post hoc summary of
existing predictions; no probe was retrained). `--haca3` runs the same analysis on amendment 13's
whole-head slice probes (raw-trained and HACA3-trained, tested on HACA3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harmonit.metrics.subject_evaluation import balanced_accuracy_draws, stratified_indices  # noqa: E402

TARGET_SITE = 5
TESTS = {"histogram_matching": "histogram_matching", "cyclegan_tuned": "cyclegan_tuned",
         "diffusion_20k": "diffusion_20k_redraw"}  # probe training source -> matched test artifact
FAMILIES = {"whole_head": "sliceprobe_{source}_seed{seed}_model_{ckpt}_9methods",
            "brain_only": "brainprobe_{source}_seed{seed}_model_{ckpt}"}


def predictions(runs, pattern, source, seed, ckpt, artifact):
    frame = pd.read_csv(runs / pattern.format(source=source, seed=seed, ckpt=ckpt) / f"{artifact}_subjects.csv")
    return frame[frame.site_id != TARGET_SITE].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, help="results/isbi2027/runs")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--ckpt", choices=("best", "last"), default="best")
    p.add_argument("--haca3", action="store_true", help="Amendment 13 runs (results/isbi2027/haca3/runs)")
    args = p.parse_args()
    runs = Path(args.runs)
    families, tests = FAMILIES, TESTS
    if args.haca3:
        families = {"whole_head": "sliceprobe_{source}_seed{seed}_model_{ckpt}_haca3",
                    "brain_only": "brainprobe_{source}_seed{seed}_model_{ckpt}"}  # amendment 17
        families = {k: v for k, v in families.items()
                    if (runs / v.format(source="raw", seed=args.seeds[0], ckpt=args.ckpt)).is_dir()}
        tests = {"haca3": "haca3"}
    report = {"ckpt": args.ckpt, "seeds": args.seeds, "replicates": 2000, "bootstrap_seed": 20260913, "results": {}}
    for family, pattern in families.items():
        for source, artifact in tests.items():
            frames = {(kind, s): predictions(runs, pattern, kind_src, s, args.ckpt, artifact)
                      for kind, kind_src in (("raw", "raw"), ("output", source)) for s in args.seeds}
            ref = frames[("raw", args.seeds[0])]
            for frame in frames.values():
                assert (frame.subject_id == ref.subject_id).all()
            labels = ref.site_id.to_numpy()
            indices = stratified_indices(labels)
            ba = {k: balanced_accuracy_draws(labels, f.harmonized_prediction.to_numpy(), indices)
                  for k, f in frames.items()}
            mean = lambda kind: (np.mean([ba[(kind, s)][0] for s in args.seeds]),
                                 np.mean([ba[(kind, s)][1] for s in args.seeds], axis=0))
            (raw_est, raw_draws), (out_est, out_draws) = mean("raw"), mean("output")
            diff = out_draws - raw_draws
            report["results"][f"{family}/{source}"] = {
                "test_artifact": artifact, "n_subjects": int(len(labels)),
                "raw_trained_ba": round(float(raw_est), 4), "output_trained_ba": round(float(out_est), 4),
                "difference": round(float(out_est - raw_est), 4),
                "difference_ci95": [round(float(v), 4) for v in np.quantile(diff, [.025, .975])],
            }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for key, r in report["results"].items():
        print(f"{key:32s} raw {r['raw_trained_ba']:.2f}  output {r['output_trained_ba']:.2f}  "
              f"diff {r['difference']:+.2f} {r['difference_ci95']}")


if __name__ == "__main__":
    main()
