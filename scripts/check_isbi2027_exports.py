"""Gate harmonized-probe training on export integrity (protocol amendment 2).

1. Each method's test re-export reproduces its historical test artifact.
2. Train/val exports cover the split, and embedded raw slices, subjects and slice indices
   are identical across methods.
Writes a JSON report; exits nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NPZ = {"histogram_matching": "histogram_matching_slices.npz", "cyclegan_tuned": "cyclegan_nyu_slices.npz",
       "diffusion_20k": "diffusion_img2img_nyu_slices.npz"}


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in ("images", "raw_images", "subject_ids", "site_ids", "slice_indices")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports", required=True)
    p.add_argument("--historical", action="append", required=True, help="method=historical test NPZ")
    p.add_argument("--splits-path", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()
    exports, splits = Path(args.exports), json.loads(Path(args.splits_path).read_text())
    report, ok = {"test_reproduction": {}, "train_val": {}}, True

    for spec in args.historical:
        name, _, path = spec.partition("=")
        old, new = load(path), load(exports / name / "test" / NPZ[name])
        diff = np.abs(old["images"] - new["images"]).reshape(len(old["images"]), -1).max(1)
        same_ids = all(np.array_equal(old[k], new[k]) for k in ("subject_ids", "site_ids", "slice_indices"))
        passed = bool(same_ids and diff.max() <= args.atol)
        report["test_reproduction"][name] = dict(passed=passed, ids_match=same_ids, max_abs_diff=float(diff.max()),
                                                 subjects_over_atol=int((diff > args.atol).sum()))
        ok &= passed

    for split in ("val", "train"):
        first = None
        for name, npz in NPZ.items():
            path = exports / name / split / npz
            if not path.is_file():
                report["train_val"][f"{name}/{split}"] = dict(passed=False, reason="missing")
                ok = False
                continue
            data = load(path)
            covered = set(data["subject_ids"].astype(str)) == set(splits[split]) and \
                len(data["subject_ids"]) == len(splits[split])
            finite = bool(np.isfinite(data["images"]).all())
            if first is None:
                first, match = data, True
            else:
                match = all(np.array_equal(first[k], data[k]) for k in ("subject_ids", "site_ids", "slice_indices")) \
                    and np.allclose(first["raw_images"], data["raw_images"], atol=args.atol)
            passed = bool(covered and finite and match)
            report["train_val"][f"{name}/{split}"] = dict(passed=passed, covers_split=covered, finite=finite,
                                                          raw_matches_first_method=bool(match), n=len(data["images"]))
            ok &= passed

    report["all_passed"] = bool(ok)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
