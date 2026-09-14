"""Target-site (NYU) intensity alignment for a completed ISBI evaluation run.

Reuses the run's validated raw reference, artifacts and source bootstrap indices, so
intervals are paired with the main table. Reference: raw NYU training slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from harmonit.metrics.subject_evaluation import interval, target_alignment, target_reference


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-run", required=True, help="Directory containing COMPLETE.json")
    p.add_argument("--reference-npz", required=True, help="Train-split export with raw_images and site_ids")
    p.add_argument("--splits-path", required=True)
    p.add_argument("--target-site-id", type=int, default=5)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    run = Path(args.eval_run)
    out = Path(args.out_dir)
    if not (run / "COMPLETE.json").is_file():
        p.error("Evaluation run is incomplete")
    if out.exists():
        p.error("Output directory exists")
    protocol = json.loads((run / "protocol.json").read_text())
    splits = json.loads(Path(args.splits_path).read_text())

    with np.load(args.reference_npz, allow_pickle=False) as data:
        if str(np.asarray(data["split"]).item()) != "train":
            raise ValueError("Reference must come from the training split")
        subjects = data["subject_ids"].astype(str)
        if set(subjects) != set(splits["train"]):
            raise ValueError("Reference export does not cover the training split")
        target = data["site_ids"] == args.target_site_id
        reference = target_reference(data["raw_images"][target, 0])

    with np.load(run / "raw_reference.npz", allow_pickle=False) as data:
        raw = data["raw_images"][:, 0]
        test_subjects = data["subject_ids"].astype(str)
        sites = data["site_ids"]
    source = sites != args.target_site_id
    indices = np.load(run / "bootstrap_indices_source_non_nyu.npy")

    rows, summary = [], {}
    for spec in protocol["args"]["artifact"]:
        name, _, path = spec.partition("=")
        recorded = json.loads((run / f"{name}_summary.json").read_text())["artifact_sha256"]
        if sha256(path) != recorded:
            raise ValueError(f"{name}: artifact changed since the evaluation run")
        with np.load(path, allow_pickle=False) as data:
            images = data["images"][:, 0]
            if not np.array_equal(data["subject_ids"].astype(str), test_subjects):
                raise ValueError(f"{name}: subject order differs from the raw reference")
        frame = pd.DataFrame([dict(method=name, subject_id=test_subjects[i], site_id=int(sites[i]),
                                   **target_alignment(raw[i], images[i], reference))
                              for i in np.flatnonzero(source)])
        rows.append(frame)
        summary[name] = {}
        for metric in ("wasserstein", "kl"):
            raw_v = frame[f"target_{metric}_raw"].to_numpy()
            harm_v = frame[f"target_{metric}_harmonized"].to_numpy()
            delta = harm_v - raw_v
            summary[name][metric] = {
                "raw": interval(raw_v.mean(), raw_v[indices].mean(axis=1)),
                "harmonized": interval(harm_v.mean(), harm_v[indices].mean(axis=1)),
                "harmonized_minus_raw": interval(delta.mean(), delta[indices].mean(axis=1)),
            }

    out.mkdir(parents=True)
    pd.concat(rows).to_csv(out / "target_alignment_subjects.csv", index=False)
    (out / "target_alignment_summary.json").write_text(json.dumps({
        "eval_run": str(run), "reference_npz": args.reference_npz, "reference_sha256": sha256(args.reference_npz),
        "reference_subjects": reference["n_subjects"], "foreground": "raw slice > 0.02",
        "kl_direction": "KL(image || NYU reference)", "cohort": "source (non-NYU) test subjects",
        "methods": summary}, indent=2) + "\n")
    lines = ["| Method | W raw->NYU | W harm->NYU | dW | KL raw | KL harm | dKL |", "| --- | --- | --- | --- | --- | --- | --- |"]
    fmt = lambda v: f"{v['estimate']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"
    for name, s in summary.items():
        w, k = s["wasserstein"], s["kl"]
        lines.append(f"| {name} | {fmt(w['raw'])} | {fmt(w['harmonized'])} | {fmt(w['harmonized_minus_raw'])} | "
                     f"{fmt(k['raw'])} | {fmt(k['harmonized'])} | {fmt(k['harmonized_minus_raw'])} |")
    (out / "target_alignment.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
