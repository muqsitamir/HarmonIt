"""Evaluate frozen slice artifacts with paired subject-level ISBI summaries.

The fresh reference is built once for all methods. This never trains a model,
modifies old artifacts, or interprets raw/harmonized distances as NYU alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import eval_harmonized_npz as legacy
from harmonit.metrics.subject_evaluation import (
    METRICS, balanced_accuracy_draws, interval, pixel_metrics,
    stratified_indices, summarize_subjects,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cohort(manifest, splits, split):
    if manifest.subject_id.duplicated().any():
        raise ValueError("Manifest must contain one row per subject")
    groups = []
    for name in ("train", "val", "test"):
        ids = splits[name]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate subjects in {name}")
        groups.append(set(ids))
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError("Subject splits overlap")
    if set.union(*groups) != set(manifest.subject_id):
        raise ValueError("Manifest and split subject sets differ")
    return set(splits[split])


def summarize_frame(frame, indices):
    result = {m: summarize_subjects(frame[m].to_numpy(), indices) for m in METRICS}
    y = frame.site_id.to_numpy()
    raw, raw_draws = balanced_accuracy_draws(y, frame.raw_prediction.to_numpy(), indices)
    harm, harm_draws = balanced_accuracy_draws(y, frame.harmonized_prediction.to_numpy(), indices)
    result.update(raw_site_ba=interval(raw, raw_draws), harmonized_site_ba=interval(harm, harm_draws),
                  site_ba_drop=interval(raw - harm, raw_draws - harm_draws))
    return result, harm_draws


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", action="append", required=True, help="Unique label=NPZ path; repeat per method")
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--splits-path", required=True)
    p.add_argument("--site-probe-ckpt", required=True)
    p.add_argument("--out-dir", required=True, help="New output directory; existing directories are rejected")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--target-site-id", type=int, default=5)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20260913)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--slice-map", help="Frozen subject->slice JSON the reference must reproduce")
    p.add_argument("--reference", choices=("fresh", "embedded"), default="fresh",
                   help="Embedded is a diagnostic only, not a verified benchmark")
    p.add_argument("--probe-input-mask", help="NPZ of per-subject masks multiplied into raw and harmonized probe "
                   "inputs only (brain-only control, amendment 7); pixel metrics are unchanged")
    args = p.parse_args()
    torch.set_num_threads(4)
    legacy.set_seed(42)
    out = Path(args.out_dir)
    if out.exists():
        p.error("Output directory already exists; choose a new run directory")
    artifacts = {}
    for spec in args.artifact:
        name, sep, path = spec.partition("=")
        if not sep or not name or name in artifacts or "/" in name or name in (".", ".."):
            p.error("Artifacts need unique simple names and paths: label=path")
        artifacts[name] = Path(path).resolve()
    if not Path(args.site_probe_ckpt).is_file():
        p.error("Frozen probe checkpoint is required")
    manifest = pd.read_csv(args.manifest_path)
    splits = json.loads(Path(args.splits_path).read_text())
    expected_subjects = validate_cohort(manifest, splits, args.split)
    first = legacy.load_npz_artifact(next(iter(artifacts.values())))
    if args.reference == "fresh":
        ds_args = argparse.Namespace(manifest_path=args.manifest_path, splits_path=args.splits_path,
            valid_nonzero_frac=.02, fg_bbox_thr=.02, seed=42, volume_cache_size=2)
        dataset = legacy.build_raw_dataset(ds_args, args.split, (256, 256))
        raw, subjects, sites, slices = legacy.extract_raw_fixed_slices(dataset, args.batch_size, args.num_workers, (256, 256))
    else:
        raw = first["raw_images"]
        subjects, sites, slices = (first[k] for k in ("subject_ids", "site_ids", "slice_indices"))
    if len(set(subjects)) != len(subjects) or set(subjects) != expected_subjects:
        raise ValueError("Evaluation cohort is duplicated, incomplete, or differs from split")
    if args.slice_map:
        # Fixed-slice selection has exact foreground ties; argsort may break them per host.
        frozen = json.loads(Path(args.slice_map).read_text())
        flipped = {s: (frozen.get(s), int(k)) for s, k in zip(subjects, slices) if frozen.get(s) != int(k)}
        if flipped:
            raise ValueError(f"Reference slices differ from frozen map (frozen, loaded): {flipped}")
    if "site_id" not in manifest:
        manifest["site_id"] = manifest.site.map({s: i for i, s in enumerate(sorted(manifest.site.unique()))})
    if not np.array_equal(manifest.set_index("subject_id").loc[subjects, "site_id"].to_numpy(), sites):
        raise ValueError("Reference site IDs differ from manifest")
    target_names = manifest.loc[manifest.site_id == args.target_site_id, "site"].unique().tolist()
    if target_names != ["NYU"]:
        raise ValueError(f"Protocol requires NYU target; got {target_names}")

    probe_mask = None
    if args.probe_input_mask:
        with np.load(args.probe_input_mask, allow_pickle=False) as data:
            order = {s: i for i, s in enumerate(data["subject_ids"].astype(str))}
            if set(order) != set(subjects):
                raise ValueError("Probe input mask subjects differ from the evaluation cohort")
            rows = [order[s] for s in subjects]
            if not np.array_equal(data["slice_indices"][rows], slices):
                raise ValueError("Probe input mask slices differ from the reference")
            probe_mask = data["masks"][rows].astype(np.float32)

    out.mkdir(parents=True)
    np.savez_compressed(out / "raw_reference.npz", images=raw, raw_images=raw,
                        subject_ids=subjects, site_ids=sites, slice_indices=slices,
                        split=np.asarray(args.split), method=np.asarray("raw_reference"))
    repo = Path(__file__).resolve().parents[1]
    sources = list((repo / "src").rglob("*.py")) + [Path(__file__), repo / "scripts/eval_harmonized_npz.py"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    provenance = {
        "protocol": "isbi2027_subject_v1", "reference": args.reference,
        "scope": "whole preprocessed image; raw/harmonized change, not NYU alignment",
        "psnr_data_range": 1.0, "distribution_aggregation": "mean of per-subject distances",
        "histogram": "50 fixed [0,1] bins plus underflow/overflow; probability smoothing 1e-8",
        "bootstrap": "paired site-stratified subject percentile intervals; fixed checkpoints and site counts",
        "args": vars(args), "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__,
        "repository_context_commit": git.stdout.strip() if git.returncode == 0 else None,
        "code_provenance": "source_sha256 identifies executed code; repository context may differ for deployed snapshots",
        "source_sha256": {str(s.relative_to(repo)): sha256(s) for s in sources},
        "inputs": {"manifest": sha256(args.manifest_path), "splits": sha256(args.splits_path),
                   "probe": sha256(args.site_probe_ckpt), "raw_reference": sha256(out / "raw_reference.npz"),
                   "slice_map": sha256(args.slice_map) if args.slice_map else None,
                   "probe_input_mask": sha256(args.probe_input_mask) if args.probe_input_mask else None},
    }
    (out / "protocol.json").write_text(json.dumps(provenance, indent=2) + "\n")
    masks = {"all": np.ones(len(sites), dtype=bool), "source_non_nyu": sites != args.target_site_id,
             "target_nyu": sites == args.target_site_id}
    bootstrap = {g: stratified_indices(sites[mask], args.bootstrap_replicates, args.bootstrap_seed)
                 for g, mask in masks.items() if mask.any()}
    for g, indices in bootstrap.items():
        np.save(out / f"bootstrap_indices_{g}.npy", indices)
    summaries, frames, ba_draws, flat = {}, {}, {}, []
    raw_predictions = None
    for name, path in artifacts.items():
        print(f"Evaluating {name}: {path}", flush=True)
        artifact = legacy.load_npz_artifact(path)
        if artifact["split"] != args.split:
            raise ValueError(f"{name}: missing or mismatched split metadata")
        legacy.assert_artifact_matches_raw(artifact, raw, subjects, sites, slices, (256, 256))
        probe_raw, probe_harm = raw, artifact["images"]
        if probe_mask is not None:
            probe_raw, probe_harm = raw * probe_mask, artifact["images"] * probe_mask
        preds = legacy.evaluate_site_probe(probe_raw, probe_harm, sites, Path(args.site_probe_ckpt),
                                           args.batch_size, include_predictions=True)
        if raw_predictions is not None and not np.array_equal(raw_predictions, preds["raw_predictions"]):
            raise ValueError("Raw probe predictions changed across methods")
        raw_predictions = preds["raw_predictions"]
        rows = []
        for i, (a, b) in enumerate(zip(raw, artifact["images"])):
            rows.append(dict(subject_id=subjects[i], site_id=int(sites[i]), slice_idx=int(slices[i]),
                raw_prediction=int(raw_predictions[i]), harmonized_prediction=int(preds["harmonized_predictions"][i]),
                **pixel_metrics(a, b)))
        frame = pd.DataFrame(rows)
        frames[name] = frame
        frame.to_csv(out / f"{name}_subjects.csv", index=False)
        device = preds["device"]
        if device.startswith("cuda"):
            device = f"{device} ({torch.cuda.get_device_name(0)})"
        entry = {"artifact": str(path), "artifact_sha256": sha256(path), "probe_device": device, "groups": {}}
        for group, indices in bootstrap.items():
            subset = frame.loc[masks[group]].reset_index(drop=True)
            metrics, draws = summarize_frame(subset, indices)
            ba_draws[name, group] = draws
            entry["groups"][group] = {
                "n": len(subset), "sites": sorted(subset.site_id.unique().tolist()),
                "site_counts": {str(k): int(v) for k, v in subset.site_id.value_counts().items()},
                "exact_identities": int(subset.exact_identity.sum()), "near_identities_atol_1e_minus6": int(subset.near_identity.sum()),
                "metrics": metrics,
            }
            for metric, value in metrics.items():
                flat.append(dict(method=name, group=group, metric=metric, estimate=value["estimate"],
                    ci_low=value["ci95"][0] if value["ci95"] else None,
                    ci_high=value["ci95"][1] if value["ci95"] else None, n=len(subset)))
        summaries[name] = entry
        (out / f"{name}_summary.json").write_text(json.dumps(entry, indent=2, allow_nan=False) + "\n")
        print(f"Completed {name}", flush=True)

    differences = []
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            group = "source_non_nyu"
            idx = bootstrap[group]
            fa, fb = frames[a].loc[masks[group]], frames[b].loc[masks[group]]
            for metric in METRICS:
                av, bv = fa[metric].to_numpy(), fb[metric].to_numpy()
                if np.isfinite(av).all() and np.isfinite(bv).all():
                    values = av - bv
                    item = interval(values.mean(), values[idx].mean(axis=1))
                    differences.append(dict(method_a=a, method_b=b, metric=metric, **item))
            point = summaries[a]["groups"][group]["metrics"]["harmonized_site_ba"]["estimate"] - summaries[b]["groups"][group]["metrics"]["harmonized_site_ba"]["estimate"]
            differences.append(dict(method_a=a, method_b=b, metric="harmonized_site_ba",
                **interval(point, ba_draws[a, group] - ba_draws[b, group])))
    (out / "paired_source_differences.json").write_text(json.dumps(differences, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(flat).to_csv(out / "metrics_long.csv", index=False)
    lines = [f"# ISBI evaluation: probe {args.site_probe_ckpt}", "",
        f"Reference: {args.reference}. Whole-image metrics; NYU excluded from this table.",
        "95% paired site-stratified subject bootstrap intervals, conditional on the saved checkpoints.",
        "Wasserstein/KL below are mean per-subject raw-to-harmonized distances, not historical pooled distances or NYU alignment.", "",
        "| Method | Source BA | BA drop | PSNR | Pixel cosine | XCorr | Subject W | Subject KL |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    cols = ("harmonized_site_ba", "site_ba_drop", "psnr", "pixel_cosine_similarity", "cross_correlation", "subject_wasserstein_raw_harm", "subject_kl_raw_harm")
    for name, entry in summaries.items():
        values = []
        for metric in cols:
            v = entry["groups"]["source_non_nyu"]["metrics"][metric]
            values.append(f'{v["estimate"]:.4f} [{v["ci95"][0]:.4f}, {v["ci95"][1]:.4f}]' if v["estimate"] is not None and v["ci95"] else "undefined (see counts)")
        lines.append("| " + " | ".join([name] + values) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "COMPLETE.json").write_text(json.dumps({"methods": names, "subjects": len(subjects)}) + "\n")
    print(f"Complete: {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
