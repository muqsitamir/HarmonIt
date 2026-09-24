"""Within-site acquisition heterogeneity of the ABIDE I cohort (protocol amendment 8).

ABIDE I gives no per-subject scanner identifier, and our 17 labels merge the released
sub-samples (UM_1/UM_2, UCLA_1/UCLA_2, Leuven_1/Leuven_2). Voxel size and matrix shape
read from the NIfTI headers bound how much acquisition variation a single site label
can hide. Writes a per-subject CSV and a per-site summary.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import nibabel as nib
import pandas as pd


def geometry(path):
    header = nib.load(str(path)).header
    zooms = tuple(round(float(z), 3) for z in header.get_zooms()[:3])
    shape = tuple(int(s) for s in header.get_data_shape()[:3])
    return zooms, shape


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--splits-path", required=True)
    p.add_argument("--data-root", default=".", help="Directory that manifest t1_path entries are relative to")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest_path)
    splits = json.loads(Path(args.splits_path).read_text())
    where = {s: name for name in ("train", "val", "test") for s in splits[name]}
    root = Path(args.data_root)

    rows = []
    for record in manifest.itertuples():
        path = Path(record.t1_path)
        if not path.exists():
            path = root / record.t1_path
        zooms, shape = geometry(path)
        rows.append(dict(subject_id=record.subject_id, site=record.site, split=where.get(record.subject_id),
                         scan_label=record.scan_label, voxel_mm=" x ".join(f"{z:g}" for z in zooms),
                         matrix=" x ".join(str(s) for s in shape),
                         min_voxel_mm=min(zooms), max_voxel_mm=max(zooms)))
    frame = pd.DataFrame(rows)
    frame["geometry"] = frame.voxel_mm + " mm, " + frame.matrix

    summary = {}
    for site, part in frame.groupby("site"):
        counts = collections.Counter(part.geometry)
        summary[site] = {
            "n": len(part), "distinct_geometries": len(counts),
            "dominant": counts.most_common(1)[0][0],
            "dominant_fraction": round(counts.most_common(1)[0][1] / len(part), 3),
            "geometries": [{"geometry": g, "n": n} for g, n in counts.most_common()],
        }
    heterogeneous = sorted(s for s, v in summary.items() if v["distinct_geometries"] > 1)
    thick = frame[frame.max_voxel_mm >= 3]
    report = {
        "n_subjects": len(frame), "n_sites": frame.site.nunique(),
        "sites_with_multiple_geometries": len(heterogeneous), "heterogeneous_sites": heterogeneous,
        "non_isotropic_or_thick_slice": {
            "criterion": "largest voxel dimension >= 3 mm",
            "n": len(thick), "subjects": thick.subject_id.tolist(),
            "by_split": thick.split.value_counts().to_dict(), "sites": sorted(thick.site.unique().tolist()),
        },
        "note": ("ABIDE I releases UM, UCLA and Leuven as two sub-samples each; the manifest merges them, "
                 "so one label can cover more than one acquisition protocol."),
        "per_site": summary,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["geometry"]).to_csv(out / "acquisition_per_subject.csv", index=False)
    (out / "acquisition_heterogeneity.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"{len(frame)} subjects, {frame.site.nunique()} sites; "
          f"{len(heterogeneous)} sites with more than one acquisition geometry")
    for site, v in sorted(summary.items(), key=lambda kv: -kv[1]["distinct_geometries"]):
        print(f"  {site:10s} n={v['n']:4d} distinct={v['distinct_geometries']:3d}  "
              f"dominant {v['dominant']} ({v['dominant_fraction']:.0%})")
    print(f"thick-slice/anisotropic (>= 3 mm): {len(thick)} subjects {report['non_isotropic_or_thick_slice']['by_split']}")


if __name__ == "__main__":
    main()
