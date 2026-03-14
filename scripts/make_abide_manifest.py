import csv
from pathlib import Path

def parse_subject_id(path: Path) -> str:
    # data/ABIDE/<SUBJECT>/scans/.../(mprage.nii.gz | hires.nii.gz)
    parts = path.parts
    # find index of 'ABIDE' and take next
    if "ABIDE" in parts:
        i = parts.index("ABIDE")
        return parts[i + 1]
    # fallback: parent 5 levels up (best-effort)
    return path.parents[5].name

def parse_site(subject_id: str) -> str:
    return subject_id.split("_")[0] if "_" in subject_id else subject_id

def parse_scan_label(path: Path) -> str:
    # .../<SUBJECT>/scans/<SCAN_LABEL>/resources/NIfTI/files/(mprage.nii.gz | hires.nii.gz)
    parts = path.parts
    if "scans" in parts:
        i = parts.index("scans")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""

def main():
    # Default paths assume this script is run from the repo root.
    root = Path("data/ABIDE").resolve()
    out_csv = Path("data/abide_manifest.csv").resolve()

    # ABIDE T1 volumes are typically `mprage.nii.gz`, but a small subset (e.g., some UCLA subjects)
    # may be stored as `hires.nii.gz`. Include both.
    patterns = ["mprage.nii.gz", "hires.nii.gz"]
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))

    # Sort and de-duplicate (Path objects are hashable)
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No T1 NIfTI files ({', '.join(patterns)}) found under {root}")

    rows = []
    for f in files:
        subject_id = parse_subject_id(f)
        site = parse_site(subject_id)
        scan_label = parse_scan_label(f)
        rows.append({
            "subject_id": subject_id,
            "site": site,
            "scan_label": scan_label,
            "t1_path": str(f),
            "size_bytes": f.stat().st_size
        })

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_csv}")

    # Quick summary: counts per site
    from collections import Counter
    c = Counter(r["site"] for r in rows)
    print("Counts per site:")
    for site, n in sorted(c.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {site}: {n}")

if __name__ == "__main__":
    main()