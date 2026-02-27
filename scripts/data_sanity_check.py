import os
import math
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

# --------- CONFIG ----------
ROOT = "../data/ABIDE/"  # set to your ABIDE folder root
PATTERN = (".nii", ".nii.gz")          # files to consider
ONLY_MPRAGE = False                    # set True if you only want "mprage.nii.gz"
MPRAGE_NAME = "mprage.nii.gz"

# Reasonable spacing bounds for T1 MRI in mm (broad on purpose)
SPACING_MIN_MM = 0.3
SPACING_MAX_MM = 3.5

# Intensity sanity thresholds
MIN_NONZERO_FRACTION = 0.001   # at least 0.1% voxels should be nonzero
MIN_FINITE_FRACTION = 0.99     # at least 99% should be finite

OUT_CSV = "sanity_check_1_report.csv"
# ---------------------------

def is_reasonable_spacing(zooms):
    if zooms is None or len(zooms) < 3:
        return False
    x, y, z = zooms[:3]
    for v in (x, y, z):
        if v is None or not np.isfinite(v) or v <= 0:
            return False
        if v < SPACING_MIN_MM or v > SPACING_MAX_MM:
            return False
    return True

def summarize_file(path):
    row = {
        "file": path,
        "status": "PASS",
        "reason": "",
        "shape": "",
        "zooms_mm": "",
        "dtype": "",
        "min": np.nan,
        "max": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "finite_frac": np.nan,
        "nonzero_frac": np.nan,
    }

    try:
        img = nib.load(path)
    except Exception as e:
        row["status"] = "FAIL"
        row["reason"] = f"nibabel load error: {type(e).__name__}: {e}"
        return row

    hdr = img.header
    shape = img.shape
    zooms = None
    try:
        zooms = hdr.get_zooms()
    except Exception:
        zooms = None

    row["shape"] = str(shape)
    row["zooms_mm"] = str(zooms[:3] if zooms else None)
    row["dtype"] = str(hdr.get_data_dtype())

    # --- Dimensionality checks ---
    # Expected: 3D. Allow 4D only if last dim is 1 (common odd export).
    if len(shape) == 3:
        pass
    elif len(shape) == 4 and shape[3] == 1:
        # treat as 3D by squeezing later
        pass
    else:
        row["status"] = "FAIL"
        row["reason"] = f"not 3D (got shape {shape}) — possible fMRI or wrong modality"
        return row

    # --- Spacing check ---
    if not is_reasonable_spacing(zooms):
        row["status"] = "FAIL"
        row["reason"] = f"unreasonable voxel spacing (zooms={zooms})"
        return row

    # --- Intensity checks ---
    # Use get_fdata for stable float conversion; try not to load huge data repeatedly.
    try:
        data = img.get_fdata(dtype=np.float32)
    except Exception as e:
        row["status"] = "FAIL"
        row["reason"] = f"get_fdata error: {type(e).__name__}: {e}"
        return row

    if data.ndim == 4 and data.shape[3] == 1:
        data = np.squeeze(data, axis=3)

    finite = np.isfinite(data)
    finite_frac = float(finite.mean()) if data.size else 0.0
    row["finite_frac"] = finite_frac

    if data.size == 0:
        row["status"] = "FAIL"
        row["reason"] = "empty array (size=0)"
        return row

    if finite_frac < MIN_FINITE_FRACTION:
        row["status"] = "FAIL"
        row["reason"] = f"too many NaN/Inf values (finite_frac={finite_frac:.4f})"
        return row

    # Compute stats on finite values only
    d = data[finite]
    row["min"] = float(np.min(d))
    row["max"] = float(np.max(d))
    row["mean"] = float(np.mean(d))
    row["std"] = float(np.std(d))

    nonzero_frac = float((d != 0).mean())
    row["nonzero_frac"] = nonzero_frac

    # Catch all-zero or near-empty intensity volumes
    if nonzero_frac < MIN_NONZERO_FRACTION:
        row["status"] = "FAIL"
        row["reason"] = f"nearly all zeros (nonzero_frac={nonzero_frac:.6f})"
        return row

    # Catch completely constant images (rare but possible corruption)
    if math.isclose(row["std"], 0.0, abs_tol=1e-8):
        row["status"] = "FAIL"
        row["reason"] = "constant intensity (std≈0) — possible corruption"
        return row

    return row

def collect_files(root):
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if ONLY_MPRAGE:
                if fn != MPRAGE_NAME:
                    continue
                hits.append(os.path.join(dirpath, fn))
            else:
                if fn.endswith(PATTERN):
                    hits.append(os.path.join(dirpath, fn))
    return sorted(hits)

def main():
    files = collect_files(ROOT)
    if not files:
        print(f"No NIfTI files found under: {ROOT} (ONLY_MPRAGE={ONLY_MPRAGE})")
        return

    rows = []
    for f in tqdm(files, desc="Sanity Check 1"):
        rows.append(summarize_file(f))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # Console summary
    total = len(df)
    passed = int((df["status"] == "PASS").sum())
    failed = total - passed
    print("\n=== Sanity Check 1 Summary ===")
    print(f"Total: {total} | PASS: {passed} | FAIL: {failed}")
    print(f"Saved report: {OUT_CSV}")

    if failed:
        print("\nTop failure reasons:")
        print(df[df["status"] == "FAIL"]["reason"].value_counts().head(10).to_string())

        print("\nExample failed files:")
        print(df[df["status"] == "FAIL"][["file", "shape", "zooms_mm", "reason"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()