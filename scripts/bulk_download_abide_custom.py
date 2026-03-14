import os
from pathlib import Path
from tqdm import tqdm
import xnat
from dotenv import load_dotenv

"""
Bulk-download ABIDE T1 (MPRAGE) NIfTI from NITRC XNAT.

What it does:
- Connects to https://www.nitrc.org/ir
- Iterates subjects in project "ABIDE"
- Finds MR sessions
- Selects scans likely to be T1 (tries common labels)
- Downloads NIfTI resources (e.g., mprage.nii.gz) when available

Outputs (example):
data/ABIDE/CMU_50642/scans/anat-unknown/resources/NIfTI/files/mprage.nii.gz
"""

# Heuristics: scan IDs / descriptions that usually correspond to T1
T1_KEYWORDS = [
    "mprage", "t1", "t1w", "spgr", "anat", "sag", "bravo"
]
# Keywords that indicate we should avoid (fMRI, etc.)
AVOID_KEYWORDS = [
    "rest", "bold", "fmri", "epi", "func"
]

load_dotenv()

def is_likely_t1(scan) -> bool:
    text = " ".join([
        str(getattr(scan, "type", "") or ""),
        str(getattr(scan, "series_description", "") or ""),
        str(getattr(scan, "quality", "") or ""),
        str(getattr(scan, "note", "") or ""),
        str(getattr(scan, "id", "") or ""),
        str(getattr(scan, "label", "") or "")
    ]).lower()

    if any(k in text for k in AVOID_KEYWORDS):
        return False
    return any(k in text for k in T1_KEYWORDS)

def main():
    xnat_url = os.getenv("XNAT_URL")
    user = os.getenv("XNAT_USER")
    pw = os.getenv("XNAT_PASS")

    out_root = Path(os.environ.get("ABIDE_OUT", "../data/ABIDE")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    max_subjects = int(os.environ.get("ABIDE_MAX_SUBJECTS", "0"))  # 0 = all
    start_at = int(os.environ.get("ABIDE_START_AT", "0"))          # for resume
    verbose = os.environ.get("ABIDE_VERBOSE", "0") == "1"

    with xnat.connect(xnat_url, user=user, password=pw) as session:
        proj = session.projects["ABIDE"]

        subjects = list(proj.subjects.values())
        subjects = subjects[start_at:]
        if max_subjects > 0:
            subjects = subjects[:max_subjects]

        print(f"Connected to {xnat_url}")
        print(f"Project: ABIDE | Subjects to process: {len(subjects)}")
        print(f"Output root: {out_root}")

        for subj in tqdm(subjects, desc="Subjects"):
            subj_label = subj.label  # e.g., CMU_50642
            subj_dir = out_root / subj_label

            try:
                experiments = list(subj.experiments.values())
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed listing experiments for {subj_label}: {e}")
                continue

            # Keep only MR sessions (raw)
            mr_sessions = [e for e in experiments if "MR" in (e.__class__.__name__.upper()) or "MR" in (getattr(e, "modality", "") or "").upper()]
            if not mr_sessions:
                # Some XNAT instances name them differently; fallback: keep those with scans
                mr_sessions = [e for e in experiments if hasattr(e, "scans")]

            for sess in mr_sessions:
                try:
                    scans = list(sess.scans.values())
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Failed listing scans for {subj_label}: {e}")
                    continue

                # Pick T1 scans by heuristic
                t1_scans = [sc for sc in scans if is_likely_t1(sc)]
                if not t1_scans:
                    # fallback: sometimes XNAT uses 'anat' as scan id
                    t1_scans = [sc for sc in scans if "anat" in (str(getattr(sc, "id", "")) + str(getattr(sc, "type", ""))).lower()]

                for sc in t1_scans:
                    try:
                        resources = list(sc.resources.values())
                    except Exception:
                        continue

                    # Prefer NIfTI resource
                    nifti_res = None
                    for r in resources:
                        if str(r.label).lower() == "nifti":
                            nifti_res = r
                            break
                    if nifti_res is None:
                        continue

                    try:
                        files = list(nifti_res.files.values())
                    except Exception:
                        continue

                    # Prefer mprage.nii.gz if present, else download all .nii/.nii.gz
                    preferred = [f for f in files if "mprage" in f.name.lower() and (f.name.endswith(".nii") or f.name.endswith(".nii.gz"))]
                    candidates = preferred if preferred else [f for f in files if f.name.endswith(".nii") or f.name.endswith(".nii.gz")]
                    if not candidates:
                        continue

                    # Mirror XNAT-ish folder structure you observed
                    # Example: <subj>/scans/<scan_label>/resources/NIfTI/files/<filename>
                    scan_label = str(getattr(sc, "label", "") or getattr(sc, "id", "scan")).strip()
                    if scan_label == "":
                        scan_label = str(getattr(sc, "id", "scan"))

                    base_dir = subj_dir / "scans" / scan_label / "resources" / "NIfTI" / "files"
                    base_dir.mkdir(parents=True, exist_ok=True)

                    for f in candidates:
                        out_path = base_dir / f.name
                        if out_path.exists() and out_path.stat().st_size > 0:
                            continue  # already downloaded

                        try:
                            # xnatpy supports download to file-like or path
                            f.download(str(out_path))
                        except Exception as e:
                            if verbose:
                                print(f"[WARN] Download failed {subj_label} {scan_label} {f.name}: {e}")

if __name__ == "__main__":
    main()