"""Predict HD-BET brain masks for every raw ABIDE T1 volume (protocol amendment 7).

Run with the separate HD-BET environment (hd-bet 2.0.1). Masks are written as
<out-dir>/<subject_id>.nii.gz in each volume's original space; existing masks are skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from HD_BET.checkpoint_download import maybe_download_parameters
from HD_BET.hd_bet_prediction import get_hdbet_predictor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-repo", required=True, help="Directory containing data/abide_manifest.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chunk", type=int, default=50)
    args = p.parse_args()
    repo, out = Path(args.data_repo), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(repo / "data/abide_manifest.csv")
    todo = [(r.subject_id, repo / r.t1_path) for r in manifest.itertuples()
            if not (out / f"{r.subject_id}.nii.gz").exists()]
    print(f"{len(manifest)} subjects, {len(todo)} to predict", flush=True)
    maybe_download_parameters()
    predictor = get_hdbet_predictor(use_tta=True, device=torch.device("cuda"))
    start = time.time()
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i:i + args.chunk]
        predictor.predict_from_files([[str(path)] for _, path in chunk], [str(out / sid) for sid, _ in chunk],
                                     save_probabilities=False, overwrite=False, num_processes_preprocessing=3,
                                     num_processes_segmentation_export=3)
        done = i + len(chunk)
        print(f"{done}/{len(todo)} masks, {(time.time() - start) / done:.1f} s per volume", flush=True)
    missing = [r.subject_id for r in manifest.itertuples() if not (out / f"{r.subject_id}.nii.gz").exists()]
    print(f"finished; missing masks: {missing}", flush=True)


if __name__ == "__main__":
    main()
