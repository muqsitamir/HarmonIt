"""Prepare ABIDE volumes for the official HCLD codebase.

The HCLD repository expects normalized 3D ``.npy`` volumes plus TSV files whose
first column is a filename stem. This adapter exports the current ABIDE
manifest/splits into that format while preserving the deterministic site-id
mapping used by HarmonIt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from harmonit.data.abide_slices_dataset import robust_normalize


def parse_shape(values: Iterable[int]) -> tuple[int, int, int]:
    values = tuple(int(v) for v in values)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("--out-shape expects D H W")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ABIDE NIfTI volumes to HCLD npy/tsv format.")
    parser.add_argument("--manifest-path", default="data/abide_manifest.csv")
    parser.add_argument("--splits-path", default="data/splits.json")
    parser.add_argument("--out-dir", default="outputs/hcld_abide")
    parser.add_argument("--out-shape", nargs=3, type=int, default=(192, 192, 64), help="D H W volume shape.")
    parser.add_argument("--target-site-id", type=int, default=5, help="Default 5 = NYU.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_t1_path(path_value: str, data_root: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    parts = path.parts
    if "ABIDE" in parts:
        fallback = data_root / Path(*parts[parts.index("ABIDE") :])
        if fallback.exists():
            return fallback
    return path


def resize_volume_to_dhw(volume_xyz: np.ndarray, out_shape: tuple[int, int, int]) -> np.ndarray:
    # Convert from nibabel's spatial XYZ array to a CNN-friendly D,H,W tensor.
    volume_dhw = np.transpose(volume_xyz, (2, 0, 1)).astype(np.float32, copy=False)
    tensor = torch.from_numpy(volume_dhw).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=out_shape, mode="trilinear", align_corners=False)
    return tensor[0, 0].cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    args = build_arg_parser().parse_args()
    out_shape = parse_shape(args.out_shape)
    manifest_path = Path(args.manifest_path)
    splits = json.loads(Path(args.splits_path).read_text())
    data_root = manifest_path.parent

    manifest = pd.read_csv(manifest_path)
    if "site_id" not in manifest.columns:
        site_map = {site: idx for idx, site in enumerate(sorted(manifest["site"].unique()))}
        manifest["site_id"] = manifest["site"].map(site_map).astype(int)
    manifest = manifest.sort_values(["site", "subject_id"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    volume_dir = out_dir / "volumes"
    label_dir = out_dir / "labels"
    volume_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    split_names = [name for name in ("train", "val", "test") if isinstance(splits.get(name), list)]
    rows_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in split_names}
    split_by_subject = {subject_id: split for split in split_names for subject_id in splits[split]}

    for idx, row in manifest.iterrows():
        subject_id = str(row["subject_id"])
        split = split_by_subject.get(subject_id)
        if split is None:
            continue
        filename = f"{subject_id}_site{int(row['site_id'])}"
        out_path = volume_dir / f"{filename}.npy"
        if args.overwrite or not out_path.exists():
            t1_path = resolve_t1_path(str(row["t1_path"]), data_root)
            image = nib.as_closest_canonical(nib.load(str(t1_path)))
            volume = robust_normalize(image.get_fdata(dtype=np.float32))
            volume = resize_volume_to_dhw(volume, out_shape)
            np.save(out_path, volume[None])
        rows_by_split[split].append(
            {
                "filename": filename,
                "site": int(row["site_id"]),
                "subject_id": subject_id,
                "site_name": str(row["site"]),
            }
        )
        if (idx + 1) % 50 == 0:
            print(f"processed={idx + 1} saved={out_path}", flush=True)

    metadata = {
        "target_site_id": int(args.target_site_id),
        "target_site_name": str(manifest.loc[manifest["site_id"] == args.target_site_id, "site"].iloc[0]),
        "out_shape_dhw": list(out_shape),
        "n_volumes": int(sum(len(rows) for rows in rows_by_split.values())),
    }
    (out_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for split, rows in rows_by_split.items():
        frame = pd.DataFrame(rows)
        frame.to_csv(label_dir / f"{split}.tsv", sep="\t", index=False)
        frame.loc[frame["site"] != args.target_site_id].to_csv(label_dir / f"{split}_src.tsv", sep="\t", index=False)
        frame.loc[frame["site"] == args.target_site_id].to_csv(label_dir / f"{split}_tar.tsv", sep="\t", index=False)
        print(
            f"{split}: total={len(frame)} src={int((frame['site'] != args.target_site_id).sum())} "
            f"tar={int((frame['site'] == args.target_site_id).sum())}",
            flush=True,
        )


if __name__ == "__main__":
    main()
