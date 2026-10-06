"""HACA3 harmonization of ABIDE T1 volumes to NYU, exported on the frozen ISBI 2027 slices.

HACA3 (Zuo et al., NeuroImage 2023) is run as published: the authors' code
(github.com/lianruizuo/haca3) and public pretrained weights, no retraining.
1. prepare: N4 bias-field correction and rigid registration (SimpleITK, Mattes mutual
   information over the dilated template brain) of each raw volume to the MNI152NLin2009cAsym
   1 mm template, cropped to HACA3's 192x224x192 grid, as HACA3's preprocessing requires. The
   transform is inverted after harmonization.
2. target: the NYU target image is the NYU training volume whose HACA3 contrast code (theta, mean
   over axial slices with anatomy) is the medoid of the NYU training volumes' codes.
   harmonize: HACA3 with that target image; axial, coronal and sagittal passes fused by the
   authors' fusion network, as in `haca3-test`.
3. export: each output is resampled back onto the subject's native grid with the inverse
   transform, normalized like the raw volume (`robust_normalize`) and cut at the frozen slice with
   the raw slice's head mask and crop (`fixed_slice_from_volume`), so outputs pair with raw slices.
Native geometry is taken from nibabel's canonical arrays and affines, the arrays the slice
pipeline reads, so the round trip lands on the raw voxel grid by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harmonit.data.abide_slices_dataset import AbideSlicesDataset, robust_normalize  # noqa: E402

METHOD = "haca3"
TARGET_SITE = 5
CROP = (slice(0, 192), slice(4, 228), slice(0, 192))  # template voxels kept: HACA3's 192x224x192 grid
LPS = np.diag([-1.0, -1.0, 1.0])
ORIENTATIONS = ("axial", "coronal", "sagittal")


def to_sitk(array, affine):
    """SimpleITK image whose index (i, j, k) is array[i, j, k] at nibabel's RAS position, in LPS."""
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.ascontiguousarray(np.asarray(array, np.float32).transpose(2, 1, 0)))
    linear = affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    u, _, vt = np.linalg.svd(LPS @ linear / spacing)  # drop scanner shear; the round trip uses this grid both ways
    image.SetSpacing(spacing.tolist())
    image.SetDirection((u @ vt).ravel().tolist())
    image.SetOrigin((LPS @ affine[:3, 3]).tolist())
    return image


def from_sitk(image):
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(image).transpose(2, 1, 0)


def load_template(path):
    """Template T1w and its brain mask (TemplateFlow `_desc-brain_mask` next to it), cropped to HACA3's grid."""
    image = nib.load(path)
    affine = image.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ np.array([c.start for c in CROP])
    mask = nib.load(str(path).replace("_T1w.nii.gz", "_desc-brain_mask.nii.gz")).get_fdata() > 0
    return np.asarray(image.get_fdata(dtype=np.float32))[CROP], mask[CROP], affine


def native(t1_path):
    image = nib.as_closest_canonical(nib.load(str(t1_path)))
    return np.clip(image.get_fdata(dtype=np.float32), 0, None), image.affine


def n4(image):
    import SimpleITK as sitk

    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    correct = sitk.N4BiasFieldCorrectionImageFilter()
    correct.SetMaximumNumberOfIterations([50, 50, 50, 50])
    correct.Execute(sitk.Shrink(image, [4] * 3), sitk.Shrink(mask, [4] * 3))
    return sitk.Cast(image / sitk.Exp(sitk.Cast(correct.GetLogBiasFieldAsImage(image), sitk.sitkFloat32)), sitk.sitkFloat32)


def head_geometry(image):
    """Head centroid (x, y) and vertex height (z) in physical LPS coordinates, from an Otsu mask."""
    import SimpleITK as sitk

    small = sitk.Shrink(image, [2] * 3)
    index = np.argwhere(sitk.GetArrayFromImage(sitk.OtsuThreshold(small, 0, 1, 200)) > 0)[:, ::-1]
    direction = np.array(small.GetDirection()).reshape(3, 3)
    points = np.array(small.GetOrigin()) + (direction @ (index * np.array(small.GetSpacing())).T).T
    return points.mean(0), np.percentile(points[:, 2], 99.5)


def register(fixed, fixed_mask, moving, seed):
    """Rigid registration with the metric restricted to the dilated template brain, which every scan
    covers. Partial-coverage scans (e.g. UM slabs) defeat a centre-of-mass start, so it starts from
    the moments initialization and from vertex-aligned starts (+-15 mm) and keeps the start with the
    best coarse-level metric. Any scale term was exploited: affine stretched UM slabs to fill the
    template head, and a brain-masked similarity shrank heads into the mask."""
    import SimpleITK as sitk

    def run(transform, shrink=(4, 2, 1), sigmas=(2, 1, 0), iterations=200):
        method = sitk.ImageRegistrationMethod()
        method.SetMetricAsMattesMutualInformation(32)
        method.SetMetricSamplingStrategy(method.RANDOM)
        method.SetMetricSamplingPercentage(0.1, seed)
        method.SetMetricFixedMask(fixed_mask)
        method.SetInterpolator(sitk.sitkLinear)
        method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=iterations,
                                                        relaxationFactor=0.5)
        method.SetOptimizerScalesFromPhysicalShift()
        method.SetShrinkFactorsPerLevel(list(shrink))
        method.SetSmoothingSigmasPerLevel(list(sigmas))
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        method.SetInitialTransform(transform, inPlace=True)
        method.Execute(fixed, moving)
        return transform, method.GetMetricValue()

    starts = [sitk.Euler3DTransform(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS))]
    (fixed_centre, fixed_top), (moving_centre, moving_top) = head_geometry(fixed), head_geometry(moving)
    for dz in (0.0, -15.0, 15.0):
        start = sitk.Euler3DTransform()
        start.SetCenter(fixed_centre.tolist())
        start.SetTranslation([*(moving_centre[:2] - fixed_centre[:2]), moving_top - fixed_top + dz])
        starts.append(start)
    coarse = [run(start, shrink=(4, 2), sigmas=(2, 1), iterations=100) for start in starts]
    return run(min(coarse, key=lambda pair: pair[1])[0])[0]


def prepare_one(job):
    import SimpleITK as sitk

    sid, t1_path, template_path, out_dir, seed = job
    out_dir = Path(out_dir)
    if (out_dir / "xfm" / f"{sid}.tfm").exists():
        return None
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    start = time.time()
    template, brain, template_affine = load_template(template_path)
    fixed = to_sitk(template, template_affine)
    dilated = ndimage.binary_dilation(brain, iterations=5)
    fixed_mask = sitk.Cast(to_sitk(dilated.astype(np.float32), template_affine) > 0.5, sitk.sitkUInt8)
    array, affine = native(t1_path)
    moving = n4(to_sitk(array, affine))
    transform = register(fixed, fixed_mask, moving, seed)
    registered = from_sitk(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0)).astype(np.float32)
    nib.save(nib.Nifti1Image(registered, template_affine), out_dir / "mni" / f"{sid}.nii.gz")
    sitk.WriteTransform(transform, str(out_dir / "xfm" / f"{sid}.tfm"))
    ncc = float(np.corrcoef(template[brain], registered[brain])[0, 1])
    return {"subject_id": sid, "brain_ncc_to_template": round(ncc, 4), "seconds": round(time.time() - start, 1)}


def subjects(manifest_path, splits_path, split, site=None):
    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    ids = set(json.loads(Path(splits_path).read_text())[split])
    rows = manifest[manifest.subject_id.isin(ids) & ((manifest.site == site) if site else True)]
    root = Path(manifest_path).parent
    return [(r.subject_id, root / Path(*Path(r.t1_path).parts[1:])) for r in rows.itertuples()]


def cmd_prepare(args):
    out_dir = Path(args.out_dir)
    for sub in ("mni", "xfm"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    jobs = [(sid, str(path), args.template, str(out_dir), args.seed)
            for sid, path in subjects(args.manifest_path, args.splits_path, args.split, args.site)
            if not args.subject or sid in args.subject]
    qc_path = out_dir / "registration_qc.csv"
    with Pool(args.workers) as pool, qc_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "brain_ncc_to_template", "seconds"])
        if handle.tell() == 0:
            writer.writeheader()
        for row in pool.imap_unordered(prepare_one, jobs):
            if row:
                writer.writerow(row)
                handle.flush()
                print(row, flush=True)


def cmd_target(args):
    """Pick the NYU target volume: medoid of the training volumes' mean HACA3 theta."""
    from haca3.encode import axial_slices
    from haca3.modules.model import HACA3

    out_dir = Path(args.out_dir)
    model = HACA3(beta_dim=5, theta_dim=2, eta_dim=2, pretrained_haca3=args.harmonization_model, gpu_id=0)
    ids = [sid for sid, _ in subjects(args.manifest_path, args.splits_path, "train", "NYU")]
    thetas = []
    for sid in ids:
        slices, _ = axial_slices(out_dir / "mni" / f"{sid}.nii.gz")
        theta, foreground = model.encode_theta(slices)
        thetas.append(theta.numpy()[foreground.numpy() >= 0.01].mean(0))
    thetas = np.array(thetas)
    distances = np.linalg.norm(thetas[:, None] - thetas[None], axis=-1).sum(1)
    choice = ids[int(np.argmin(distances))]
    report = {"target": choice, "rule": "medoid of mean theta over the NYU training volumes", "n": len(ids),
              "theta": dict(zip(ids, thetas.round(5).tolist()))}
    (out_dir / "target.json").write_text(json.dumps(report, indent=1) + "\n")
    print("target", choice)


def cmd_harmonize(args):
    import torch
    from haca3.modules.model import HACA3
    from haca3.test import load_source_images, obtain_single_image

    out_dir = Path(args.out_dir)
    (out_dir / "haca3").mkdir(parents=True, exist_ok=True)
    model = HACA3(beta_dim=5, theta_dim=2, eta_dim=2, pretrained_haca3=args.harmonization_model, gpu_id=0)
    target_id = args.target or json.loads((out_dir / "target.json").read_text())["target"]
    target, _, norm_val = obtain_single_image(out_dir / "mni" / f"{target_id}.nii.gz", True)
    target = [target.permute(2, 1, 0).permute(0, 2, 1).flip(1)[100:120, ...]]  # as haca3-test
    views = {"axial": lambda x: x.permute(2, 0, 1), "coronal": lambda x: x.permute(0, 2, 1).flip(1),
             "sagittal": lambda x: x.permute(1, 2, 0).flip(1)}
    todo = sorted(p.name.removesuffix(".nii.gz") for p in (out_dir / "mni").glob("*.nii.gz"))
    todo = [sid for sid in todo if (not args.subject or sid in args.subject)
            and not (out_dir / "haca3" / f"{sid}_harmonized_fusion.nii.gz").exists()]
    for sid in todo:
        start = time.time()
        torch.manual_seed(args.seed)  # Gumbel-softmax anatomy sampling
        source, header = load_source_images([out_dir / "mni" / f"{sid}.nii.gz"])
        out_path = out_dir / "haca3" / f"{sid}.nii.gz"
        for orientation in ORIENTATIONS:
            model.harmonize(source_images=[views[orientation](image) for image in source], target_images=target,
                            target_theta=None, target_eta=None, out_paths=[out_path], header=header,
                            recon_orientation=orientation, norm_vals=[norm_val], num_batches=4)
        paths = [out_dir / "haca3" / f"{sid}_harmonized_{o}.nii.gz" for o in ORIENTATIONS]
        model.combine_images(paths, out_path, norm_val, args.fusion_model)
        for path in paths:
            path.unlink()
        print(f"{sid} {time.time() - start:.1f}s", flush=True)


def back_to_native(harmonized_path, transform_path, t1_path):
    import SimpleITK as sitk

    shape_source = nib.as_closest_canonical(nib.load(str(t1_path)))
    grid = to_sitk(np.zeros(shape_source.shape[:3], np.float32), shape_source.affine)
    image = nib.load(str(harmonized_path))
    moved = sitk.Resample(to_sitk(image.get_fdata(dtype=np.float32), image.affine), grid,
                          sitk.ReadTransform(str(transform_path)).GetInverse(), sitk.sitkLinear, 0.0)
    return robust_normalize(np.clip(from_sitk(moved), 0, None))


_EXPORT = {}


def _export_init(args):
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    dataset = AbideSlicesDataset(manifest_path=args.manifest_path, splits_path=args.splits_path, split=args.split,
                                 out_hw=(256, 256), slice_mode="fixed", seed=42, valid_nonzero_frac=0.02,
                                 fg_bbox_thr=0.02, volume_cache_size=2, mask_mode="none", bg_suppress=True,
                                 input_mode="image")
    dataset.aug_affine = False
    _EXPORT.update(args=args, dataset=dataset, paths=dict(subjects(args.manifest_path, args.splits_path, args.split)))


def _export_one(job):
    from hcld_export_ldm import fixed_slice_from_volume

    row, index = job
    args, dataset = _EXPORT["args"], _EXPORT["dataset"]
    sample, out_dir = dataset.samples[row], Path(args.out_dir)
    sid = str(sample.subject_id)
    raw = fixed_slice_from_volume(dataset, row, dataset._load_volume(sample), index)
    volume = back_to_native(out_dir / "haca3" / f"{sid}_harmonized_fusion.nii.gz", out_dir / "xfm" / f"{sid}.tfm",
                            _EXPORT["paths"][sid])
    harmonized = np.clip(fixed_slice_from_volume(dataset, row, volume, index), 0, 1)
    return harmonized[None], raw[None], sid, int(sample.site_id), index


def slice_index_map(path):
    """Frozen slice indices from a JSON map or from an existing export NPZ of the same split."""
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            return dict(zip(data["subject_ids"].astype(str), data["slice_indices"].astype(int).tolist()))
    return {str(k): int(v) for k, v in json.loads(Path(path).read_text()).items()}


def cmd_export(args):
    out_dir = Path(args.out_dir)
    index = slice_index_map(args.slice_index_map)
    _export_init(args)
    jobs = [(row, index[str(s.subject_id)]) for row, s in enumerate(_EXPORT["dataset"].samples)
            if not args.subject or str(s.subject_id) in args.subject]
    with Pool(args.workers, initializer=_export_init, initargs=(args,)) as pool:
        results = pool.map(_export_one, jobs, chunksize=4)
    rows = dict(zip(("images", "raw_images", "subject_ids", "site_ids", "slice_indices"), map(list, zip(*results))))
    target = out_dir / "export" / args.split
    target.mkdir(parents=True, exist_ok=True)
    name = "haca3_slices.npz" if not args.subject else "haca3_slices_subset.npz"
    np.savez_compressed(target / name, images=np.stack(rows["images"]).astype(np.float32),
                        raw_images=np.stack(rows["raw_images"]).astype(np.float32),
                        subject_ids=np.asarray(rows["subject_ids"], dtype=str),
                        site_ids=np.asarray(rows["site_ids"], dtype=np.int64),
                        slice_indices=np.asarray(rows["slice_indices"], dtype=np.int64),
                        split=np.asarray(args.split), method=np.asarray(METHOD))
    print(target / name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("prepare", "target", "harmonize", "export"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--splits-path", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--subject", action="append", help="Restrict to these subjects (repeatable)")
    p.add_argument("--site", help="Restrict prepare to one site (e.g. NYU)")
    p.add_argument("--template", help="tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz (prepare)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target", help="Override the NYU target subject (default: target.json from the target step)")
    p.add_argument("--harmonization-model", help="harmonization_public.pt")
    p.add_argument("--fusion-model", help="fusion.pt")
    p.add_argument("--slice-index-map", help="Frozen slice map: JSON, or an export NPZ of the same split (export)")
    args = p.parse_args()
    {"prepare": cmd_prepare, "target": cmd_target, "harmonize": cmd_harmonize, "export": cmd_export}[args.step](args)


if __name__ == "__main__":
    main()
