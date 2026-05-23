# Reproducibility Guide

This guide describes how to reproduce the current HarmonIt ABIDE benchmark
artifacts from a fresh checkout, assuming the ABIDE T1 data and frozen
site-probe checkpoint are available.

## Environment

Create the main HarmonIt environment as described in
`documentation/INSTALLATION.md`.

For HCLD experiments, the Slurm scripts assume two environments on the cluster:

- `harmonit311`: HarmonIt data/evaluation environment
- `hcld`: HCLD/Generative Models environment

The HCLD repository is expected at:

```text
/home/muqsitamir/repos/HCLD
```

The HarmonIt repository is expected at:

```text
/home/muqsitamir/repos/HarmonIt
```

Override `REPO_DIR`, `PYTHON`, `HCLD_PYTHON`, or `HARMONIT_PYTHON` in the Slurm
environment if your paths differ.

## Data Preparation

Generate the ABIDE manifest after data download:

```bash
python scripts/make_abide_manifest.py
python scripts/make_splits.py
```

For the 3D HCLD baseline, prepare resized volumes and source/target TSV labels:

```bash
sbatch slurm/hcld_prepare_abide.sbatch
```

Expected HCLD prep output:

```text
outputs/hcld_abide/
  volumes/
  labels/train.tsv
  labels/val.tsv
  labels/test.tsv
  labels/train_src.tsv
  labels/train_tar.tsv
  prep_config.json
```

## Evaluation Contract

Every harmonization method should export an NPZ artifact under:

```text
outputs/harmonized/<method_name>/test/<method_name>_slices.npz
```

Evaluate with:

```bash
python scripts/eval_harmonized_npz.py \
  --npz-path outputs/harmonized/<method_name>/test/<method_name>_slices.npz \
  --site-probe-ckpt checkpoints/site_probe_v0.3_aug_ramp15/model_best.pt \
  --out-dir outputs/harmonized/<method_name>/test/metrics
```

The evaluator writes:

```text
summary_metrics.json
summary_metrics.csv
pairwise_preservation_metrics.csv
distribution_by_site.csv
site_probe_raw_cm.npy
site_probe_harmonized_cm.npy
```

## Diffusion Img2Img 20k

Run:

```bash
sbatch slurm/diffusion_img2img_nyu.sbatch
```

This trains, exports, and evaluates the NYU-target diffusion baseline at:

```text
outputs/harmonized/diffusion_img2img_nyu_s20000_strength035/
```

Key settings:

- `steps=20000`
- `num_timesteps=1000`
- `base_channels=64`
- `channel_mults=(1, 2, 4, 4)`
- `ddim_steps=50`
- `strength=0.35`
- `target_site_id=5` (NYU)

## Adapted HCLD

The adapted HCLD pipeline has three stages after data prep.

### 1. Autoencoder

Smoke test:

```bash
sbatch slurm/hcld_adapted_aekl_smoke.sbatch
```

Full training:

```bash
sbatch slurm/hcld_adapted_aekl_train.sbatch
```

Expected checkpoint:

```text
outputs/hcld_abide/adapted_hcld/aekl/checkpoints/model_latest.pt
```

### 2. Latent Diffusion

Smoke test:

```bash
sbatch slurm/hcld_adapted_ldm_smoke.sbatch
```

Full training:

```bash
sbatch slurm/hcld_adapted_ldm_train.sbatch
```

Expected checkpoint:

```text
outputs/hcld_abide/adapted_hcld/ldm/checkpoints/model_latest.pt
```

### 3. Export and Evaluation

```bash
sbatch slurm/hcld_adapted_export_eval.sbatch
```

Expected artifact:

```text
outputs/harmonized/adapted_hcld/test/adapted_hcld_slices.npz
outputs/harmonized/adapted_hcld/test/metrics/summary_metrics.json
```

The export/evaluation script generates an evaluator-compatible slice-index map
before export. Keep that behavior enabled for reproducible metrics.

## Figure Generation

### Harmonization Comparison Panel

Use `scripts/visualize_harmonization_panel.py` with one `--method LABEL=NPZ`
argument per completed method. Example:

```bash
python scripts/visualize_harmonization_panel.py \
  --subject-id USM_50514 \
  --ncols 5 \
  --out outputs/figures/harmonization_panel_2x5_with_hcld.png \
  --format both \
  --method "NeuroCombat=outputs/harmonized/neurocombat/test/neurocombat_slices.npz" \
  --method "Diffusion=outputs/harmonized/diffusion_img2img_nyu_s20000_strength035/test/diffusion_img2img_nyu_slices.npz" \
  --method "Adapted HCLD=outputs/harmonized/adapted_hcld/test/adapted_hcld_slices.npz"
```

Add the other method NPZs when they are present locally.

### Preprocessing Stages Panel

Use:

```bash
python scripts/visualize_preprocessing_stages.py \
  --subject-id USM_50514 \
  --slice-idx 253 \
  --out outputs/figures/preprocessing_stages_usm_50514.png \
  --format both
```

The preprocessing figure script is self-contained for visualization and uses
`nibabel`, `PIL`, `scipy`, `numpy`, and `pandas`. It avoids importing the full
dataset module so it can run even when a local torch install is unavailable.

## Reporting Notes

When reporting the current table, include both harmonization and preservation
metrics:

- Lower site balanced accuracy means less site information remains.
- Higher balanced-accuracy drop means stronger site removal.
- Higher PSNR, feature similarity, and cross-correlation mean better
  preservation.
- Lower Wasserstein and KL mean lower global distribution shift.

Do not rank methods by site balanced accuracy alone. Aggressive methods can
remove site information while visibly damaging anatomy.

