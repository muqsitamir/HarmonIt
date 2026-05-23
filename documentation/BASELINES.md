# Harmonization Baselines

This document records the image-level harmonization baselines used in the ABIDE
benchmark table. All methods export the same NPZ contract so they can be
evaluated by `scripts/eval_harmonized_npz.py`.

## Common Benchmark Contract

All exported test artifacts should contain:

- `images`: harmonized slices with shape `[N, 1, 256, 256]`
- `raw_images`: deterministic fixed raw slices with the same shape
- `subject_ids`, `site_ids`, `slice_indices`: metadata aligned row-by-row
- `split`: usually `test`
- `method`: method identifier

The evaluator reloads the fixed-slice ABIDE dataset and checks that subject IDs,
site IDs, slice indices, and raw images match before computing metrics. This is
intentional: it prevents accidental evaluation on a different slice policy.

Default benchmark settings:

- Split: `test`
- Target/reference site: NYU (`site_id=5`)
- Output size: `256 x 256`
- Fixed-slice filtering: `valid_nonzero_frac=0.02`, `fg_bbox_thr=0.02`
- Background suppression: enabled before resize
- Site leakage model: `checkpoints/site_probe_v0.3_aug_ramp15/model_best.pt`

## DLEST

Script: `scripts/methods/dlest_disentangled.py`

DLEST is a lightweight disentangled site/style transfer baseline. It uses:

- a content encoder for anatomy-preserving representation,
- a learned site-style embedding table,
- an AdaIN-style decoder that recombines content with a target site style, and
- a site discriminator/classifier for adversarial and domain-classification
  pressure.

Training losses include reconstruction, cycle consistency, content consistency,
gradient consistency, adversarial loss, and site classification loss.

The reported `DLEST 1000` and `DLEST 1500` rows are checkpoint variants of the
same model family. Both export non-NYU subjects toward the NYU target style and
leave NYU test samples unchanged via the identity-target rule.

Interpretation:

- `DLEST 1000` is the most conservative checkpoint. It preserves anatomy very
  strongly but leaves substantial site information.
- `DLEST 1500` shifts slightly further toward harmonization. Site balanced
  accuracy improves relative to `DLEST 1000`, while PSNR drops modestly.

Example export/evaluation pattern:

```bash
python scripts/methods/dlest_disentangled.py export \
  --checkpoint outputs/harmonized/dlest_nyu/train/checkpoints/step_001500.pt \
  --out-dir outputs/harmonized/dlest_nyu \
  --split test \
  --target-site-id 5

python scripts/eval_harmonized_npz.py \
  --npz-path outputs/harmonized/dlest_nyu/test/dlest_nyu_slices.npz \
  --site-probe-ckpt checkpoints/site_probe_v0.3_aug_ramp15/model_best.pt \
  --out-dir outputs/harmonized/dlest_nyu/test/metrics
```

## Diffusion Img2Img

Script: `scripts/methods/diffusion_img2img.py`

The diffusion baseline is a site-conditional DDPM trained on ABIDE 2D slices.
The model is a conditional U-Net denoiser with site conditioning and an EMA copy
used for export.

The completed `Diffusion img2img 20k` row used:

- Train steps: `20000`
- Batch size: `16`
- Diffusion timesteps: `1000`
- U-Net base channels: `64`
- Channel multipliers: `1 2 4 4`
- Target/reference site: NYU (`site_id=5`)
- Export sampler: DDIM-style image-to-image
- DDIM steps: `50`
- Img2img strength: `0.35`
- Identity-target rule: enabled

Slurm launcher:

```bash
sbatch slurm/diffusion_img2img_nyu.sbatch
```

The Slurm script trains, exports, and evaluates:

```bash
python scripts/methods/diffusion_img2img.py train \
  --out-dir outputs/harmonized/diffusion_img2img_nyu_s20000_strength035 \
  --steps 20000 \
  --batch-size 16 \
  --base-channels 64 \
  --channel-mults 1 2 4 4 \
  --num-timesteps 1000

python scripts/methods/diffusion_img2img.py export \
  --checkpoint outputs/harmonized/diffusion_img2img_nyu_s20000_strength035/checkpoints/model_latest.pt \
  --out-dir outputs/harmonized/diffusion_img2img_nyu_s20000_strength035 \
  --ddim-steps 50 \
  --strength 0.35
```

Interpretation:

- Diffusion gives a stronger harmonization/preservation trade-off than DLEST.
- It reduces site balanced accuracy substantially while retaining acceptable
  PSNR, feature similarity, and cross-correlation.
- Its low Wasserstein and KL values indicate that the global intensity
  distribution remains close to the raw test distribution.

## Adapted HCLD

Scripts:

- `scripts/methods/hcld_prepare_abide.py`
- `scripts/methods/hcld_train_autoencoder.py`
- `scripts/methods/hcld_train_ldm.py`
- `scripts/methods/hcld_export_ldm.py`

Config:

- `configs/adapted_hcld_aekl.json`

Slurm launchers:

- `slurm/hcld_prepare_abide.sbatch`
- `slurm/hcld_adapted_aekl_smoke.sbatch`
- `slurm/hcld_adapted_aekl_train.sbatch`
- `slurm/hcld_adapted_ldm_smoke.sbatch`
- `slurm/hcld_adapted_ldm_train.sbatch`
- `slurm/hcld_adapted_export_eval.sbatch`

HCLD is a volumetric latent diffusion baseline adapted to the available A100
40GB hardware. Faithful encoder/decoder nonlocal attention exceeded available
memory, so the adapted configuration disables encoder and decoder nonlocal
attention while keeping the conditioning pipeline, deepest attention,
channels, latent dimensionality, and residual block count close to the faithful
setup. It also enables gradient checkpointing and bf16 autocast.

Completed adapted HCLD configuration:

- Input volume shape: `[1, 192, 192, 64]`
- Autoencoder channels: `[32, 64, 64]`
- Latent channels: `6`
- Residual blocks: `2`
- Deepest attention: enabled
- Encoder nonlocal attention: disabled
- Decoder nonlocal attention: disabled
- Gradient checkpointing: enabled
- AMP dtype: `bf16`

Important evaluation note: HCLD operates on 3D volumes, but the HarmonIt
benchmark evaluates deterministic 2D fixed slices. The export script maps HCLD
volumetric outputs back to the original volume orientation and extracts the same
fixed benchmark slice. `slurm/hcld_adapted_export_eval.sbatch` first generates a
slice-index map with the HarmonIt evaluation environment, then forces HCLD
export to use that exact map. This avoids small runtime-dependent tie-breaking
differences in slice selection.

Interpretation:

- Adapted HCLD strongly removes site information.
- In the completed run it over-harmonizes visually and has weaker preservation
  metrics than conservative methods, so it should be reported as a powerful but
  aggressive adapted volumetric baseline.

