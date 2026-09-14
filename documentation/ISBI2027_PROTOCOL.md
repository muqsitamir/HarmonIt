# ISBI 2027 evaluation protocol v1

Frozen 2026-09-13 before corrected table generation. Deadline: 26 October 2026;
full-paper notification: 12 January 2027. Official schedule:
https://biomedicalimaging.org/2027/

Scope: existing 2D benchmark artifacts, corrected evaluator, retrained site
probe and valid shuffled-label control, paired subject confidence intervals.
No 2.5D development or volumetric experiments in this submission.

## Cohort and interpretation

Use the existing 890/113/109 subject splits. Confirm disjointness and exact
manifest coverage. The 109 test subjects have been inspected historically;
this is a retrospective benchmark reanalysis, not a newly untouched test.
Do not tune generators or select checkpoints using the corrected test results.
Record statistical/cohort-fit methods separately from inductive methods.

Primary translation cohort: 90 non-NYU subjects. Report all 109 subjects and
the 19 NYU subjects separately, including identity counts and undefined metrics.
Use the same freshly reconstructed raw slice reference for every artifact;
check subject/site/slice metadata and raw pixels. Embedded-reference mode is
diagnostic only and labeled explicitly.

Fixed slices are frozen in `configs/isbi2027/test_slice_indices.json` (amendment,
2026-09-13). Selection takes the top foreground fraction via an unstable argsort, and
4/109 test subjects tie exactly (Leuven_50711, SBL_51570, UM_50375, USM_50483), so
the chosen slice can differ by host. The map is the vpulab fresh reference that all
eight vpulab-exported artifacts reproduce. The historical cl HCLD export picked slice
68 for SBL_51570 (map: 69), so HCLD is re-exported on cl with the frozen map, same
checkpoints, seed, GPU type and settings; its other 108 subjects are compared with the
historical export as a reproducibility check. The evaluator rejects any reference that
does not reproduce the map.

## Metrics

- Fixed-scale PSNR: data_range=1; do not clip outputs. Exact identities are
  infinite. A mean or CI containing nonfinite values is null, with counts and
  a separately labeled finite-only diagnostic mean.
- Whole-image MSE, MAE, pixel cosine and cross-correlation. No brain-mask or
  VGG-feature claims. Zero-norm cosine and constant-input correlation are undefined.
- Per-subject raw/harmonized Wasserstein and directional KL; report their means.
  KL uses 50 fixed [0,1] bins with underflow/overflow bins and 1e-8 probability
  smoothing. No output clipping. These values are not target-domain alignment.
  Their subject-mean aggregation differs from historical pooled-cohort distances.
- Frozen-probe raw/harmonized BA and paired BA drop, macro-averaged over the
  true classes present in each reported cohort. Predictions retain all 17 classes.

Bootstrap: 2,000 replicates, seed 20260913, resample subjects within acquisition
site while retaining site counts. Same indices for every method and for raw
versus harmonized predictions. Percentile 95% intervals for subject means and
BA; paired differences for source-cohort method comparisons. These are descriptive
intervals conditional on the cohort's sites and trained checkpoints, not seed or
unseen-site uncertainty. Pairwise intervals are not multiplicity-adjusted tests.

## Probe experiments

Reuse ResNet-18 and the recorded production augmentation and optimization
configuration. Train raw and shuffled-label controls with the same training
budget; select checkpoints on full validation BA and evaluate test once after
selection. Shuffled labels are permuted between subjects separately within each
split, remain fixed across slices/epochs, and preserve class counts. Record the
subject-to-label mapping. Never interpret class-ID relabeling as a shuffle control.

Train-on-harmonized probes, if artifact exports fit the time budget, are restricted
to a predefined set: histogram matching, tuned CycleGAN, and diffusion img2img 20k.
This set spans intensity, adversarial and diffusion families and is chosen before
the corrected table. Requires train/validation exports from fixed generators;
no use of test data to train or tune probes. The raw/shuffle pair is mandatory.

## Execution and release

`scripts/eval_isbi2027.py` writes to a new directory and rejects overwriting.
It records source/input/probe hashes, versions, fresh raw references, per-subject
predictions/metrics, group summaries, bootstrap indices, and paired differences.
`COMPLETE.json` is written only after every requested method succeeds. Partial
directories are diagnostic and must not be treated as complete tables.

Historical results remain untouched. The initial corrected table uses the existing
frozen probe and must be labeled accordingly; the new probe is a subsequent
experiment. No new files are pushed to GitHub without revisiting the user's
existing changed-files-only publication preference.

Go/no-go: 4 October. Require a reproducible table, interpretable probe controls,
and a supported scientific finding beyond correction of implementation mistakes.
