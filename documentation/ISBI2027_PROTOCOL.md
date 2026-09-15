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
does not reproduce the map. Result (cl job 206377, A100): the other 108 subjects are
bit-identical to the historical export; only SBL_51570 changed (slice 68 -> 69).

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

The recipe is the frozen checkpoint's run (20260417_164204): batch 64, 10 epochs x 50
steps sampled with replacement, AdamW 3e-4, affine augmentation p=0.9, rotation 12 deg,
translation 32 px, scale jitter 0.2, random valid training slices, fixed validation
slices; launched by `scripts/vpulab_isbi2027_probe.sh`. Its validation BA fluctuated
0.47-0.93 across epochs, so best-epoch selection is noisy. Known inherited quirk, kept
for fidelity: the dataset's RandomState is copied into each DataLoader worker without
reseeding, so the 4 workers repeat the same slice/augmentation draw sequence.

Train-on-harmonized probes, if artifact exports fit the time budget, are restricted
to a predefined set: histogram matching, tuned CycleGAN, and diffusion img2img 20k.
This set spans intensity, adversarial and diffusion families and is chosen before
the corrected table. Requires train/validation exports from fixed generators;
no use of test data to train or tune probes. The raw/shuffle pair is mandatory.

## Amendment 2, 2026-09-14 (written before any of these results were seen)

Seen at this point: frozen-probe nine-method table; seed-42 raw and shuffle probes
and their nine-method test evaluation (retrained raw test BA 0.822 vs frozen 0.951;
retrained probe rated diffusion, CycleGAN and aggressive StarGAN as retaining more site
information). The following were added in response and are therefore post hoc to that
observation; they are fixed before their own outcomes exist.

Probe seed variability. Raw probes with the same recipe for seeds 1-4 (plus 42). Each
seed's best-validation and final-epoch checkpoints are evaluated once on the nine test
artifacts. Report per-method spread of harmonized source BA across seeds and the
rank agreement of method orderings (Kendall tau between seeds). No seed is selected
using test results; all are reported.

Harmonized-probe (adversary) design. For histogram matching, tuned CycleGAN and
diffusion 20k (the pre-selected set), export fixed-slice train/val outputs with the
settings of the historical test artifacts (`scripts/vpulab_isbi2027_exports.sh`).
Before use, each method's test re-export must reproduce its historical test artifact,
and embedded raw slices must be identical across methods. `scripts/train_slice_probe.py`
trains ResNet-18 on one fixed slice per subject with the production optimizer, budget
and affine augmentation; sources are raw slices (matched baseline) or one method's
outputs; seeds 1-3 per source; one raw subject-level shuffle control (seed 1). Select on
validation BA from the same source; evaluate best and final checkpoints once on raw
test and all nine test artifacts. Primary quantity: source (non-NYU) BA of the
method-trained probe on that method's own test outputs, compared with the raw-trained
slice probe on the same outputs and on raw test slices. A method is said to hide rather
than remove site information when the method-trained probe recovers substantially more
site signal than the raw-trained probe; report intervals rather than a threshold.
NYU training subjects are passed through unchanged by CycleGAN and diffusion but
transformed by histogram matching; note this asymmetry.

Target alignment. Separate from raw-to-harmonized change. Reference: foreground pixels
of NYU training subjects' fixed raw slices, each subject weighted equally. Foreground is
defined from the raw slice (> 0.02), never from a method's output, and applied to both
raw and harmonized images. Per non-NYU test subject: Wasserstein-1 and directional KL
(harmonized || reference; 50 fixed [0,1] bins plus under/overflow, 1e-8 smoothing) for
the harmonized and the raw image; report both and their paired difference with the
subject bootstrap. Intensity alignment is not evidence of anatomical correctness.

Amendment 3, 2026-09-14 (after the test re-exports, before any train/val use). The
exact-reproduction gate (1e-5) failed and was revised as follows; the observations are
recorded here. Histogram matching: 108/109 identical, one NYU subject differs in 906
pixels (max 0.0064) from rank tie-breaking with identical raw input and reference
quantiles. CycleGAN: all 90 translated subjects differ by at most 5.8e-4 (GPU kernels).
Diffusion 20k: same checkpoint file (written 2 min before the historical export),
batch size, DDIM steps and strength, yet outputs differ substantially: median
foreground mean |historical - re-export| 0.085 versus 0.075 between output and raw,
median correlation 0.87, while mean source PSNR is unchanged (21.71 vs 21.73 dB).
The img2img start noise is not reproducible across GPUs, so pixel outputs are a
sampling draw. Revised gate: deterministic methods require identical subjects/slices
and max abs difference <= 0.01; diffusion requires identical subjects/slices and
source PSNR within 0.1 dB. The vpulab re-export is evaluated as an additional test
artifact, `diffusion_20k_redraw`, sampled like the diffusion train/val exports; the
diffusion-trained probe's primary test set is the redraw. Differences between the two
draws under the same probe quantify sampling variability and are reported.

Amendment 4, 2026-09-14 17:10 (post hoc; written after the harmonized-probe results
and before this control's result). Observed: probes trained on method outputs reached
source BA 0.85-0.97 on those methods' test outputs versus 0.09-0.49 for raw-trained
slice probes. Because site labels can also be predicted from head geometry, crop and
field of view, a silhouette control is added: slice probes (seeds 1-3, same recipe)
trained and evaluated on filled binary head silhouettes (raw slice > 0.02, holes filled)
of the same fixed slices. It bounds how much site recognition is available without
intensity or texture; it does not separate scanner effects from demographic or
anatomical cohort differences, which remain a stated limitation.

Amendment 5, 2026-09-14 17:25 (post hoc; written after the first silhouette probe's
validation BA of 0.79 and before any histogram-probe result). Head geometry alone
identifies sites, so site recovery by image probes does not show that intensity
(scanner appearance) information survives harmonization. An intensity-only probe is
added: features are the 50-bin [0,1] probability histogram (plus under/overflow bins,
1e-8 smoothing) of foreground pixels, foreground defined on the raw slice (> 0.02);
classifier is standardized multinomial logistic regression, inverse regularization C
chosen from {0.01, 0.1, 1, 10} by validation BA from the same training source;
deterministic, so no seeds. Trained on raw training slices (evaluated on raw and all
test outputs) and on each of the three exported methods' training outputs (evaluated
on that method's test outputs). Source BA with the evaluation run's stratified
bootstrap indices; paired difference own-trained minus raw-trained on the same outputs.

Method tracks. NeuroCombat was fit on the test cohort itself (transductive) and
histogram matching uses a pooled 17-site training reference, not NYU; describe both
accordingly and keep NeuroCombat in a separately labeled transductive row.

## Execution and release

`scripts/eval_isbi2027.py` writes to a new directory and rejects overwriting.
It records source/input/probe hashes, versions, fresh raw references, per-subject
predictions/metrics, group summaries, bootstrap indices, and paired differences.
`COMPLETE.json` is written only after every requested method succeeds. Partial
directories are diagnostic and must not be treated as complete tables.

Historical results remain untouched. The initial corrected table uses the existing
frozen probe and must be labeled accordingly; the new probe is a subsequent
experiment. Publication decided 2026-09-15: code, protocol, versioned results and the
manuscript are pushed to GitHub on this branch and linked from the paper.

Go/no-go: 4 October. Require a reproducible table, interpretable probe controls,
and a supported scientific finding beyond correction of implementation mistakes.
