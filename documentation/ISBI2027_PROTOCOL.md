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

Amendment 6, 2026-09-15 16:10 (post hoc; written before any result of this experiment).
Seen at this point: every result in the manuscript, including the five production-recipe
probes' per-epoch validation BA, which fluctuated under the constant learning rate (for
example seed 2: 0.88, 0.50, 0.87 in epochs 8-10; seed 1: 0.89, 0.61, 0.90 in epochs 6-8).
A reviewer could attribute the spread of verdicts across seeds (finding 1) to unconverged,
unstable training rather than to probe dependence as such. This experiment tests that.

Converged recipe: architecture, data, splits, preprocessing, random valid training slices,
affine augmentation, batch 64 and AdamW 3e-4 as in the production recipe, with four
changes: (1) 40 epochs x 50 steps (2,000 steps, four times the budget); (2) linear warmup
over 50 steps, then cosine decay to zero; (3) each DataLoader worker's RandomState is
reseeded from (seed, worker id), removing the inherited quirk; (4) implementation only:
normalized volumes are read from a local cache written by the dataset's own loader and
checked for exact equality of returned slices before use. Seeds 5-9 raw probes and one
subject-level shuffled-label control (seed 5). Primary checkpoint: final epoch (no
validation selection); the best-validation checkpoint is also evaluated and reported.
Each checkpoint is evaluated once on the ten test outputs with `eval_isbi2027.py`
(same bootstrap seed and indices).

Convergence is reported per seed as the range of validation BA over the last five epochs;
the recipe is called stable if that range is at most 0.05 for every seed. Outcomes: per
output, the range (min, max) of source BA across the five final checkpoints and Kendall
tau between seeds, next to the production-recipe values. Interpretation fixed now: if for
CycleGAN, diffusion, histogram matching and aggressive StarGAN the converged range is no
wider than the widest 95% bootstrap interval of a single converged probe on any output,
we report that verdict variability in the benchmark recipe is largely a training-
stability effect and revise finding 1 accordingly (a probe's recipe and convergence must
be reported); otherwise we report that convergence does not remove probe dependence. No
further recipe variants are tried after these results are seen.

Outcome of amendment 6 (recorded 2026-09-15 after the results; `analysis/converged_probes.json`).
Stable: last-five-epoch validation BA range at most 0.035 for every seed; raw test BA 1.00 for
all final checkpoints; shuffle control 0.03 [0.00, 0.05]. Converged ranges for CycleGAN,
diffusion, histogram matching and aggressive StarGAN were 0.11, 0.07, 0.13 and 0.16, within the
widest single-probe interval (0.18), so finding 1 is revised as fixed above. Converged probes
also rated every output except HCLD and NeuroCombat above the frozen probe's interval (e.g.
aggressive StarGAN 0.65-0.81 versus 0.21) and reordered methods (Kendall tau with the frozen
probe 0.64-0.73); the manuscript reports this as dependence on probe training.

Amendment 7, 2026-09-15 17:05 (post hoc; written before any brain-only probe or mask
statistic was computed). Seen at this point: every result in the manuscript, including the
head-silhouette control (BA 0.73-0.86) and amendment 6 training logs (no evaluations yet).
Common harmonization pipelines skull-strip before evaluation, whereas our probes see whole
head slices, and the silhouette control shows head geometry identifies sites. This control
asks whether the removed-versus-hidden result (amendment 2) holds when the probe sees only
brain tissue.

Brain masks: HD-BET 2.0.1 (release 2.0.0 parameters, default test-time augmentation) on each
raw T1 volume, reoriented to the canonical orientation, cut at the frozen fixed slice,
cropped with the same head bounding box and resized with the dataset's mask resizer to
256x256. Gate before any probe use: recomputed raw slices must equal the exports' raw
slices; masks must be non-empty on every fixed slice. Brain-to-head area ratios are
reported; subjects with ratio < 0.10 are listed (not excluded). One masked version of each
slice is shared by raw and harmonized images: brain-only input = image x raw-derived mask.

Probes: slice probes with the amendment 2 recipe (seeds 1-3, best and final checkpoints)
trained on brain-only raw slices, on brain-only outputs of histogram matching, CycleGAN and
diffusion (the pre-selected set), and on the binary brain masks (brain-shape control); one
subject-level shuffled-label control on brain-only raw slices (seed 1). Each checkpoint is
evaluated once, with brain-only inputs, on the raw test slices and the ten test outputs
(diffusion-trained probes' primary test set remains the redraw). Secondary: the amendment 5
intensity-histogram probe with foreground restricted to the brain mask.

Primary quantity as in amendment 2: source BA of the method-trained brain-only probe on
that method's brain-only test outputs, next to the raw-trained brain-only probe on the same
outputs, with the brain-shape control as a bound. Interpretation fixed now: if the
method-trained probes' intervals lie above the raw-trained probes' for every seed, the
hidden-not-removed conclusion is reported as holding without head geometry; otherwise the
manuscript restricts that conclusion to the evidence that survives (intensity-only probe)
and says so.

Outcome of amendment 7 (recorded 2026-09-15 after the results). Gate passed: recomputed raw
slices identical to the exports (max difference 0); no empty masks; brain-to-head area ratio
0.23-0.87. With best-validation checkpoints the method-trained brain-only probes' intervals lay
above the raw-trained probes' for all nine method-seed pairs (mean BA 0.91, 0.79, 0.74 versus
0.62, 0.48, 0.39); with final-epoch checkpoints for five of nine, which the manuscript states.
Brain-mask shape control 0.52-0.54. Secondary brain-foreground histogram probe: trained on
outputs 0.53, 0.55 and 0.38 versus 0.56 on raw brains, so the head-level conclusion that
histogram matching removed intensity site information does not hold within the brain; the
manuscript was corrected.

Amendment 8, 2026-09-24 (cohort description, no experiment). Reviewer question: are there
scanner differences within a site? ABIDE I releases no per-subject scanner identifier, and our
17 labels merge the released sub-samples (UM_1/UM_2, UCLA_1/UCLA_2, Leuven_1/Leuven_2).
`scripts/isbi2027_acquisition_heterogeneity.py` reads voxel size and matrix from every NIfTI
header and writes `results/isbi2027/analysis/acquisition_heterogeneity.json` plus a per-subject
CSV. Outcome: 11 of 17 sites hold more than one acquisition geometry (UM has two dominant
protocols, 1.016x1.016x1.2 mm for 61 subjects and 1.016x1.016x1.4 mm for 33; UCLA includes 10
scans at 1.5x1.5x4 mm, all in the training split); Leuven, NYU, Pitt, SBL, SDSU and Stanford are
homogeneous. This bounds what a single site label can mean and is reported as a limitation; no
probe or generator was retrained.

Amendment 9, 2026-10-02 (post hoc summary of existing predictions, after all probe results;
prompted by an external review). No probe was retrained. For each slice-probe method
(histogram matching, CycleGAN, diffusion tested on its second draw), the paired difference in
source BA between probes trained on that method's outputs and probes trained on raw slices,
averaged over seeds 1-3, with the evaluator's site-stratified bootstrap indices (2,000
replicates, seed 20260913), on whole heads and brain-only slices.
`scripts/isbi2027_probe_difference.py` writes `results/isbi2027/analysis/probe_difference.json`
(best-validation checkpoints) and `probe_difference_last.json` (final epoch). Outcome (best):
whole head +0.42 [0.36, 0.48], +0.55 [0.52, 0.58], +0.50 [0.45, 0.54]; brain only +0.29 [0.23,
0.35], +0.31 [0.24, 0.37], +0.36 [0.28, 0.43]. Final epoch: every interval above zero (whole head
+0.38 to +0.59, brain only +0.15 to +0.23). These intervals are conditional on the trained
checkpoints and replace the per-seed disjoint-interval statements in the paper.

Amendment 10, 2026-10-02 (post hoc description of existing predictions, prompted by an external
review; nothing retrained). Where the frozen probe sends source outputs it misclassifies:
`scripts/isbi2027_error_targets.py` writes `results/isbi2027/analysis/frozen_probe_error_targets.json`
from the canonical run. Outcome: adapted HCLD 84 errors of 90, none named NYU, 83 named Yale;
CycleGAN 80% of errors named NYU, aggressive StarGAN 75%, diffusion 65-67%, DLEST 22-32%,
histogram matching 18% (pooled reference, not NYU), NeuroCombat 3%.

Amendment 11, 2026-10-05 (post hoc; requested by P. Coupé's review). How much site is decodable
from demographics and brain size alone: multinomial logistic regression (standardized inputs,
balanced class weights) trained on the 890 training subjects, scored on the 90 source test
subjects with the evaluator's bootstrap. Inputs: age, sex, diagnosis; then also FastSurfer total
brain volume. `scripts/isbi2027_demographics_probe.py`, `results/isbi2027/analysis/demographics_probe.json`.
Outcome: source BA 0.22 [0.15, 0.29] from age, sex and diagnosis; 0.31 [0.22, 0.40] with brain
volume; raw-image probes reach 0.81-1.00.

Amendment 12, 2026-10-05 (written before running; requested by P. Coupé). The adapted HCLD output
is our retraining with reduced capacity, so a synthetic destroyed control is tried in its place.
Control: every raw test slice blurred with a 2D Gaussian (sigma 2, 4 and 8 pixels on the 256x256
slice, zero padding), scored by the frozen probe with `eval_isbi2027.py` (fresh reference, frozen
slice map, same bootstrap). Decision rule fixed now: if at sigma 8 the frozen-probe source BA is
below that of every real harmonizer (lowest: aggressive StarGAN, 0.21), the blur control replaces
the adapted HCLD output in the paper; otherwise HCLD is kept with its disclaimer. No other blur
settings are tried after the results are seen.
Outcome (`scripts/isbi2027_blur_control.py`, run `results/isbi2027/runs/frozen_probe_blur_control_20261005`,
alignment `results/isbi2027/analysis/target_alignment_blur`): source BA 0.075 [0.0625, 0.10] at
sigma 2 (PSNR 24.7 dB, XCorr 0.974; 87 of 90 labelled Yale, none NYU), 0.0625 at sigma 4 (the
constant-prediction value 1/16) and 0.068 at sigma 8, all below aggressive StarGAN. By the rule
the blur control replaces adapted HCLD in the paper. Table 1 shows sigma 2, the mildest setting;
sigma 4 and 8 are reported in the text. Blur moved intensities away from NYU (Delta W_NYU +0.016,
+0.031, +0.044; KL to NYU 0.57, 0.75, 0.92 against 0.30 raw).
Ranking statistics in the paper are recomputed over the nine remaining outputs scored by the
original probes (`scripts/isbi2027_tau_nine_outputs.py`, `analysis/kendall_tau_nine_outputs.json`):
Kendall tau 0.39-0.83 (benchmark recipe), 0.78-0.94 (converged), 0.56-0.67 (frozen vs converged);
largest seed range 0.46 and 0.16; every converged probe above the frozen interval on all outputs
but NeuroCombat. The amendment 6 values above include adapted HCLD.

Amendment 13, 2026-10-06 (written and committed before any HACA3 output was scored by any
probe; requested by P. Coupé and an external review: does the audit hold for a faithful,
published harmonizer?). Method: HACA3 (Zuo et al., Comput. Med. Imaging Graph. 2023) as published, with the
authors' code (github.com/lianruizuo/haca3, commit a1e0bd8) and public pretrained weights
(`harmonization_public.pt` SHA-256 a002391f..., `fusion.pt` 0cff9362...), no retraining or tuning.
Preprocessing HACA3 requires: N4 bias correction and registration to MNI152NLin2009cAsym 1 mm
(TemplateFlow, cropped to HACA3's 192x224x192 grid). Registration is rigid, with Mattes mutual
information over the dilated template brain and a multi-start search; a feasibility check on five
subjects (images and PSNR only, no probe) showed that affine registration stretched UM's
partial-coverage slabs by 40% along one axis and that a brain-masked similarity transform shrank
heads. Each output is resampled back onto the native grid with the inverse transform, normalized
like the raw volumes and cut at the frozen slices with the raw head mask and crop
(`scripts/methods/haca3_abide.py`; driver `scripts/vpulab_isbi2027_haca3.sh`). Target: the NYU
training volume whose mean HACA3 contrast code (theta) is the medoid of the 147 NYU training
volumes' codes. All 1,112 subjects are processed, NYU included; train/val slices are those of the
histogram-matching exports.
Analyses, all with the existing evaluator and bootstrap: (a) change, target alignment, frozen-probe
source BA and share labelled NYU; (b) the benchmark-recipe (seeds 42, 1-4) and converged (seeds
5-9) raw probes, retrained with the original recipes and seeds
(`scripts/vpulab_isbi2027_haca3_probes.sh`) and evaluated on the eight test artifacts as a
reproduction check and on HACA3; (c) whole-head slice probes (amendment 2 recipe,
seeds 1-3) trained on raw slices and on HACA3 outputs, with the paired difference of amendment 9.
Brain-only and histogram probes are not part of this amendment. Kendall tau and
seed-spread statistics stay over the original nine outputs. Reporting is fixed now: HACA3 replaces
the DLEST-style 1500 row in Table 1 whatever its results (that row stays in the results files),
and HACA3's results are reported in the paper whatever they show.
Outcome (2026-10-07; runs in `results/isbi2027/haca3/runs`, collected in
`analysis/haca3_runs_long.csv`; target NYU_51127): (a) PSNR 18.9 dB, XCorr 0.954, W 0.062,
Delta W_NYU -0.016 [-0.022, -0.011] (16% of the raw distance), KL to NYU 2.29 against 0.30 raw
(HACA3 renders CSF and the scalp-brain gap near zero: a median 14% of raw head pixels fall below
0.02); frozen-probe source BA 0.25 [0.18, 0.32], 40% of source outputs labelled NYU, 36 of 67
errors named NYU. (b) Retrained raw probes on HACA3: benchmark recipe 0.29 [0.18, 0.37], converged
0.28 [0.23, 0.33] (mean [min, max]). Reproduction on the eight common outputs: converged probes
matched the originals closely (raw BA 1.00; per-output differences median 0.02, max 0.085; Kendall
tau with the original orderings 0.86-1.00); benchmark-recipe probes did not (raw BA 0.91-0.97
against 0.81-0.98; per-output differences median 0.09, max 0.32), the training instability the
paper reports. (c) Whole-head slice probes on HACA3 test outputs (three-seed mean, best checkpoints):
raw-trained 0.28, HACA3-trained 0.75, paired difference +0.47 [0.43, 0.52]; final epoch +0.61
[0.57, 0.66]. Raw-trained slice probes reached 0.91-0.98 on raw test slices
(`analysis/probe_difference_haca3_{best,last}.json`).

Amendment 14, 2026-10-07 (post hoc, written before running; prompted by an internal review of
amendment 13). HACA3-trained image probes (0.75) fall inside the range of probes that see only
head silhouettes (0.73-0.86), so their accuracy may come from head geometry, not intensity. Test:
the amendment 5 intensity-only probe (logistic regression on foreground intensity histograms,
foreground = raw slice > 0.02, C chosen on validation) trained on HACA3 training outputs and on
raw training slices, tested on HACA3 test outputs with the evaluator's bootstrap
(`scripts/isbi2027_histogram_probe.py --methods haca3`). Brain-only probes are not part of this
amendment. Reported whatever the result.
Outcome (`results/isbi2027/analysis/histogram_probe_haca3.json`): the raw-trained histogram probe
reproduced amendment 5 on raw heads (0.742 [0.650, 0.835]) and gave 0.19 [0.13, 0.25] on HACA3; the
HACA3-trained histogram probe gave 0.71 [0.63, 0.80] (validation 0.76), +0.52 [0.43, 0.62] over the
raw-trained one. HACA3 outputs keep site information in intensities alone, not only in head shape.

Amendment 15, 2026-10-07 (written before running; prompted by an external review). Preprocessing-only
control for HACA3: the N4-corrected, MNI-registered volume that HACA3 receives is mapped back to the
native grid and cut at the frozen slices exactly like HACA3's output, without HACA3
(`haca3_abide.py export --source preproc`, output `haca3_preproc`). HACA3's own 95th-percentile
scaling and background removal are part of the harmonizer and are not applied. Scored, paired with
HACA3 on the same test subjects: the frozen probe, the ten retrained raw probes of amendment 13 and
its three raw-trained slice probes; the intensity-only probe of amendment 14, raw-trained and
trained on the control's own outputs. Question: how much of HACA3's low raw-trained-probe accuracy
the preprocessing alone produces. Reported whatever the result.
Outcome (`results/isbi2027/haca3/runs_ctrl`, `analysis/haca3_ctrl_runs_long.csv`,
`analysis/histogram_probe_haca3ctrl.json`): the control kept PSNR 23.8 dB and XCorr 0.985; frozen
source BA 0.61 [0.53, 0.68] (HACA3 0.25; paired HACA3 minus control -0.35 [-0.44, -0.26]), 9% of
source outputs and 8 of 34 errors labelled NYU (HACA3 40%, 36 of 67); benchmark-recipe probes 0.67
[0.64, 0.68], converged 0.80 [0.65, 0.87], raw-trained slice probes 0.70 [0.44, 0.87] (HACA3 0.29,
0.28, 0.28). Intensity-only: raw-trained 0.24 on the control (0.19 on HACA3), control-trained 0.67
[0.59, 0.74]. Preprocessing alone accounts for about a third of the frozen-probe drop and for most
of the raw-trained histogram probe's failure; HACA3 accounts for the rest of the image-probe drop
and for the shift of errors toward NYU.

Amendment 16, 2026-10-07 (written before running; prompted by the same review: 90 source test
subjects). HACA3 never saw ABIDE, so its outputs for all 1,112 subjects are untouched. Five-fold
cross-validation over all subjects (folds stratified by site, seed 20261007; within the training
folds a stratified 10% is the validation set): whole-head slice probes (amendment 2 recipe, best
validation epoch, one seed per fold) and intensity-only probes trained on raw slices, on HACA3
outputs and on the amendment 15 control, each tested on its held-out fold; raw-trained probes are
also tested on HACA3 and control outputs. Source BA pools the held-out predictions of all non-NYU
subjects, with 2,000 within-site bootstrap replicates (seed 20260913), paired across probes. The
frozen probe is not evaluated (it was trained on these subjects). `scripts/isbi2027_haca3_cv.py`.
Outcome (`analysis/haca3_cv.json`, 1,112 subjects, 928 source): raw-trained image probes 0.95
[0.93, 0.97] on raw slices, 0.25 [0.23, 0.28] on HACA3 and 0.57 [0.54, 0.60] on the control;
probes trained on HACA3 0.77 [0.74, 0.80] (+0.52 [0.48, 0.56] over raw-trained) and on the control
0.92 [0.90, 0.94]. Intensity-only: raw-trained 0.71 on raw, 0.18 on HACA3, 0.20 on the control;
HACA3-trained 0.66 [0.63, 0.69] (+0.48 [0.45, 0.52]), control-trained 0.74 [0.71, 0.77]. The
held-out test-set results are reproduced on all subjects.

Amendment 17, 2026-10-07 (written before running; requested by the author). Brain-only probes for
HACA3, as amendment 7 for the other methods: HD-BET masks for all 1,112 raw volumes
(`run_hdbet_masks.py`, hd-bet 2.0.1), brain-only
exports built from the HACA3 export (`make_brain_npz.py --methods haca3 --reference haca3`, gated on
exact raw-slice reproduction), whole-brain slice probes (amendment 2 recipe, seeds 1-3) trained on
masked raw slices and on masked HACA3 outputs and evaluated on HACA3 test outputs with brain-only
probe inputs, the paired difference of amendment 9, and the brain-foreground intensity-only probe.
Driver `scripts/vpulab_isbi2027_haca3_brain.sh`. Reported whatever the result.
Outcome (`results/isbi2027/haca3/runs_brain`, `analysis/haca3_brain_runs_long.csv`,
`analysis/probe_difference_haca3_brain_{best,last}.json`, `analysis/histogram_probe_haca3_brain.json`):
the exports reproduced every raw slice exactly; raw-trained brain probes reached 0.83 on raw brains
(original amendment 7 probes 0.85) and 0.14 on HACA3 brains, HACA3-trained 0.49, paired difference
+0.35 [0.30, 0.40] (final epoch +0.22 [0.18, 0.26]). Brain intensity histograms: raw-trained 0.56 on
raw brains (as amendment 7) and 0.10 on HACA3, HACA3-trained 0.48 [0.39, 0.58], +0.39 [0.29, 0.49].

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
