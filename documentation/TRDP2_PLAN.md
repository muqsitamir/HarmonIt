# TRDP2: 2.5D diffusion for anatomy-preserving MRI harmonization

Updated 2026-09-12. This is the current research direction and supersedes the
earlier evaluation-first, new-method-as-stretch schedule in the submission audit.

## Research question

Does neighbouring-slice context improve anatomical preservation and volumetric
consistency over a matched 2D diffusion harmonizer, while retaining useful
reduction in site predictability?

TRDP1 supplies preprocessing, subject splits, baseline implementations, model
artifacts, and practical knowledge of failure modes. TRDP2 reuses this work.
The main investment is a 2.5D diffusion method; a bounded evaluation repair runs
alongside development. A complete retraining of all historical baselines is
not a prerequisite.

## Starting point

The 2D implementation is already versioned at
[`scripts/methods/diffusion_img2img.py`](https://github.com/muqsitamir/HarmonIt/blob/main/scripts/methods/diffusion_img2img.py).
This was confirmed from Git during the documentation update. Recovery of its
source no longer depends on remote-server availability; checkpoints and exact
training/export provenance still require verification.

The existing baseline uses a site-conditioned DDPM with an EMA model and an
image-to-image export path. The historical 20,000-step run is a starting point,
not a substitute for a controlled, matched 2D comparison.

## First implementation

1. Establish an isolated development branch from the baseline implementation;
   identify its config, checkpoint, preprocessing, and dependencies.
2. Extend data loading to adjacent axial slices from the same subject. Use
   shared crop and spatial augmentation across each slice stack. Define and
   record volume-edge handling, axial spacing, and physical slice coverage.
3. Supply three-slice context `[z-1, z, z+1]` and predict the centre slice. Specify
   which inputs are noised versus supplied as conditioning; apply the same
   convention at training and inference. Neighbour context can also carry site
   style, so conditioning design must be tested rather than assumed beneficial.
4. Run a several-hundred-step smoke with finite losses, checkpoint reload,
   sample exports, peak VRAM, and measured throughput.
5. Export a complete volume with recorded geometry and consistent normalization
   and spatial transforms. Inspect axial, coronal, and sagittal views for
   boundaries, discontinuities, and subject-identity changes.

Acceptance: a reloadable model, reproducible export, correctly aligned volume,
finite outputs, and recorded resource use. A smoke verifies the pipeline;
it does not establish scientific improvement.

## Controlled comparison

| Experiment | Purpose | Status |
| --- | --- | --- |
| Matched 2D backbone | Establish the reference under the new protocol | Planned |
| Three-slice context | Isolate benefit of immediate neighbours | First 2.5D experiment |
| Five-slice context | Test whether wider context helps | Conditional follow-up |
| Anatomy/continuity constraint on/off | Address a demonstrated failure mode | Conditional follow-up |

Hold subjects, target site, validation selection, normalization, export settings,
and backbone capacity as comparable as possible. Record architectural changes,
updates, effective batch size, wall time, and peak GPU memory. Neighbour spacing
in slices is not uniform physical spacing across scans; document this and
consider spacing-aware sampling as an explicit design choice.

Select settings using validation data. Do not select noise strength, checkpoints,
or constraints on the final test ranking. Repeat the leading configurations
with multiple seeds before the final comparison.

## Evaluation work in parallel

The existing local audit contains fixed-scale PSNR, shared KL histogram edges,
pixel-cosine naming, and subject-level shuffled-label corrections. These changes
are not included in this documentation-only release and require a reviewed
implementation commit and versioned re-evaluation.

Minimum needed for the 2D/2.5D comparison:
- Validate subject, slice, geometry, and raw-reference alignment.
- Separate non-NYU translation metrics from unchanged NYU identity outputs.
- Use fixed PSNR scale and report exact identities and finite-value counts.
- Distinguish raw-to-harmonized intensity change from NYU-target alignment.
- Define foreground masks consistently; do not call head masks brain masks.
- Export per-subject results and estimate uncertainty at the subject level.

Re-evaluate saved outputs where possible. Retraining is reserved for the matched
2D control and cases where the scientific comparison requires it. Broader probe
controls and additional baseline re-runs are scaled to the paper's claims.

## Volumetric evidence

Stacked predictions alone do not establish 3D consistency. Evaluate anatomical
continuity with full-volume qualitative views and quantitative measures that
do not simply reward blur. Combine these with preservation metrics and fixed
segmentation-pipeline QC.

Dice between raw and harmonized automatic segmentations measures consistency,
not ground-truth accuracy. Regional volume changes and failures must be reported.
An unseen-site or downstream experiment strengthens the contribution, but its
scope is selected after the initial 2D/2.5D result and data availability review.

## Working schedule

| Period | Deliverable |
| --- | --- |
| Weeks 1-2 | Baseline provenance, minimal 2.5D data/model extension, evaluator corrections, smoke and volume export |
| Weeks 3-6 | Matched 2D versus three-slice experiments; initial preservation and continuity analysis |
| Weeks 7-10 | Targeted context/constraint ablations; segmentation and regional-volume evaluation |
| Weeks 11-13 | Leading-run repeats and a selected generalization/downstream experiment if feasible |
| Weeks 14-16 | Final analysis, supervisor review, reproducible release, manuscript submission |

Working allocation: approximately two-thirds method development and experiments,
one-third evaluation and writing. Internal submission target: 15 January 2027;
this is not a journal deadline or an acceptance forecast.

## Decision gates

- If the smoke fails, fix the identified pipeline or memory issue before scaling.
- If context does not help, inspect its conditioning and physical coverage before
  adding complexity. Record a negative result rather than select flattering slices.
- If improvements are explained by smoothing, they do not support an anatomy claim.
- A method contribution requires evidence beyond additional input channels:
  a justified design, controlled ablations, and meaningful measured benefit.
- Journal choice follows the validated contribution. Journal of Neuroscience
  Methods and Computer Methods and Programs in Biomedicine remain provisional.

See [the audit](SUBMISSION_GAP_REPORT.md) for historical evidence gaps and
[metric definitions](METRICS.md) for interpretation boundaries.
