# HarmonIt: submission gap audit and execution plan

Audit date: 2026-09-12. Status: local evidence reviewed; historical rankings provisional.

> Subsequent steering, 2026-09-12: the active plan now prioritizes 2.5D diffusion
> development with a bounded evaluation repair in parallel. See
> [TRDP2_PLAN.md](TRDP2_PLAN.md), which supersedes this report's original
> evaluation-first schedule. Git inspection also located the additional baseline
> code, including the 2D diffusion implementation, on a research branch; it has
> since been merged into `main`.
> Checkpoint and historical run provenance still require verification. The
> scientific findings below remain applicable. Local evaluator corrections
> described here are not included in the documentation-only publication commit.

## Research decision

The primary research question is: **Does apparent scanner harmonization preserve
individual anatomy and improve generalization to unseen acquisition sites?**
This supports a benchmark and validation contribution even if a new generator
does not outperform existing methods. A 2.5D diffusion extension is a gated
secondary contribution. It must demonstrate added value beyond changing the
number of input channels.

This report builds on `TRDP2_PLAN.md`. Existing working-tree changes were present
before this audit and have been preserved. No training, remote synchronization,
historical-results replacement, manuscript edits, or publication took place.

## Evidence inspected and limits

- Local manuscript source: `paper/isbi2027/main.tex`. The previously supplied
  `/Users/apple/Downloads/HarmonIt_Manuscript.pdf` is absent, so this audit does
  not certify the latest PDF or its layout.
- Current evaluator, metric implementations, label-shuffle changes, baseline
  script, and five evaluation regression tests.
- Local NeuroCombat and adapted HCLD test NPZs and historical summary files.
- Manifest: 1,112 rows; split lists: 890 train, 113 validation, 109 test.
- Local HEAD: `04e33ed` (`added code to evaluate neurocombat`), with pre-existing
  modifications and untracked files. This is not an identified complete release.
- Only NeuroCombat method scripts are present in the local methods directory;
  the local checkpoint directory is empty. Recover remote training code,
  configs, checkpoints, and logs before asserting reproducibility of all methods.

## Submission blockers

| Priority | Evidence | Consequence and required action |
| --- | --- | --- |
| P0 | `compute_distribution_metrics` compares raw and harmonized arrays; saved keys include `kl_raw_to_harmonized`. | Withdraw the interpretation that these numbers demonstrate NYU matching. Retain them as intensity-change measures; add separately named target-alignment metrics. |
| P0 | Manuscript describes brain-masked PSNR/XCorr, VGG pool3 features, and symmetrized KL. Evaluator uses whole-image pixels, pixel cosine, and directional KL. | Align descriptions to the generating implementation. A conservative head mask is not a brain segmentation. Adding new metrics requires a new result version. |
| P0 | Historical PSNR code inferred range from output; historical KL used independently generated histogram edges. Existing uncommitted fixes address both. | Recompute with fixed PSNR scale and common histogram coordinates. Record exact evaluator revision and do not silently overwrite old tables. |
| P0 | All 19 NYU HCLD outputs are exactly identical to embedded raw images. Historical pooled PSNR is 36.992 dB, median 12.711 dB. | Separate non-NYU translation from target identity behavior. Do not interpret the pooled PSNR as evidence of preservation. |
| P0 | Old shuffle code permuted class IDs, rather than subject labels. Current code contains a subject-level correction. | Retrain and evaluate actual shuffled-label controls. A low score after relabeling an existing classifier does not establish a valid random-label control. |
| P0 | Local NeuroCombat call fits selected-split pixels jointly with site covariates and no `ref_batch`; saved config selects test. | Describe this as a transductive cohort correction unless remote provenance establishes otherwise. It is not demonstrated to be train-only NYU-target translation. Establish separate comparison tracks or an appropriate inductive implementation. |
| P1 | Frozen-probe failure can reflect distribution shift, and site labels can correlate with biological variables. | Add probes retrained on harmonized training outputs; examine demographics and biology preservation. Avoid equating low frozen-probe BA with removal of scanner effects. |
| P1 | Manuscript calls HCLD initialized from published weights; local training/checkpoint evidence is missing. | Recover initialization, upstream revision/license, adaptation diff, settings, and logs. Treat the initialization claim as unverified. |
| P1 | Probe sanity table is captioned validation, but its 0.9817/0.9510 values match the saved test evaluation. | Verify every row's split, checkpoint, and protocol. Numerical equality is a warning, not proof of mislabeling. |
| P1 | No complete local record of validation decisions or repeated-seed results. | Reconstruct selection history. Freeze future selection rules before inspecting new test outcomes. Existing test results are exploratory if they influenced tuning. |

### Local artifact diagnostic

These are arithmetic checks against embedded raw arrays, not a fresh benchmark
evaluation against source MRI. Both files contain 109 unique subjects, 19 with
site ID 5, finite outputs, and arrays of shape `[109,1,256,256]`.

| Check | Adapted HCLD | NeuroCombat |
| --- | ---: | ---: |
| Exactly unchanged NYU subjects | 19/19 | 0/19 |
| Exactly unchanged non-NYU subjects | 0/90 | 0/90 |
| Non-NYU fixed-scale mean PSNR (dB) | 13.9407 | 19.9988 |
| Non-NYU fixed-scale median PSNR (dB) | 13.5854 | 20.5512 |
| Non-NYU mean squared error | 0.0460084 | 0.0113081 |
| Output minimum / maximum | 0 / 1 | -0.183824 / 1.224955 |

Calculation: per-subject MSE over all pixels, then `-10*log10(MSE)` with
`data_range=1`; source-only summary selects `site_id != 5`. These diagnostics
must not replace publication numbers until raw-image identity and provenance
are verified. NeuroCombat's out-of-range values were not clipped.

The current evaluator's `summarize()` silently excludes nonfinite values.
Since exact identities have infinite PSNR, a revised output must explicitly
count exact matches and finite observations. Do not substitute arbitrary caps
or compare means formed from different hidden subsets.

## First experiment batch: evaluation repair

1. Recover a complete artifact inventory from remote machines when available:
   method, upstream and local commits, split and subject IDs, preprocessing,
   training configuration, checkpoint hash, export command, initialization,
   target handling, and evaluator revision. Preserve failed HCLD logs too.
2. Validate subject/slice alignment, split disjointness, unique IDs, finite
   images, and identical raw references across methods. Confirm target-site
   mapping from the manifest. Count exclusions explicitly.
3. Produce a versioned evaluator with separate all-subject, non-NYU, and NYU
   summaries. Use fixed-scale PSNR, MSE/MAE, explicitly named pixel cosine,
   XCorr, and labeled whole-image versus foreground measurements. Masks must
   come from a fixed reference procedure, not each method's output.
4. Add target-alignment measures using a frozen NYU training reference with
   documented subject weighting. Choose histogram edges/smoothing from the
   training protocol and keep them common across methods; report out-of-range
   mass. Report raw-to-target alongside harmonized-to-target. Low intensity
   distance alone is not evidence of anatomical or biological correctness.
5. Re-evaluate all recovered outputs, preserving historical artifacts. Include
   per-subject/per-site results and paired, site-stratified subject bootstrap
   intervals (2,000 replicates, fixed recorded seed). Do not resample pixels as
   independent subjects. Compute source-only BA over source classes explicitly.
6. Retrain raw-image, shuffled-label, background-only, and mask-only probes
   with documented input generation and subject splits. Verify diagnostic
   inputs are nontrivial before interpreting chance performance. Then train
   independent probes on harmonized training outputs for selected methods.

Acceptance gate: every reported row has identifiable inputs and generating
code; metric tests pass; no unexplained missing subjects; target identities
are separate; claims accurately describe what each measure observes.

## Four-month programme

| Weeks | Work and output | Decision gate |
| --- | --- | --- |
| 1-2 | Recover provenance; complete evaluator and regenerate existing results; reconcile paper claims. | Correctness and release completeness before additional model selection. |
| 3-4 | Retrained probes, valid shortcut controls, uncertainty estimates, selection audit; define anatomy and external-site protocols. | Freeze protocol and select baselines using validation evidence. |
| 5-8 | Export coherent volumes for selected methods; assess segmentation agreement, regional volumes, slice continuity, and unseen-site transfer. | Confirm an anatomy/generalization failure worth addressing. |
| 9-12 | Conditional 2.5D diffusion prototype: adjacent-slice context and explicit anatomy/continuity constraint. Compare against the same 2D backbone. | Require reproducible benefit at comparable compute with acceptable anatomy preservation. |
| 13-16 | Repeated-seed final experiments, selected held-out evaluation, figures, limitations, reproducible release, journal manuscript. | Submit a supported contribution; remove unsupported claims regardless of model ranking. |

For anatomy evaluation, lock a segmentation pipeline and record failures/QC
on both raw and harmonized volumes. Dice between their automated segmentations
measures consistency, not accuracy. Manual or independently justified labels
are needed for accuracy claims. Regional volume shifts need demographic
context; preserving a raw segmentation can also preserve its errors.

Do not extrapolate one-slice results to 3D. Existing slice artifacts cannot
support volumetric segmentation. A slice method needs a deterministic full
volume export with consistent spatial coordinates and intensity handling.

For unseen-site evaluation, choose site holdouts using training-cohort
availability and metadata, not the current test ranking. Exclude held-out
sites from harmonizer fitting and tuning. Audit whether each method can handle
an unseen site without estimating parameters on its evaluation cohort; report
transductive methods separately. An external/travelling-subject dataset is
valuable if access and compatible acquisitions can be secured.

The 2.5D experiment should include 2D versus adjacent-slice input and constraint
on/off ablations, matched data and compute reporting, and at least three seeds
for final leading configurations if feasible. Volume evaluation, anatomical
consistency, and held-out-site transfer determine whether it merits a method
claim. The benchmark paper remains viable as a research direction without it;
acceptance cannot be promised.

## Journal decision and publication checklist

Journal of Neuroscience Methods remains the provisional first choice for a
validated neuroscience evaluation contribution; Computer Methods and Programs
in Biomedicine is the provisional alternative. This is a positioning decision,
not a completed editorial-fit assessment.

Official scope pages attempted on 2026-09-12 returned HTTP 403:
- https://www.sciencedirect.com/journal/journal-of-neuroscience-methods/about/aims-and-scope
- https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine/about/aims-and-scope

Before committing to a venue, verify current article types, recent comparable
papers, length/supplement limits, data/code policies, and publication costs
against official guidance. Do not infer current requirements from the local
ISBI template, whose README references 2021 despite the directory name.

Required manuscript package:
- Accurate metric definitions and an explicit inductive/transductive protocol.
- Traceable method implementations and adaptations with verified licenses.
- Probe validation, uncertainty, anatomy evidence, and generalization tests.
- Selection rules, target-site rationale grounded in metadata, and limitations.
- Reproducible commands/configurations, environment, artifact hashes, data access
  instructions, and a clean-checkout reproduction of a small evaluation.

## Next action

The immediate next implementation is the versioned evaluator and artifact
provenance inventory above. Remote availability need not block local evaluator
work.

Verification in this audit: all five existing tests pass using
`/Users/apple/anaconda3/envs/harmonit311/bin/python -m unittest discover -s tests -v`.
No full benchmark or new GPU experiment has been run.
