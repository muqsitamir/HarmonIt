# HarmonIt

### MRI harmonization, with anatomy as the constraint.

**How much scanner variation can we remove before we change the patient?**

HarmonIt investigates this question through a multi-site brain MRI benchmark
and the development of a harmonization method. We study the trade-off
between reducing site predictability, preserving individual anatomy, and
producing consistent volumes that remain useful for downstream analysis.

**1,112 subjects | 17 acquisition sites | ABIDE I T1-weighted MRI**

[Research roadmap](#research-roadmap) |
[Evaluation definitions](documentation/METRICS.md) |
[Installation](documentation/INSTALLATION.md) |
[Baseline implementations](scripts/methods) |
[ISBI 2027 evaluation audit](#isbi-2027-evaluation-audit)

> **Research status:** the first-phase benchmark and its evaluation audit are
> complete. The audit manuscript for ISBI 2027, its written protocol, code and
> versioned per-subject results are in this repository. The next contribution is a
> harmonization method designed and judged against the audit's criteria. It is in
> development and not yet part of this repository; no method results or clinical
> benefit are claimed.

## ISBI 2027 Evaluation Audit

**Frozen Site-Probe Accuracy Is Not a Standalone Harmonization Score: An Audit of
Evaluation Practice for Multi-Site MRI.** Manuscript for ISBI 2027:
[PDF](paper/isbi2027/main.pdf) | [LaTeX source](paper/isbi2027/main.tex)

The audit uses ten outputs from seven harmonization methods, one of them HACA3 with its
published pretrained weights, plus a blur control, as test cases for the evaluation, not as a
leaderboard, on the 90 source-site test subjects:

- **The verdict depends on how the probe is trained.** Rerunning the frozen probe's recipe
  with new random seeds gave the same CycleGAN outputs balanced accuracies of 0.13-0.59;
  probes trained to convergence agreed (0.44-0.54) but found more site information than the
  frozen probe in eight of ten outputs.
- **A blur fools the frozen probe.** A Gaussian blur of the inputs, which moves intensities
  away from NYU and keeps PSNR at 24.7 dB, brought the frozen probe to 0.07, within the
  shuffled-label range; the probe labelled the blurred slices Yale, none NYU.
- **Site remains decodable after harmonization.** Probes trained on harmonized outputs still
  decoded the source site from whole heads, brain-only slices and intensity histograms. For
  HACA3, the frozen probe scored 0.25, a probe trained on its outputs 0.75 and an
  intensity-histogram probe 0.71; five-fold cross-validation over all 928 source subjects and a
  preprocessing-only control agree.
- **High similarity does not imply alignment.** Diffusion changed intensities least but barely
  moved them toward the target site.

| Resource | Contents |
| --- | --- |
| [Protocol](documentation/ISBI2027_PROTOCOL.md) | Cohort, metrics, statistics and dated amendments, each written before its outcome was seen |
| [Results](results/isbi2027/README.md) | Versioned per-subject metrics, probe predictions and bootstrap summaries behind every number in the paper |
| [Evaluator](scripts/eval_isbi2027.py) | Subject-level evaluation with fixed-scale PSNR, target alignment and a paired site-stratified bootstrap |
| [Reproducibility](documentation/REPRODUCIBILITY.md) | Environment and reproduction commands |

### Details not in the paper

The four-page paper keeps the main numbers; everything else is here.

| Detail | Where |
| --- | --- |
| Probe training settings (optimizer, learning rate, batch size, steps, augmentation) and the converged recipe | [Protocol](documentation/ISBI2027_PROTOCOL.md): "Probe experiments" and amendment 6 |
| Harmonization method settings | [Baselines](documentation/BASELINES.md); HACA3 preprocessing in protocol amendment 13 |
| Kendall tau between probe rankings and per-output seed ranges | [`kendall_tau_nine_outputs.json`](results/isbi2027/analysis/kendall_tau_nine_outputs.json) |
| 95% confidence intervals for every reported number | per-run `*_summary.json` in [`results/isbi2027/runs/`](results/isbi2027/runs) and the files in [`results/isbi2027/analysis/`](results/isbi2027/analysis) |
| Per-subject predictions and metrics | per-run `*_subjects.csv` in [`results/isbi2027/runs/`](results/isbi2027/runs) |
| Demographics probe (age, sex, diagnosis, brain volume) | protocol amendment 11; [`demographics_probe.json`](results/isbi2027/analysis/demographics_probe.json) |
| Blur control at sigma 4 and 8 | protocol amendment 12; [`frozen_probe_blur_control_20261005`](results/isbi2027/runs/frozen_probe_blur_control_20261005) |
| HACA3 preprocessing-only control, cross-validation, brain-only and intensity-only probes | protocol amendments 13-17; [`results/isbi2027/haca3/`](results/isbi2027/haca3) and `results/isbi2027/analysis/*haca3*` |

## Why This Matters

MRI scans carry signatures of the scanner, acquisition protocol, and site.
Models can exploit those signatures and struggle when applied elsewhere.
Harmonization aims to reduce acquisition variation while preserving the
biological information that matters.

The difficult part is deciding whether it worked. A blurred or distorted image
can fool a site classifier. An unchanged image can achieve excellent similarity
scores. Matching an intensity histogram does not establish anatomical fidelity.
HarmonIt evaluates these questions together and makes the limitations explicit.

## Research Roadmap

| Phase | Contribution | Status |
| --- | --- | --- |
| Benchmark: establish the comparison | Common slice pipeline, subject splits, site probe, baseline exports and qualitative comparisons | Complete; outputs re-evaluated in the ISBI 2027 audit |
| Evaluation audit (ISBI 2027) | Correct metric semantics, separate translated subjects from target identities, retrain and converge site probes, train probes on outputs, measure target alignment | Complete; manuscript, protocol and results in this repository |
| Harmonization method | A method designed against the audit's criteria: less site information recoverable by probes trained on its outputs, anatomy preserved, movement toward the target site, and consistent volumes | In development; code is added here when the method is submitted |
| Volumetric validation | Full-volume evaluation: slice continuity, segmentation consistency and regional volumes | Planned, with the method |

An earlier plan for a 2.5D diffusion model, with adjacent slices as context, was not
pursued: adjacent-slice harmonization already exists, and the audit pointed to a
different gap.

## Benchmark Design

| Component | Current benchmark |
| --- | --- |
| Data | Raw ABIDE I structural T1 MRI; 1,112 subjects from 17 sites |
| Subject splits | 890 training / 113 validation / 109 test |
| Slice representation | Canonical RAS orientation, robust normalization, axial selection, conservative head-mask cropping, background handling, 256 x 256 resize |
| Reference domain | NYU (`site_id=5`, 184 subjects) for target-conditioned methods |
| Site probe | Single-channel ResNet-18, frozen for raw/harmonized comparison |
| Evaluation unit | Deterministic subject slices; volumetric validation is planned |

NYU was chosen as the largest site in this cohort. It is a reference acquisition
domain, not an anatomical ground truth. The head mask is not a brain tissue
segmentation. Method-specific fitting and target handling must be recorded:
the existing NeuroCombat implementation performs cohort-level correction
without an explicit NYU reference, so it is not the same protocol as learned
NYU translation.

## Baseline Coverage

The first phase explored the following families. The ISBI 2027 audit re-evaluates
their outputs under the corrected protocol as test cases and does not rank them;
first-phase rankings are superseded.

| Baseline | Role in the comparison |
| --- | --- |
| NeuroCombat | Statistical correction of slice-pixel features |
| Histogram matching | Lightweight intensity-distribution baseline |
| CycleGAN | Unpaired translation from pooled non-NYU sites to NYU |
| StarGAN | Multi-domain translation with conservative/aggressive settings |
| DLEST-style model | Disentangled content/style baseline; 1,000- and 1,500-step variants |
| Diffusion img2img | Site-conditioned 2D diffusion; completed 20,000-step experiment |
| Adapted HCLD | Volumetric latent diffusion adapted to available 40 GB A100 memory (not in the ISBI paper) |
| HACA3 | Authors' pretrained model, used as released (`scripts/methods/haca3_abide.py`, protocol amendment 13) |

**Where is the code?** `main` contains the core pipeline, evaluators, all baseline
implementations (`scripts/methods/`), HCLD configurations, figure scripts and Slurm
launchers. See the [baseline notes](documentation/BASELINES.md) and
[reproduction commands](documentation/REPRODUCIBILITY.md). Those historical notes
should be read alongside the current metric definitions and the ISBI 2027 protocol.

## What Counts as Progress?

We ask three separate questions:

1. **Does site information remain recoverable?** Frozen-probe balanced accuracy
   measures one classifier's response. The ISBI 2027 audit adds retrained and
   converged probes, probes trained on harmonized outputs, and geometry-, intensity-
   and brain-only controls.
2. **What changed in the image?** PSNR, pixel cosine similarity, cross-correlation,
   and intensity-distribution changes describe preservation proxies. They do
   not establish anatomical accuracy or closeness to NYU.
3. **Does the result remain useful as a volume?** Planned tests assess slice
   continuity, segmentation agreement, regional volumes, and selected downstream
   or unseen-site outcomes.

The audit identified output-dependent PSNR scaling, inconsistent historical KL
histogram coordinates, and target identities mixed into pooled preservation
scores. The corrected subject-level evaluator (`scripts/eval_isbi2027.py`) and its
versioned results are in [`results/isbi2027`](results/isbi2027/README.md), under the
[written protocol](documentation/ISBI2027_PROTOCOL.md); old numbers are not silently replaced.
Read the [metric definitions](documentation/METRICS.md) for exact scope and
limitations.

## Get Started

```bash
git clone https://github.com/muqsitamir/HarmonIt.git
cd HarmonIt
conda create -n harmonit311 python=3.11 -y
conda activate harmonit311
```

Install the CPU or CUDA PyTorch build appropriate for your machine using the
[installation guide](documentation/INSTALLATION.md), then:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py
```

Obtain ABIDE data separately under its access terms. The manifest builder
expects the directory layout described in the
[preprocessing guide](documentation/PREPROCESSING.md). Published experiment
reproduction also requires matching split files, checkpoints, and export
settings; installing the package alone does not reproduce the result table.

An exported fixed-slice artifact carries `images`, `subject_ids`, `site_ids`,
and `slice_indices`, with `raw_images`, `split`, and `method` metadata where
available. Images have shape `[N, 1, 256, 256]`. Inspect the evaluator interface:

```bash
python scripts/eval_harmonized_npz.py --help
```

## Explore the Project

| Resource | Contents |
| --- | --- |
| [Research roadmap](#research-roadmap) | Completed benchmark and audit, and the next step |
| [Metrics](documentation/METRICS.md) | Metric definitions, interpretation and limits |
| [ISBI 2027 audit](#isbi-2027-evaluation-audit) | [Manuscript](paper/isbi2027/main.pdf), [protocol](documentation/ISBI2027_PROTOCOL.md), [results](results/isbi2027/README.md) |
| [Baselines](documentation/BASELINES.md) and [reproducibility](documentation/REPRODUCIBILITY.md) | Method notes and reproduction commands |
| [Preprocessing](documentation/PREPROCESSING.md) | ABIDE-to-slice pipeline |
| [Installation](documentation/INSTALLATION.md) | Local and GPU setup |
| [Containers](documentation/CONTAINERS.md) | Container configuration |

## Research Use and License

HarmonIt is research software. Preservation of pathology, downstream benefit,
and unseen-scanner generalization have not yet been established. There is no
claimed journal acceptance or validated clinical deployment.

Repository code is provided under [Apache 2.0](LICENSE). Datasets, upstream
implementations, and pretrained weights retain their own licenses and access
conditions.
