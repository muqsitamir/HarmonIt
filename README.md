# HarmonIt

### MRI harmonization, with anatomy as the constraint.

**How much scanner variation can we remove before we change the patient?**

HarmonIt investigates this question through a multi-site brain MRI benchmark
and the development of diffusion-based harmonization. We study the trade-off
between reducing site predictability, preserving individual anatomy, and
producing consistent volumes that remain useful for downstream analysis.

**1,112 subjects | 17 acquisition sites | ABIDE I T1-weighted MRI**

[Research roadmap](#from-benchmark-to-25d-diffusion) |
[Evaluation definitions](documentation/METRICS.md) |
[Installation](documentation/INSTALLATION.md) |
[Baseline implementations](https://github.com/muqsitamir/HarmonIt/tree/codex/add-stargan-baseline/scripts/methods)

> **Research status:** first-phase baseline experiments are complete; their
> evaluation is being audited. The next contribution is a planned **2.5D
> diffusion model**. No 2.5D training results or clinical benefit are claimed yet.

## Why This Matters

MRI scans carry signatures of the scanner, acquisition protocol, and site.
Models can exploit those signatures and struggle when applied elsewhere.
Harmonization aims to reduce acquisition variation while preserving the
biological information that matters.

The difficult part is deciding whether it worked. A blurred or distorted image
can fool a site classifier. An unchanged image can achieve excellent similarity
scores. Matching an intensity histogram does not establish anatomical fidelity.
HarmonIt evaluates these questions together and makes the limitations explicit.

## From Benchmark to 2.5D Diffusion

| Phase | Contribution | Status |
| --- | --- | --- |
| TRDP1: establish the comparison | Common slice pipeline, subject splits, site probe, baseline exports and qualitative comparisons | Experiments completed; historical metrics under review |
| Evaluation repair | Correct metric semantics, separate translated subjects from target identities, trace generating code and artifacts | Local corrections and audit in progress |
| TRDP2: use neighbouring anatomy | Site-conditioned diffusion with adjacent slices as context and a harmonized centre-slice output | Planned; initial design defined |
| Volumetric validation | Full-volume export, slice continuity, segmentation consistency and regional-volume analysis | Planned |

The first 2.5D experiment will compare **one-slice input against three-slice
context** (`z-1`, `z`, `z+1`) using the same diffusion backbone as far as possible.
Five-slice context and anatomy constraints follow only when the initial
comparison motivates them. Stacking slice predictions into a volume does not
guarantee 3D consistency: that is an outcome we will measure.

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

The first phase explored the following families. Historical rankings are
provisional until evaluated with the corrected, documented protocol.

| Baseline | Role in the comparison |
| --- | --- |
| NeuroCombat | Statistical correction of slice-pixel features |
| Histogram matching | Lightweight intensity-distribution baseline |
| CycleGAN | Unpaired translation from pooled non-NYU sites to NYU |
| StarGAN | Multi-domain translation with conservative/aggressive settings |
| DLEST-style model | Disentangled content/style baseline; 1,000- and 1,500-step variants |
| Diffusion img2img | Site-conditioned 2D diffusion; completed 20,000-step experiment |
| Adapted HCLD | Volumetric latent diffusion adapted to available 40 GB A100 memory |

**Where is the code?** `main` currently contains the core pipeline, evaluator,
and NeuroCombat implementation. The additional methods, HCLD configurations,
figure scripts, and Slurm launchers are on
[`codex/add-stargan-baseline`](https://github.com/muqsitamir/HarmonIt/tree/codex/add-stargan-baseline).
See its [baseline notes](https://github.com/muqsitamir/HarmonIt/blob/codex/add-stargan-baseline/documentation/BASELINES.md)
and [reproduction commands](https://github.com/muqsitamir/HarmonIt/blob/codex/add-stargan-baseline/documentation/REPRODUCIBILITY.md).
Those historical notes should be read alongside the current metric definitions;
implementation adaptations and run provenance still need consolidation.

## What Counts as Progress?

We ask three separate questions:

1. **Does site information remain recoverable?** Frozen-probe balanced accuracy
   measures one classifier's response. Retrained probes are planned to test
   whether apparent removal survives a new classifier.
2. **What changed in the image?** PSNR, pixel cosine similarity, cross-correlation,
   and intensity-distribution changes describe preservation proxies. They do
   not establish anatomical accuracy or closeness to NYU.
3. **Does the result remain useful as a volume?** Planned tests assess slice
   continuity, segmentation agreement, regional volumes, and selected downstream
   or unseen-site outcomes.

The audit identified output-dependent PSNR scaling, inconsistent historical KL
histogram coordinates, and target identities mixed into pooled preservation
scores. The local corrected evaluator is not yet a validated benchmark release.
We will publish versioned results rather than silently replace old numbers.
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

For the additional baseline implementations, use the research branch **in a
clean checkout**:

```bash
git switch --track origin/codex/add-stargan-baseline
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
| [Research direction](#from-benchmark-to-25d-diffusion) | Completed benchmark work and planned 2.5D comparison |
| [Metrics](documentation/METRICS.md) | Implemented quantities, interpretation, and planned corrections |
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
