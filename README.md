# HarmonIt

Unified system to harmonize MR images across sites, scanners, and protocols while preserving anatomy and pathology.

## Overview
Multi-site MRI datasets often suffer from **domain shift** (scanner/vendor/protocol/site differences). HarmonIt aims to learn an **image-level harmonization** mapping that reduces these non-biological variations while keeping clinically meaningful information intact.

## Objectives
- **Harmonize** MR images to reduce site/scanner/protocol variability
- **Preserve anatomy** (structures, boundaries, geometry)
- **Preserve pathology** (lesions/tumors/abnormalities should not be removed or hallucinated)
- **Improve downstream robustness** (segmentation/classification should not degrade)

## Scope (initial)
- Start with **structural brain MRI** (e.g., T1/T2/FLAIR) and a **2D slice** pipeline for rapid iteration
- Support **unpaired multi-site** training (paired/traveling-subject data is optional when available)
- Establish baselines, then move toward a **domain-general, diffusion-based** harmonization model

## Planned Method Families
HarmonIt will compare and/or build on:
- **Statistical harmonization** (feature-level baselines such as ComBat-family methods)
- **GAN-based** image-to-image harmonization (e.g., CycleGAN-style)
- **Disentanglement / anatomy–contrast separation** approaches
- **Diffusion-based** multi-domain harmonization (current SOTA direction)

## Evaluation (what “good” means)
We will evaluate harmonization using a mix of:
- **Domain confusion:** site/scanner classifier accuracy should drop after harmonization
- **Anatomy preservation:** segmentation/registration consistency (or boundary/structure metrics)
- **Downstream performance:** classification/segmentation should remain stable or improve
- **Image similarity metrics:** (SSIM/PSNR) when paired or synthetic ground truth exists
- **Qualitative checks:** side-by-side visual grids + failure case analysis

## Milestones (high-level)
1) **Baseline setup:** data pipeline + no-harmonization reference
2) **Classic baseline:** CycleGAN / U-Net restoration baseline (depending on pairing)
3) **Modern baseline:** diffusion-based harmonization
4) **Anatomy-preserving constraints:** segmentation/registration consistency
5) **Paper-ready results:** ablations + evaluation protocol + figures/tables

## Reproducibility
- Fixed seeds, saved configs for every run
- Versioned splits
- Logged metrics + qualitative samples
- Environment captured via `requirements.txt` or `environment.yml`

## Status
Early stage — setting up dataset, baselines, and evaluation protocol.
