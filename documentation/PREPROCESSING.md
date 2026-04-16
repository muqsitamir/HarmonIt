# PREPROCESSING.md — HarmonIt (ABIDE T1 → 2D pipeline)

This document defines the preprocessing pipeline for HarmonIt. The goal is to (i) make ABIDE T1 scans batchable and comparable across sites, (ii) avoid leakage for the site-probe, and (iii) preserve anatomy/pathology signals for harmonization research.

---

## What the raw file contains (MRI intuition)

ABIDE T1 scans are stored as NIfTI volumes (`.nii.gz`), which contain:
- a **3D voxel intensity array** (a volumetric image), and
- metadata (voxel spacing, orientation, and an affine mapping from voxel indices to real-world coordinates).
NIfTI was designed as a standard neuroimaging interchange format to support consistent spatial interpretation. 

A **slice** is simply a 2D cross-section of the 3D volume (e.g., an axial slice is a top-down view). In a 2D training setup, we train on these 2D slices extracted consistently from the same 3D volume.

---

## Pipeline summary (v0.2)

**Input:** ABIDE T1 NIfTI volumes (1112 subjects)  
**Output (training-time):** 2D axial slices, normalized and standardized for batch training with geometry standardization (foreground bounding box crop + resize) and a robust slice selection policy focusing on mid-brain proxy slices.

We avoid pre-saving all slices (storage explosion). Instead:
1) preprocess **per-volume on-the-fly** (CPU-friendly),
2) extract standardized **2D axial slices** using a foreground bounding box crop with margin and resizing,
3) select slices based on percentile anchors and max foreground area scoring,

4) feed slices to the site-probe and harmonizer models.

**Key update (masking robustness):** We use a conservative **head/foreground mask** (largest connected component + hole filling + dilation) computed on the normalized slice *before any resizing*. We then (optionally) **suppress background** by setting pixels outside this mask to 0 *before interpolation*. This prevents the site-probe from exploiting framing/background shortcuts while preserving in-mask anatomy exactly.

---

## Steps, rationale, and citations

### Step 0 — Verify volume + metadata integrity (Sanity check)
**What:** ensure file loads, is 3D, has finite intensities, sensible voxel spacing.  
**Why it’s necessary:** corrupted images or wrong modality (e.g., fMRI) will silently poison training and evaluation.  
**How we enforce:** nibabel load + header zoom check + basic intensity stats.

---

### Step 1 — Canonical orientation (reorient to a standard axis order)
**What:** reorder/flip voxel axes to a canonical orientation (we use `nibabel.as_closest_canonical`).  

- MRI volumes can come stored with different axis conventions (L/R flips or axis permutations).
- A CNN trained on mixed orientations wastes capacity on coordinate inconsistencies and can learn incorrect spatial priors.
**Evidence:** nibabel explicitly defines `as_closest_canonical` as reordering image data to be closest to canonical axis order.  [oai_citation:0‡nipy.org](https://nipy.org/nibabel/reference/nibabel.funcs.html?utm_source=chatgpt.com)  
**If we skip:** you can get site-dependent performance artifacts (some sites stored differently) and unstable training.

**Ablation:** compare probe accuracy and harmonizer stability with/without canonical reorientation.

---

### Step 2 — Intensity normalization (mandatory in MRI)
**What:** per-volume robust normalization: clip intensities to [p1, p99] then z-score (within nonzero mask or within brain crop).  
**Implementation note:** after z-scoring, we explicitly restore true background voxels to exact 0 to prevent background from becoming a constant negative value (which would break threshold-based masking).
**Why it’s necessary:**
- MRI intensities are not standardized units; scale/contrast vary across scanners and protocols.
- Without normalization, models often overfit to site-specific intensity scaling rather than anatomy.
**Evidence:** Reinhold et al. show intensity normalization improves MR deep learning synthesis across methods and suggest it can be vital for successful DL-based MR synthesis.  [oai_citation:1‡PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/?utm_source=chatgpt.com)  
**If we skip:** the site-probe may become trivially strong due to global scaling, and harmonization may simply learn intensity rescaling rather than meaningful distribution alignment.

**Ablation:** compare site-probe balanced accuracy and harmonizer outputs under:
- z-score only
- percentile clip + z-score (default)
- alternative normalizations (optional later)

---

### Step 3 — Bias field correction (optional but medically standard; N4)
**What:** apply N4 bias field correction to reduce smooth intensity inhomogeneity across the brain.  
**Why it matters:**
- Bias fields are scanner/coil artifacts (slow shading), not anatomy.
- Removing them can improve cross-site consistency and reduce nuisance variation.
**Evidence:** Tustison et al. propose N4ITK as an improved N3 bias correction method for intensity non-uniformity.  [oai_citation:2‡PubMed](https://pubmed.ncbi.nlm.nih.gov/20378467/?utm_source=chatgpt.com)  
**Why optional:** N4 adds compute cost and can be deferred until after baseline results.  
**If we skip:** you may see residual shading differences that increase site separability.

**Ablation:** run probe/harmonizer with vs without N4 on a subset to quantify impact.

---

### Step 4 — Brain extraction / masking (optional early; useful to prevent probe “cheating”)
**What:** skull stripping or brain masking to remove non-brain tissue (eyes, skull, scalp, neck).  
**Why it matters:**
- Non-brain regions often encode site-specific acquisition differences (FOV, padding, coil effects).
- A site-probe can “cheat” by learning background/neck patterns instead of brain signal.
**Evidence:** Smith’s BET is a robust automated brain extraction method tested across scanners and sequences.  [oai_citation:3‡PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6871816/?utm_source=chatgpt.com)  
**Why optional in v0:** skull stripping can fail and requires QC; cropping can provide a simpler first baseline.  
**If we skip entirely:** site-probe accuracy may be inflated due to non-brain cues (still informative, but less clinically meaningful).

**Masking diagnostics (probe sanity checks):** We expose masking modes to test whether the probe is learning shortcuts:
- `MASK_MODE=brain_only`: keeps only pixels **inside** the head/foreground mask.
- `MASK_MODE=bg_only`: keeps only pixels **outside** the head/foreground mask.

**Important:** `BG_SUPPRESS=1` zeros pixels outside the head mask *before resize* (production setting). Therefore `MASK_MODE=bg_only` with `BG_SUPPRESS=1` should collapse to chance by design. For a *true diagnostic* of what the probe can learn from outside-head regions, run `MASK_MODE=bg_only` with `BG_SUPPRESS=0`.

These diagnostics are used to validate evaluation integrity; they are not the harmonization objective.

**Ablation:** compare probe accuracy:
- no mask (baseline)
- simple brain-centered crop (v0 default)
- skull strip / brain mask (later)

---

### Step 5 — Spatial standardization for 2D training (crop/pad, optional resampling)
**What (v0.2):**
- Extract axial slices from the 3D volume.
- Crop each slice using a head-mask bounding box (+ margin; default 20 pixels).
- Resize the cropped slice to 256×256 using bilinear interpolation.

This approach reduces probe cheating via borders or field-of-view differences and ensures the probe focuses on in-brain scanner signatures. Geometric standardization is a common practice in medical preprocessing to improve model robustness.

**Why it’s necessary:**
- ABIDE volumes have varying shapes and voxel spacings across sites; you cannot batch them otherwise.
- Standardized tensor sizes are required for stable training and comparable evaluation.
**Evidence (practice standard):** nnU-Net emphasizes that preprocessing choices (including resampling and normalization) strongly affect performance and includes automated preprocessing as a key reason for robustness.  [oai_citation:4‡PubMed](https://pubmed.ncbi.nlm.nih.gov/33288961/?utm_source=chatgpt.com)  
**If we skip:** training will either crash (shape mismatch) or require per-site hacks that break “unified” claims.

**Optional (later):** 3D resampling to isotropic spacing is common in medical pipelines, but we defer heavy resampling because we are using 2D and compute is constrained.


**Ablation:** crop size and slice range (e.g., mid-slices only vs full range) and measure probe accuracy stability.

### Step 5a — Head-mask crop + background suppression (probe robustness)
**What (v0.2+):**
- Compute a conservative head/foreground mask on the **full normalized slice** (before any resizing).
- Derive a tight crop region from the mask bounding box plus margin; crop the slice and mask together.
- If `BG_SUPPRESS=1`, set pixels outside the mask to 0 **before resize**.
- Resize the final slice to 256×256 for model input.

**Why it’s necessary:** resizing/interpolation can blur nonzero values into the background. If masking is applied after resize, background pixels can become nonzero and the mask can degenerate (covering everything), reintroducing shortcut learning. Applying masking/suppression pre-resize makes the definition of “background” stable and interpretable.

**Evidence (our QC):** `scripts/qc_head_mask.py` reports non-degenerate mask coverage across sites (e.g., mean mask area ~0.83 with nonzero variance) and retains nearly all in-mask energy (~0.999), indicating we are not cutting anatomy while preserving a meaningful outside-mask region.

---

### Step 5b — Slice selection policy (robust mid-brain sampling)
**What:**  
- Filter valid slices by selecting those in a middle depth band of the volume and with sufficient foreground fraction.  
- Define anchor slices at fixed depth percentiles (0.40, 0.50, 0.60) of the brain volume.  
- For each anchor, choose candidate slices near it and score them by foreground area.  
- During validation, select the max-foreground slice deterministically to ensure consistency.  
- During training, sample among the top-K scored slices to add variance while staying focused on mid-brain regions.

**Why it matters:**  
This policy avoids shortcuts based on slice-level or coverage differences that could bias the probe. Sampling robust mid-brain slices ensures the probe learns relevant scanner and anatomical features rather than trivial positional cues.

---

## Why 2D?
We adopt 2D because:
- It is feasible on limited GPU access and memory.
- Strong medical baselines exist that train both 2D and 3D variants depending on compute/task requirements.
**Evidence:** nnU-Net is a widely adopted framework that includes both 2D and 3D U-Net variants and shows that pipeline choices matter greatly.  [oai_citation:5‡PubMed](https://pubmed.ncbi.nlm.nih.gov/33288961/?utm_source=chatgpt.com)

---

## Evaluation integrity: subject-level splitting
**What:** split train/val/test by subject ID, never by slice.  
**Why:** slices from the same subject are highly correlated; mixing them causes leakage and inflated performance (especially for the probe).  
**Implementation:** group by `subject_id` from the manifest, then assign all slices from that subject to one split.

---

## Default v0.2 parameters
- Orientation: `as_closest_canonical`
- Intensity: clip [p1, p99] then z-score (within nonzero mask)
- Slice selection: percentile anchors (0.40, 0.50, 0.60) + max-foreground scoring; training samples top-K, validation picks best
- 2D size: head-mask bbox crop (margin=20) then resize to 256×256
- Head mask: threshold on |slice| (default thr=0.08), largest component + hole fill + dilation (default 3)
- Background suppression (probe): `BG_SUPPRESS=1` (default) zeros pixels outside head mask pre-resize
- Mask mode default: `MASK_MODE=none` (diagnostics: `bg_only`, `brain_only`; use `BG_SUPPRESS=0` for true outside-mask diagnostics)
- Bias correction: off in v0.2; optional N4 later

---

## Planned reporting

- preprocessing details + parameter values,
- QC examples across multiple sites,
- ablation evidence for key steps (intensity normalization, masking/crop, optional N4).

---

## Run provenance (reproducibility)

Each run logs comprehensive metadata to ensure reproducibility and traceability of ablation experiments, including:  
- `preprocessing.json` capturing preprocessing parameters and pipeline versions,  
- `run_metadata.json` containing hashes for the manifest and splits files, git commit hash and dirty state,  
- MLflow parameters and tags for experiment tracking.

This metadata ensures that differences in probe or harmonizer performance can be confidently attributed to preprocessing choices.