# ISBI 2027 paper: handoff

Updated 2026-09-15. Read this first when resuming. Scientific protocol and every
amendment: [ISBI2027_PROTOCOL.md](ISBI2027_PROTOCOL.md). Result files:
[results/isbi2027/README.md](../results/isbi2027/README.md).

## Status

- Venue: ISBI 2027 four-page paper. Deadline 26 October 2026 (11:59 pm EDT); notification
  12 January 2027. Four pages of technical content including figures; an optional paid fifth
  page may hold only references, ethics and acknowledgments. Single-blind review. An AI-use
  disclosure is required in the acknowledgments. Template: `paper/isbi2027/spconf.sty`,
  `IEEEbib.bst` and `strings.bib` are identical to the official ISBI template kit (checked
  2026-09-15 against the zip the author supplied).
- Manuscript: `paper/isbi2027/main.tex`, compiled `main.pdf` is **4 pages including
  references** (a few lines spare). All experiments, including amendments 6-7, are complete and
  in the manuscript; every result number is from `results/isbi2027` (claim check done
  2026-09-15). Qualitative figure is column width; references use "et al." to save space.
- Acknowledgments now carry the ABIDE-required funding acknowledgment (NIMH K23MH087770,
  R03MH096321, Stavros Niarchos and Leon Levy Foundations) and the compute resources. If a
  supervisor names a grant, add it there; otherwise the paper carries no grant line.
- Registration for ISBI 2027 (Lausanne, EPFL, 25-28 May 2027) will be covered by P. Coup\'e if the
  paper is accepted; travel is self-funded. ISBI's no-show policy (per ISBI 2026) removes a paper
  from IEEE Xplore unless the presenting author is registered and presents in person.
- Remaining placeholders: funding (red `\pending{Funding.}`; authors to settle with the
  supervisor). The AI-use statement is drafted and may be edited by the authors.
- `paper/isbi2027/main_june2026.tex` is the superseded June benchmark draft; several of its
  claims are wrong (pooled HCLD PSNR, raw-to-output distances read as NYU alignment,
  class-ID shuffle control). Do not reuse its numbers.

## Decisions by the authors

- Framing: evaluation audit ("site-probe accuracy is not a harmonization score"); the
  methods are test cases, not a leaderboard. Not a corrected benchmark.
- Authors: June list and order (Muhammad Muqsit Islam and Gloria García Cuenco equal
  contribution; José M. Martínez Sánchez; Pierrick Coupé; UAM and University of Bordeaux).
- Code, protocol and per-subject results are released on GitHub (public, `main`); the paper
  footnote links `https://github.com/muqsitamir/HarmonIt`.
- Repository hygiene (decided 2026-09-15): no assistant instruction files, AI co-author
  trailers or tool-named branches on GitHub. The paper keeps its named AI-use disclosure.
- Compute: prefer vpulab (RTX A5000, near dedicated). Use the shared cl cluster only when
  needed (it was used for the HCLD re-export).
- No paid fifth page (decided 2026-09-15): the whole paper, including references, ethics and
  acknowledgments, must fit in four pages.
- Reviewer-risk experiments approved 2026-09-15 ("whatever improves acceptability"):
  amendment 6 (converged probe recipe) and amendment 7 (brain-only probes with HD-BET masks).

## Findings (source subjects, n = 90)

1. Verdicts depend on probe training: five benchmark-recipe probes (raw BA 0.81-0.98, unstable
   validation BA) gave CycleGAN 0.13-0.59, diffusion 0.24-0.63, histogram matching 0.27-0.59,
   aggressive StarGAN 0.18-0.54 (tau 0.51-0.87). Five converged probes (amendment 6; raw BA
   1.00) agreed (tau 0.82-0.96, ranges <= 0.16) but rated eight of ten outputs above the frozen
   probe's interval (aggressive StarGAN 0.65-0.81 vs 0.21; CycleGAN 0.44-0.54 vs 0.31) and
   reordered methods (tau with frozen 0.64-0.73).
2. Destruction scores best: HCLD BA 0.07 (chance) under every probe, PSNR 13.9 dB, XCorr 0.71,
   moves away from NYU (dW +0.108).
3. Change is not alignment: diffusion changes intensities least (W 0.008) but dW_NYU
   -0.0013 [-0.0023, -0.0004] (1% of raw distance); KL to NYU 0.30 -> 0.83. A second diffusion
   sampling draw differs from the first more than from the input.
4. Hidden, not removed: head slice probes raw-trained 0.33-0.45 on outputs, output-trained
   0.87-0.91 (silhouettes alone 0.73-0.86). Brain-only (amendment 7): raw-trained 0.39-0.62,
   output-trained 0.74-0.91 (disjoint for all seeds with best checkpoints, 5/9 with final);
   brain masks alone 0.52-0.54. Intensity histograms: head-level histogram matching 0.16 (looks
   removed), CycleGAN 0.74, diffusion 0.70 (raw 0.74); brain-level all three retained it
   (0.53, 0.55, 0.38 vs raw 0.56).

## Where things live

| What | Location |
| --- | --- |
| Code, paper, results | branch `main` of `github.com/muqsitamir/HarmonIt` (ISBI work merged 2026-09-15; commit IDs cited in older run logs map to current ones in `ISBI2027_COMMIT_MAP.tsv`) |
| Experiment root (vpulab) | `/mnt/rhome/mmi/projects/isbi2027`: `code/` (rsync snapshot, `COMMIT` file), `runs/`, `exports/`, `slice_probes/`, `probe_work/`, `inputs/`, `analysis/` |
| Data, historical artifacts, frozen probe (vpulab) | `/mnt/rhome/mmi/projects/HarmonIt` (`data/`, `outputs/harmonized/`, `checkpoints/`) |
| Python env (vpulab) | `/home/mmi/envs/harmonit-isbi` (torch 2.5.1+cu121, numpy 1.26.4); installer `/mnt/rhome/mmi/envs/install_harmonit_isbi.sh` |
| HCLD and diffusion training (cl) | `/home/muqsitamir/repos/HarmonIt`; HCLD canonical re-export in `outputs/harmonized/adapted_hcld_isbi2027_canonical` |
| Normalized-volume cache (vpulab, local disk) | `/home/mmi/cache/isbi2027_volumes` (46 GB, train+val; `cache_report.json`) |
| HD-BET env and masks (vpulab) | env `/home/mmi/envs/hdbet` (hd-bet 2.0.1, weights in `~/hd-bet_params`); masks `/mnt/rhome/mmi/projects/isbi2027/brain_masks/hdbet` |
| Brain-only exports and probes (vpulab) | `isbi2027/exports/brain/` (masks, masked train/val, brain_shape), `isbi2027/brain_probes/`, log `isbi2027/brain_run.log` |
| Figure raster quality | `isbi2027_qualitative.py` saves at dpi=600 with `interpolation="nearest"`: vector backends rasterize embedded images at the figure dpi, so the default 100 dpi stored each 256x256 slice as ~42x42 px and printed blurred. Check with `page.get_images()` after any change. |
| Figure variants | The paper uses `figures/fig_qualitative_wide.pdf` (7.0 x 2.05 in, two rows: outputs and difference maps). `isbi2027_qualitative.py --rows 1` makes a single-row variant with larger brains but no difference maps; `fig_qualitative.pdf` is the column-width fallback |
| LaTeX (Mac) | TinyTeX in `~/Library/TinyTeX` (not on PATH); `bash paper/isbi2027/build.sh` |

vpulab notes: set `https_proxy=http://192.168.22.3:8080` for downloads (the system value uses
an `https://` scheme and fails). Deploy code with rsync using root-anchored excludes
(`--exclude '/data/'`, not `data/`). Do not wait on jobs with `pgrep -f <script>` inside
`ssh`, and never `pkill -f <script>` inside `ssh`: both match the ssh command line itself
(pkill kills the session). Launch detached jobs as `ssh -n vpulab '(setsid nohup ... &)'`.

## Pipeline

| Step | Script |
| --- | --- |
| Evaluate outputs with a probe | `scripts/eval_isbi2027.py` via `scripts/vpulab_isbi2027_eval.sh` (`SITE_PROBE`, `EXTRA_ARTIFACT`, `HCLD_ARTIFACT`, `REDRAW` env) |
| Retrain site probes | `scripts/train_site_probe.py` via `scripts/vpulab_isbi2027_probe.sh` and `vpulab_isbi2027_probe_seeds.sh` |
| Train/val/test re-exports | `scripts/vpulab_isbi2027_exports.sh`; gate `scripts/check_isbi2027_exports.py` |
| Slice probes (harmonized, silhouette) | `scripts/train_slice_probe.py` via `vpulab_isbi2027_slice_probes.sh`, `vpulab_isbi2027_silhouette.sh`, `make_silhouette_npz.py` |
| HCLD re-export (cl) | `slurm/isbi2027_hcld_reexport.sbatch` |
| Target alignment | `scripts/target_alignment_isbi2027.py` |
| Intensity-only probe | `scripts/isbi2027_histogram_probe.py` (`--brain-masks` for amendment 7) |
| Converged probes (amendment 6) | `cache_normalized_volumes.py`; `RECIPE=converged VOLUME_CACHE_DIR=... SEEDS="5 6 7 8 9" vpulab_isbi2027_probe_seeds.sh`; `isbi2027_converged.py` |
| Brain-only control (amendment 7) | `run_hdbet_masks.py` (hdbet env), then `vpulab_isbi2027_brain.sh` (`make_brain_npz.py`, slice probes, `eval_isbi2027.py --probe-input-mask`) |
| Cohort acquisition heterogeneity (amendment 8) | `isbi2027_acquisition_heterogeneity.py` (runs locally; needs the NIfTI files) |
| Collect, agreement, table, figures | `isbi2027_collect.py`, `isbi2027_probe_agreement.py`, `isbi2027_tables.py`, `isbi2027_figures.py`, `isbi2027_qualitative.py` |
| Metric functions and tests | `src/harmonit/metrics/subject_evaluation.py`, `tests/test_isbi_evaluation.py` |

## Known caveats (all stated in the protocol or paper)

- Site labels are coarser than the acquisition: 11 of 17 sites hold more than one voxel size or
  matrix, UM/UCLA/Leuven merge two released sub-samples each, and 10 UCLA training scans are
  1.5x1.5x4 mm. No per-subject scanner identifier exists in ABIDE I (amendment 8).
- Fixed-slice selection has exact ties (4/109 test subjects); slices are frozen in
  `configs/isbi2027/test_slice_indices.json`, and the evaluator rejects mismatches.
- Diffusion img2img start noise is not reproducible across GPUs; its outputs are a draw.
- NeuroCombat was fit on the test cohort (transductive); histogram matching uses a pooled
  17-site training reference, not NYU.
- The dataset's NumPy RandomState is copied unchanged into DataLoader workers (inherited
  from the production probe recipe; kept for fidelity).
- Test subjects were inspected in the earlier benchmark: retrospective reanalysis.
  Amendments 4-5 (silhouette and intensity-only controls) were added post hoc.

## Next steps

1. Author read the revised `main.pdf` on 2026-09-15 and approved it; supervisor review and
   funding text (the author will add it) remain.
2. Before submission: fill the submission form. The repository is public (checked 2026-09-15),
   and the ISBI work is merged into `main`, so the paper's repository link is correct. The
   template kit matches the official one; no paid fifth page.
3. Optional if space allows: brain-only probes with a converged recipe were not run (slice probes
   use the benchmark recipe, noted as a limitation).
