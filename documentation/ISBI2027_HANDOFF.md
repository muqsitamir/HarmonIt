# ISBI 2027 paper: handoff

Updated 2026-09-25. Read this first when resuming. Scientific protocol and every
amendment: [ISBI2027_PROTOCOL.md](ISBI2027_PROTOCOL.md). Result files:
[results/isbi2027/README.md](../results/isbi2027/README.md).

## Status

- Venue: ISBI 2027 four-page paper. Deadline 26 October 2026 (11:59 pm EDT); notification
  12 January 2027. Four pages of technical content including figures; an optional paid fifth
  page may hold only references, ethics and acknowledgments. Single-blind review. An AI-use
  disclosure is required in the acknowledgments. Template: `paper/isbi2027/spconf.sty`,
  `IEEEbib.bst` and `strings.bib` are identical to the official ISBI template kit (checked
  2026-09-15 against the zip the author supplied).
- Manuscript: `paper/isbi2027/main.tex`, compiled `main.pdf` is **4 pages including references**
  (last column ends at 715/724). Final pass 2026-09-24: every result number re-verified against
  `results/isbi2027`, probe counts corrected (44 trained: 1 frozen + 5 benchmark + 5 converged +
  4 shuffled + 24 slice + 6 restricted-input, plus 8 logistic fits), probes explained as a list in
  the protocol section, discussion opens with three takeaways, and slice probes are flagged as a
  conservative bound. No placeholders remain; ready for co-author review.
- Co-author review by P. Coupé applied 2026-09-25. Title: *Frozen Site-Probe Accuracy Is Not a
  Standalone Harmonization Score: An Audit of Evaluation Practice for Multi-Site MRI*. Chance is
  1/17 (probes predict 17 classes; BA averages the 16 source classes) with the shuffled-label
  null as the empirical reference; claims say "still decodable", not "concealed" or "transformed
  signature"; 2-15 test subjects per source site and bootstrap intervals conditional on sites and
  checkpoints are stated; "the adapted HCLD output" is a counterexample for the metric, not a
  verdict on published HCLD; novelty is set against feature-level ComBat studies; the README and
  METRICS.md describe the audit as complete; the AI disclosure names the tools, the affected
  sections and the level of use.
- Acknowledgments carry the ABIDE-required funding acknowledgment (NIMH K23MH087770,
  R03MH096321, Stavros Niarchos and Leon Levy Foundations) and P. Coupé's project funding (see
  Funding below). The VPU Lab GPU line was removed on 2026-10-02 with J. M. Martínez's agreement.
- 2026-10-02: the abstract now ends with the recommendation itself (converged probes including
  output-trained ones, a shuffled-label null, target alignment separate from change), and
  contribution (4) points to the four-item checklist in Sec. 5.
- 2026-10-02 (later): abstract rewritten in plain terms (defines the site probe, then four findings
  and the recommendation). Claims are limited to site-label decodability and intensity-distribution
  alignment, not scanner-effect removal: "understate site decodability", "removal of site-label
  information", "intensity alignment to the target". Page 4 now has no free line.
- 2026-10-02 (external review by OpenAI's Astra, borderline/weak reject): applied. The
  introduction no longer cites Glocker and Dinsdale for the frozen raw-trained probe; it cites
  Glocker for site predictability and DLEST/HCLD for reading lower site accuracy as harmonization
  (they retrain on harmonized images or fit on frozen features), and names the frozen probe as the
  audited benchmark's. Recipe effects are no longer attributed to convergence alone; histograms
  "discard spatial arrangement"; "smallest intensity-distribution change" replaces
  "least-changed"; image vs logistic probes, slice sampling and the ten outputs (second diffusion
  draw) are defined; paired output-minus-raw differences (amendment 9) replace the disjoint-
  interval statements. Abstract, contributions and discussion numbers were shortened to fit.
- 2026-10-02 literature check for the frozen raw-trained probe: PRISM (Galada et al., ISBI 2025
  oral, arXiv 2411.06513) trains ResNet50/EfficientNet-B2 on pre-harmonized data and evaluates them
  on harmonized data; it is now cited for that practice. Also found but not cited for space:
  Scholz et al. 2025 (arXiv 2509.06592, radiomics scanner classifier trained on unharmonized IXI
  scanners and applied to harmonized images); UMH (Wu et al., MICCAI 2025) and ImUnity (SVM on
  radiomics) read lower site accuracy as harmonization; the 2025 MRI harmonization survey (Yang et
  al., arXiv 2507.16962) calls site discriminability testing a widely adopted strategy.
- 2026-10-02: Astra's second review: weak accept, literature objection withdrawn. Applied its last
  points: the introduction says PRISM evaluates classifiers trained on original images on
  harmonized outputs (the audited protocol); "a drop interpreted as evidence of harmonization [9]";
  HCLD is "near the shuffled-label baseline" (retrained 0.06-0.11 exceeds the 0.03-0.08 null);
  paired differences are named as three-seed mean BA.
- 2026-10-02, final round (Astra's third review, weak accept): cites Souza et al. 2023 (site
  classifiers still predict site after histogram matching) beside Marzi and states what the audit
  adds; notes PRISM's anatomy and segmentation checks; states the output-trained probes' split;
  "varied less across seeds" replaces "agreed" and the seed-range versus bootstrap comparison is
  gone; adds amendment 10 (the frozen probe labelled 83 of 84 adapted-HCLD errors Yale, while
  65-80% of CycleGAN, aggressive StarGAN and diffusion errors named NYU); merges the discussion
  takeaways into the checklist with the caveat that high output-trained accuracy is not by itself
  failed harmonization. Text frozen after this round by the author's decision.
- 2026-10-02, Figure 1: the panel of per-output site BA (which repeated Table 1) was removed at the
  author's request. Fig. 1 now has (a) whole head and (b) brain only, each image-probe pair labelled
  with its paired difference (amendment 9). The figure is 1.25 in shorter, leaving about eight free
  lines on page 4 (room for an IPCVai acknowledgment).
- 2026-10-04, accuracy corrections from a third review (all verified against code and data):
  the "DLEST" outputs come from a DLEST-style content/AdaIN baseline without DLEST's energy-based
  sampler and are now named so (text and Table 1); the converged recipe's reseeded data-loader
  workers are disclosed and the benchmark spread is called largely a training-stability effect, as
  amendment 6 committed; HCLD's 0.06-0.11 is limited to benchmark-recipe and converged probes
  (diffusion-trained slice probes give 0.19-0.29); Fig. 2's caption says median PSNR over nine
  outputs (ORDER in isbi2027_qualitative.py omits dlest_1500); two brain-histogram intervals fixed
  (double rounding); Rahbar's 0.9 is chance-corrected BA. Template minimum of 9 pt: \ninept body,
  table and references at \small (9 pt), footnotes 9 pt, Fig. 1 regenerated at 9 pt, Fig. 2 rebuilt
  at 9 pt from its extracted panels. Every text span is 9 pt except sub/superscripts. Page 4's
  right column is about half empty. IPCVai acknowledgment: not added (no funding, per J. M. Martínez).
- 2026-10-04 (later): the author rejected the 9-pt layout's look. Reverted to 10-pt body, \scriptsize table,
  \footnotesize references and the previous Fig. 1 and Fig. 2; all accuracy corrections above are
  kept, with small wording trims to stay at four pages. The template's 9-pt guidance is therefore
  not met for Table 1, references and figure labels (author's decision).
- 2026-10-05 (P. Coupé asked what "adapted HCLD" is): Section 2 now states it is the authors' code
  retrained by us on ABIDE, without encoder/decoder non-local attention and at 192x192x64 to fit a
  40 GB GPU, and that its autoencoder alone loses anatomy, so the output does not represent HCLD
  (harmonit-dev results/ae_fidelity_val_20261001: our HCLD autoencoder keeps segmentation Dice 0.54
  with the input, against 0.89 for pretrained VAEs and 0.85 for the resize alone).
- Registration for ISBI 2027 (Lausanne, EPFL, 25-28 May 2027) will be covered by P. Coupé if the
  paper is accepted; travel is self-funded. ISBI's no-show policy (per ISBI 2026) removes a paper
  from IEEE Xplore unless the presenting author is registered and presents in person.
- Ethics (decided 2026-09-25): Section 6 follows accepted ISBI papers that used ABIDE, which
  use ISBI's example statement ("ethical approval was not required as confirmed by the license
  attached with the open access data"): Wang and Dvornek, ISBI 2021 (arXiv 2105.02874); Duan et
  al., ISBI 2025 oral (2502.15595); Weng et al., ISBI 2025 (2502.19386). It adds ABIDE's own
  account (Di Martino et al. 2014, Methods) that sites shared the anonymized data with local IRB
  approval or an explicit waiver. ABIDE's usage agreement itself does not mention ethics review;
  the stronger alternative, used by Jönemo et al. (2110.10489), cites the authors' own ethics
  committee confirming that approval was not required. A two-line sentence of that kind still
  fits in Section 6 (tested).
- Funding (2026-09-30): the acknowledgments carry P. Coupé's required text verbatim (project
  HoliBrain, ANR-23-CE45-0020-01; PEPR StratifyAging, PEPR Prodrom-ND and IHU VBHI,
  ANR-23-IAHU-0001) after the ABIDE and VPU Lab lines. The authors may edit the AI-use
  statement's wording.
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
2. The lowest site accuracy came from destruction: the adapted HCLD output scored BA 0.07
   (within the shuffled-label null) under every probe, PSNR 13.9 dB, XCorr 0.71,
   moves away from NYU (dW +0.108).
3. Change is not alignment: diffusion changes intensities least (W 0.008) but dW_NYU
   -0.0013 [-0.0023, -0.0004] (1% of raw distance); KL to NYU 0.30 -> 0.83. A second diffusion
   sampling draw differs from the first more than from the input.
4. Still decodable after harmonization: head slice probes raw-trained 0.33-0.45 on outputs, output-trained
   0.87-0.91 (silhouettes alone 0.73-0.86). Brain-only (amendment 7): raw-trained 0.39-0.62,
   output-trained 0.74-0.91 (disjoint for all seeds with best checkpoints, 5/9 with final);
   brain masks alone 0.52-0.54. Intensity histograms: head-level histogram matching 0.16 (looks
   removed), CycleGAN 0.74, diffusion 0.70 (raw 0.74); brain-level all three retained it
   (0.53, 0.55, 0.38 vs raw 0.56).

## Where things live

| What | Location |
| --- | --- |
| Code, paper, results | branch `main` of `github.com/muqsitamir/HarmonIt` (ISBI work merged 2026-09-15; commit IDs cited in older run logs map to current ones in `ISBI2027_COMMIT_MAP.tsv`) |
| Experiment root (vpulab) | `/mnt/rhome/mmi/projects/isbi2027` was mostly deleted on 2026-10-01: `code/`, `exports/`, `slice_probes/`, `probe_work/`, `inputs/`, `analysis/`, brain masks and most `runs/` are gone; part of `brain_probes/` and a few run folders remain. Every reported number is versioned in `results/isbi2027`. Not bit-reproducible: the second diffusion draw (`diffusion_20k_redraw`). Deterministic exports and probes can be regenerated with the pipeline below if reviewers ask for new probe experiments. |
| Test artifacts used by the paper | The eight test artifacts under `HarmonIt/outputs/harmonized/` (vpulab) and the canonical HCLD export on cl match the SHA-256 hashes in each run's `*_summary.json` (checked 2026-10-02). |
| Data, historical artifacts, frozen probe (vpulab) | `/mnt/rhome/mmi/projects/HarmonIt` (`data/`, `outputs/harmonized/`, `checkpoints/`) |
| Python env (vpulab) | `/home/mmi/envs/harmonit-isbi` (torch 2.5.1+cu121, numpy 1.26.4); installer `/mnt/rhome/mmi/envs/install_harmonit_isbi.sh` |
| HCLD and diffusion training (cl) | `/home/muqsitamir/repos/HarmonIt`; HCLD canonical re-export in `outputs/harmonized/adapted_hcld_isbi2027_canonical` |
| Normalized-volume cache (vpulab, local disk) | `/home/mmi/cache/isbi2027_volumes` (46 GB, train+val; `cache_report.json`) |
| HD-BET env and masks (vpulab) | env `/home/mmi/envs/hdbet` (hd-bet 2.0.1, weights in `~/hd-bet_params`); masks were in `isbi2027/brain_masks/hdbet` (deleted 2026-10-01; `run_hdbet_masks.py` regenerates them) |
| Brain-only exports and probes (vpulab) | `isbi2027/exports/brain/` and `isbi2027/brain_run.log` were deleted on 2026-10-01; part of `isbi2027/brain_probes/` remains |
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
| Per-subject review panels | `isbi2027_subject_panels.py`: five source subjects, four methods, per-subject metrics and difference maps (made for P. Coupé's review, 2026-10-02) |
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

1. Co-author review: P. Coupé's points applied 2026-09-25; the author circulates the PDF to the
   co-authors. Final wording pass the same day: redraw difference defined as the median
   per-subject mean absolute foreground difference, target alignment as the share of the
   initial foreground intensity-distribution distance, link borders hidden.
2. Optional: an ethics-committee confirmation from UAM (see Status) would strengthen Section 6.
3. J. M. Martínez's review (2026-09-30) is applied: single-blind review, so the repository link
   and VPU Lab credit stay; the abstract says "ten outputs from harmonization approaches"; the
   NeuroCombat citations read [1, 11]; P. Coupé's funding text is in the acknowledgments. To fit
   it, Table 1's caption and a few phrases were shortened and `\emergencystretch` fixes two
   overfull lines. Still open: one further acknowledgment line
   (J. M. Martínez to confirm); page 4's right column has about one line free.
4. Before submission: fill the submission form. The repository is public (checked 2026-09-15),
   and the ISBI work is merged into `main`, so the paper's repository link is correct. The
   template kit matches the official one; no paid fifth page.
5. Optional if space allows: brain-only probes with a converged recipe were not run (slice probes
   use the benchmark recipe, noted as a limitation).
