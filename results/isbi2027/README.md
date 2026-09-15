# ISBI 2027 audit results

Small, versioned outputs behind every number in `paper/isbi2027/main.tex`. Large
artifacts (image NPZs, checkpoints) stay on vpulab under
`/mnt/rhome/mmi/projects/isbi2027`; their SHA-256 hashes are in
`analysis/artifact_sha256.txt`. The protocol and its amendments are in
`documentation/ISBI2027_PROTOCOL.md`; the project state is in
`documentation/ISBI2027_HANDOFF.md`.

## Layout

| Path | Contents |
| --- | --- |
| `runs/<run>/` | One `scripts/eval_isbi2027.py` run: `protocol.json` (args, versions, input hashes), `COMPLETE.json`, per-method `*_summary.json` (all/source/NYU groups with bootstrap intervals) and `*_subjects.csv` (per-subject metrics and probe predictions), `metrics_long.csv`, `paired_source_differences.json`, `summary.md`. Bootstrap index arrays and the raw reference NPZ are not versioned. |
| `site_probes/isbi2027__<raw\|shuffle>_seed<S>/<timestamp>/` | Retrained site-probe configs, preprocessing, run metadata, shuffled subject labels. |
| `slice_probes/<source>_seed<S>/` | Slice-probe configs (input NPZ hashes), per-epoch history, shuffled labels. |
| `exports/` | Train/val/test re-export configs and manifests, and `integrity_report.json` (gate for the harmonized-probe experiment). |
| `analysis/all_runs_long.csv` | All runs collected by `scripts/isbi2027_collect.py` (one run per probe identity). |
| `analysis/target_alignment_frozen10/` | NYU target alignment for the ten outputs (`scripts/target_alignment_isbi2027.py`). |
| `analysis/histogram_probe.json` | Intensity-only probe results (`scripts/isbi2027_histogram_probe.py`). |
| `analysis/probe_agreement/` | Seed spread and Kendall tau (`scripts/isbi2027_probe_agreement.py`). |
| `analysis/converged_probes.json` | Amendment 6 outcomes: spread, tau, convergence, decision rule (`scripts/isbi2027_converged.py`). |
| `analysis/histogram_probe_brain.json` | Amendment 7 intensity-only probe with brain-mask foreground. |
| `brain_probes/<source>_seed<S>/` | Amendment 7 brain-only slice-probe configs and histories. |
| `exports/brain/brain_mask_report.json` | Amendment 7 gate: raw-slice reproduction and brain-to-head area ratios. |
| `logs/` | Converged-probe training logs (validation BA per epoch) and the volume-cache verification report. |

## Run names

- `frozen_probe_v1_9methods_20260914_153813` - canonical frozen-probe run, ten outputs
  (the name predates the added `diffusion_20k_redraw` output).
- `retrained_raw_seed{42,1,2,3,4}_model_{best,last}_*` - site probes retrained on raw volumes.
- `retrained_shuffle_seed42_*` - subject-level shuffled-label control.
- `sliceprobe_<source>_seed{1,2,3}_model_{best,last}_9methods` - fixed-slice probes trained on
  `raw`, `raw_shuffle`, `histogram_matching`, `cyclegan_tuned`, `diffusion_20k` or `silhouette`.
- `converged_{raw,shuffle}_seed<S>_model_{best,last}_9methods_*` - converged-recipe site probes
  (amendment 6; raw seeds 5-9, shuffle seed 5).
- `brainprobe_<source>_seed<S>_model_{best,last}` - brain-only slice probes evaluated with
  brain-masked probe inputs (amendment 7); sources `raw`, `raw_shuffle`, the three methods and
  `brain_shape`.
- Superseded, kept for provenance: `frozen_probe_v1_8methods_20260913_222209` (without HCLD),
  `frozen_probe_v1_9methods_20260914_101417`, `retrained_*_seed42_9methods_20260914_140714`
  (nine outputs, before the diffusion redraw). The collector prefers the run with more outputs.

## Regenerating the paper assets

```bash
PY=python  # environment with pandas, matplotlib, scipy
A=results/isbi2027/analysis
$PY scripts/isbi2027_collect.py --runs results/isbi2027/runs --out $A/all_runs_long.csv
$PY scripts/isbi2027_probe_agreement.py --csv $A/all_runs_long.csv --out $A/probe_agreement
$PY scripts/isbi2027_converged.py --csv $A/all_runs_long.csv --logs results/isbi2027/logs \
  --out $A/converged_probes.json
$PY scripts/isbi2027_tables.py --csv $A/all_runs_long.csv \
  --alignment $A/target_alignment_frozen10/target_alignment_summary.json \
  --out paper/isbi2027/tables/table_main.tex
$PY scripts/isbi2027_figures.py --csv $A/all_runs_long.csv --histogram-probe $A/histogram_probe.json \
  --histogram-probe-brain $A/histogram_probe_brain.json --out-dir paper/isbi2027/figures
bash paper/isbi2027/build.sh
```

`scripts/isbi2027_qualitative.py`, `target_alignment_isbi2027.py` and
`isbi2027_histogram_probe.py` need the image NPZs on vpulab. The paper's qualitative figure is
the column-width variant: `isbi2027_qualitative.py --eval-run runs/frozen_probe_v1_9methods_20260914_153813
--width 3.39 --height 1.12 --show histogram_matching cyclegan_tuned diffusion_20k diffusion_20k_redraw adapted_hcld`.
