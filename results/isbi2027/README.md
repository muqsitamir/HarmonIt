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

## Run names

- `frozen_probe_v1_9methods_20260914_153813` - canonical frozen-probe run, ten outputs
  (the name predates the added `diffusion_20k_redraw` output).
- `retrained_raw_seed{42,1,2,3,4}_model_{best,last}_*` - site probes retrained on raw volumes.
- `retrained_shuffle_seed42_*` - subject-level shuffled-label control.
- `sliceprobe_<source>_seed{1,2,3}_model_{best,last}_9methods` - fixed-slice probes trained on
  `raw`, `raw_shuffle`, `histogram_matching`, `cyclegan_tuned`, `diffusion_20k` or `silhouette`.
- Superseded, kept for provenance: `frozen_probe_v1_8methods_20260913_222209` (without HCLD),
  `frozen_probe_v1_9methods_20260914_101417`, `retrained_*_seed42_9methods_20260914_140714`
  (nine outputs, before the diffusion redraw). The collector prefers the run with more outputs.

## Regenerating the paper assets

```bash
PY=python  # environment with pandas, matplotlib, scipy
A=results/isbi2027/analysis
$PY scripts/isbi2027_collect.py --runs results/isbi2027/runs --out $A/all_runs_long.csv
$PY scripts/isbi2027_probe_agreement.py --csv $A/all_runs_long.csv --out $A/probe_agreement
$PY scripts/isbi2027_tables.py --csv $A/all_runs_long.csv \
  --alignment $A/target_alignment_frozen10/target_alignment_summary.json \
  --histogram-probe $A/histogram_probe.json --out paper/isbi2027/tables/table_main.tex
$PY scripts/isbi2027_figures.py --csv $A/all_runs_long.csv --histogram-probe $A/histogram_probe.json \
  --out-dir paper/isbi2027/figures
bash paper/isbi2027/build.sh
```

`scripts/isbi2027_qualitative.py`, `target_alignment_isbi2027.py` and
`isbi2027_histogram_probe.py` need the image NPZs on vpulab.
