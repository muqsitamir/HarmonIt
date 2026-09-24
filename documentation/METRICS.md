# Evaluation metrics

Updated 2026-09-25. Definitions used by the corrected subject-level evaluator
(`scripts/eval_isbi2027.py`) in the ISBI 2027 audit; its frozen settings and dated
amendments are in the [protocol](ISBI2027_PROTOCOL.md). Historical first-phase numbers
were computed differently and are not comparable without recomputation.

## What is being compared?

| Quantity | Inputs | Interpretation |
| --- | --- | --- |
| Site balanced accuracy | Site-probe predictions (frozen, retrained, converged or trained on outputs) versus acquisition-site labels | Recoverability under that classifier; not proof of scanner invariance |
| Balanced-accuracy drop | Raw BA minus harmonized BA on the same cohort | Change in frozen-probe performance |
| PSNR | Same subject's raw and harmonized slice | Pixel preservation on an explicitly specified intensity scale |
| Pixel cosine similarity | Flattened raw and harmonized pixels | Vector similarity, not VGG or semantic features |
| Cross-correlation | Flattened raw and harmonized pixels | Linear intensity correlation, not an anatomical accuracy score |
| Directional KL | Raw versus harmonized intensity histograms | Intensity-distribution change, not target alignment |
| Wasserstein distance | Flattened raw versus harmonized intensity samples | Magnitude of distribution change in intensity units |
| Target alignment | Raw and harmonized foreground intensities versus a fixed NYU training reference | Movement toward the target site, not anatomical correctness |

The fixed-slice evaluator computes these image measures over the whole image,
including background. Background suppression in preprocessing does not make
the evaluation brain-masked. A conservative head mask is not tissue segmentation.

## PSNR

For normalized inputs, the corrected definition is
`PSNR = 10 * log10(1 / MSE)`, with `data_range=1` fixed across methods.
The historical implementation used the output's maximum-minus-minimum range,
which can change the score when outputs change scale. Historical numbers must
be recomputed before comparison with the corrected definition.

Exact identity has zero MSE and infinite PSNR. Report identity counts and
translated-subject summaries separately; never silently discard identities
or assign an arbitrary finite cap. Report MSE/MAE as complementary measures.
Out-of-range model values should be recorded, not silently clipped for scoring.

## Pixel cosine and cross-correlation

Pixel cosine is `dot(x,y)/(norm(x)*norm(y))`. Despite the legacy function name
`feature_similarity`, the evaluator supplies pixels rather than CNN embeddings.
It does not establish semantic or anatomical preservation.

Cross-correlation is the cosine of mean-centred pixel vectors. High correlation
can survive substantial intensity rescaling. Constant images require an explicit
undefined/degenerate-value policy; a finite score must not conceal a failed output.

## Distribution distances

Historical output keys include `kl_raw_to_harmonized`,
`kl_harmonized_to_raw`, and `wasserstein`. Each compares the raw and transformed
cohort globally, with analogous per-site comparisons. Neither distance compares
harmonized images with an NYU reference distribution; target alignment (below) does.

KL is directional and uses natural logarithms. Historical code computed separate
histogram edges for each array, so coordinates could differ. The ISBI 2027 evaluator
uses 50 fixed bins on [0, 1] plus underflow and overflow bins and `1e-8` probability
smoothing, identical for every method and subject, and averages per-subject distances.

Wasserstein uses `scipy.stats.wasserstein_distance` directly on intensity samples;
it does not require histogram bins. Its value is the minimum average intensity
movement needed to transform one empirical distribution into the other.
It does not describe spatial movement or registration of anatomy.

Low raw-to-harmonized distance means little distributional change. That may
reflect preservation or insufficient harmonization. Do not label it evidence
of NYU matching or an unconditional measure of success.

## Target alignment

The reference is the foreground pixels of the 147 NYU training subjects' fixed raw
slices, each subject weighted equally. Foreground is defined on the raw slice
(intensity > 0.02), never on a method's output, and applied to both raw and
harmonized images. For each source subject, Wasserstein-1 and KL (harmonized ||
reference) are computed for the harmonized and for the raw image. `dW_NYU` is the
harmonized distance minus the raw distance, so negative values mean closer to NYU.
Intensity alignment is not evidence of anatomical correctness.

## Target identities and uncertainty

Some target-conditioned methods leave NYU subjects unchanged, while the existing
NeuroCombat cohort correction does not. Report all-subject, non-NYU translation,
and NYU identity behaviour separately. For source-only site BA, state the class
set used in macro-averaging. The audit macro-averages the 16 source classes while
probes predict all 17, so uniform chance is 1/17 (about 0.06); shuffled-label probes
give the empirical null. Report per-site sample counts (2-15 source test subjects
per site in the audit).

Confidence intervals should resample subjects, preserving paired method
comparisons and accounting for site composition. The audit uses 2,000 bootstrap
replicates resampling subjects within site, paired across methods and probes;
its intervals are conditional on the fixed sites and trained checkpoints. Pixels
and multiple slices from the same subject are not independent test subjects.

## Planned additions

- Foreground preservation under a shared reference mask, clearly distinguished
  from true brain-tissue masking.
- Full-volume consistency, regional volumes, and segmentation agreement.
- Selected downstream and unseen-site evaluation.

Segmentation agreement is not ground-truth accuracy, and distribution matching
does not prove biological preservation. Consult the
[project overview](../README.md#from-benchmark-to-25d-diffusion) for the current
research direction.
