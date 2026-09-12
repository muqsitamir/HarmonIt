# Evaluation metrics

Updated 2026-09-12. Distinguish historical released code, local audit corrections,
and planned evaluation. This documentation update does not release the local
evaluator changes or certify the historical result table.

## What is being compared?

| Quantity | Inputs | Interpretation |
| --- | --- | --- |
| Site balanced accuracy | Frozen probe predictions versus original acquisition-site labels | Recoverability under that classifier; not proof of scanner invariance |
| Balanced-accuracy drop | Raw BA minus harmonized BA on the same cohort | Change in frozen-probe performance |
| PSNR | Same subject's raw and harmonized slice | Pixel preservation on an explicitly specified intensity scale |
| Pixel cosine similarity | Flattened raw and harmonized pixels | Vector similarity, not VGG or semantic features |
| Cross-correlation | Flattened raw and harmonized pixels | Linear intensity correlation, not an anatomical accuracy score |
| Directional KL | Raw versus harmonized intensity histograms | Intensity-distribution change, not target alignment |
| Wasserstein distance | Flattened raw versus harmonized intensity samples | Magnitude of distribution change in intensity units |

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
cohort globally, with analogous per-site comparisons. Neither distance currently
compares harmonized images with an NYU reference distribution.

KL is directional, uses natural logarithms, and defaults to 50 bins. Historical
code computed separate histogram edges for each array; those coordinates can
differ. The local correction uses common edges spanning the pair, normalizes
counts, and adds `1e-8` smoothing before SciPy entropy evaluation. This is a
pairwise correction, not yet a globally frozen histogram protocol across models.

Wasserstein uses `scipy.stats.wasserstein_distance` directly on intensity samples;
it does not require histogram bins. Its value is the minimum average intensity
movement needed to transform one empirical distribution into the other.
It does not describe spatial movement or registration of anatomy.

Low raw-to-harmonized distance means little distributional change. That may
reflect preservation or insufficient harmonization. Do not label it evidence
of NYU matching or an unconditional measure of success.

## Target identities and uncertainty

Some target-conditioned methods leave NYU subjects unchanged, while the existing
NeuroCombat cohort correction does not. Report all-subject, non-NYU translation,
and NYU identity behaviour separately. For source-only site BA, state the class
set used in macro-averaging. Report per-site sample counts.

Confidence intervals should resample subjects, preserving paired method
comparisons and accounting for site composition. Pixels and multiple slices
from the same subject are not independent test subjects.

## Planned additions

- Explicit harmonized-to-NYU and raw-to-NYU distances against a fixed training
  reference, with documented weighting, bins, smoothing, and out-of-range handling.
- Foreground preservation under a shared reference mask, clearly distinguished
  from true brain-tissue masking.
- Full-volume consistency, regional volumes, and segmentation agreement.
- Retrained site probes and selected downstream/unseen-site evaluation.

Segmentation agreement is not ground-truth accuracy, and distribution matching
does not prove biological preservation. Consult the [roadmap](TRDP2_PLAN.md)
and [submission audit](SUBMISSION_GAP_REPORT.md) for the current research scope.
