# ISBI evaluation: probe /mnt/rhome/mmi/projects/isbi2027/haca3/probes/slice_probes/raw_seed2/model_best.pt

Reference: fresh. Whole-image metrics; NYU excluded from this table.
95% paired site-stratified subject bootstrap intervals, conditional on the saved checkpoints.
Wasserstein/KL below are mean per-subject raw-to-harmonized distances, not historical pooled distances or NYU alignment.

| Method | Source BA | BA drop | PSNR | Pixel cosine | XCorr | Subject W | Subject KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| haca3 | 0.3500 [0.2885, 0.4084] | 0.5952 [0.5150, 0.6729] | 18.9213 [18.5314, 19.3454] | 0.9757 [0.9725, 0.9784] | 0.9537 [0.9476, 0.9592] | 0.0621 [0.0573, 0.0668] | 0.3301 [0.3050, 0.3562] |
| haca3_preproc | 0.8704 [0.7927, 0.9396] | 0.0749 [0.0139, 0.1406] | 23.7519 [23.3754, 24.1156] | 0.9920 [0.9914, 0.9926] | 0.9855 [0.9841, 0.9866] | 0.0428 [0.0405, 0.0452] | 0.1274 [0.1145, 0.1416] |
