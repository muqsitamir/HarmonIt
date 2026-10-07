# ISBI evaluation: probe /mnt/rhome/mmi/projects/isbi2027/haca3/probes/probe_work/runs/site_probe/isbi2027__raw_seed2/20261006_233620/model_best.pt

Reference: fresh. Whole-image metrics; NYU excluded from this table.
95% paired site-stratified subject bootstrap intervals, conditional on the saved checkpoints.
Wasserstein/KL below are mean per-subject raw-to-harmonized distances, not historical pooled distances or NYU alignment.

| Method | Source BA | BA drop | PSNR | Pixel cosine | XCorr | Subject W | Subject KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| haca3 | 0.3731 [0.3104, 0.4339] | 0.5457 [0.4790, 0.6223] | 18.9213 [18.5314, 19.3454] | 0.9757 [0.9725, 0.9784] | 0.9537 [0.9476, 0.9592] | 0.0621 [0.0573, 0.0668] | 0.3301 [0.3050, 0.3562] |
| haca3_preproc | 0.6417 [0.6021, 0.6750] | 0.2771 [0.2208, 0.3354] | 23.7519 [23.3754, 24.1156] | 0.9920 [0.9914, 0.9926] | 0.9855 [0.9841, 0.9866] | 0.0428 [0.0405, 0.0452] | 0.1274 [0.1145, 0.1416] |
