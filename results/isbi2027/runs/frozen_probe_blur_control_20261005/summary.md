# ISBI evaluation: probe /mnt/rhome/mmi/projects/HarmonIt/checkpoints/site_probe_v0.3_aug_ramp15/model_best.pt

Reference: fresh. Whole-image metrics; NYU excluded from this table.
95% paired site-stratified subject bootstrap intervals, conditional on the saved checkpoints.
Wasserstein/KL below are mean per-subject raw-to-harmonized distances, not historical pooled distances or NYU alignment.

| Method | Source BA | BA drop | PSNR | Pixel cosine | XCorr | Subject W | Subject KL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_blur_s2 | 0.0750 [0.0625, 0.1000] | 0.8729 [0.8125, 0.9250] | 24.7423 [24.6121, 24.8740] | 0.9860 [0.9853, 0.9866] | 0.9740 [0.9729, 0.9750] | 0.0175 [0.0172, 0.0179] | 0.1011 [0.0911, 0.1110] |
| gaussian_blur_s4 | 0.0625 [0.0625, 0.0625] | 0.8854 [0.8333, 0.9375] | 20.7668 [20.6579, 20.8850] | 0.9647 [0.9631, 0.9662] | 0.9333 [0.9307, 0.9358] | 0.0324 [0.0316, 0.0332] | 0.3015 [0.2800, 0.3230] |
| gaussian_blur_s8 | 0.0682 [0.0625, 0.0795] | 0.8797 [0.8220, 0.9375] | 18.2817 [18.1650, 18.4039] | 0.9358 [0.9328, 0.9386] | 0.8750 [0.8695, 0.8800] | 0.0466 [0.0454, 0.0480] | 0.7938 [0.7531, 0.8356] |
