## UCLA missing T1 recovery (manual)

- Initial automated download produces 1102/1112 subjects with `mprage.nii.gz` (using scripts/bulk_download_abide_custom.py)
- The remaining 10 missing subjects were all UCLA:
  UCLA_51232, UCLA_51233, UCLA_51242, UCLA_51243, UCLA_51244,
  UCLA_51245, UCLA_51246, UCLA_51247, UCLA_51270, UCLA_51310.

### Resolution
- These UCLA subjects store the T1 scan under `scans/hires/` (instead of `scans/anat/`).
- The NIfTI filename is `hires.nii.gz` (instead of `mprage.nii.gz`).
- We manually downloaded these 10 subjects via the XNAT web UI and verified the following structure:

  data/ABIDE/<SUBJECT_ID>/scans/hires/resources/NIfTI/files/hires.nii.gz