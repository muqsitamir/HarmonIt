# Installation

This document provides instructions to set up **without Docker**.

The recommended workflow is:
- use a **local conda environment** on each machine
- install the repo in **editable mode** with `pip install -e .`
- keep hardware-specific packages aligned with the machine you are using

## 1. Clone the repository

```bash
git clone https://github.com/muqsitamir/HarmonIt.git
cd HarmonIt
```

## 2. Create and activate a conda environment

Use Python 3.11.

```bash
conda create -n harmonit311 python=3.11 -y
conda activate harmonit311
```

## 3. Install PyTorch

Install PyTorch **before** installing the rest of the requirements.

### CPU-only machines
For Mac laptops, Windows laptops, or any machine without CUDA:

```bash
pip install torch==2.5.1 torchvision==0.20.1
```

### NVIDIA GPU machines
For Ubuntu machines with an NVIDIA GPU, install a CUDA-compatible PyTorch build.

Example using conda:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Adjust the CUDA variant if the target machine requires it.

## 4. Install project dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install makes the package importable without needing to manually set `PYTHONPATH`.

## 5. Generate the ABIDE manifest

Run this once after the dataset is available locally:

```bash
python scripts/make_abide_manifest.py
```

This writes repo-relative paths into `data/abide_manifest.csv`, which makes the manifest portable across machines.

## 6. Sanity checks

Check the environment:

```bash
python scripts/check_env.py
```

Check the dataloader and QC batch generation:

```bash
python scripts/qc_dataloader.py
```

## 7. Run training

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python scripts/train_site_probe.py
```

## 8. View MLflow UI

From the repo root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

- `requirements.txt` should stay **torch-free**. Install torch separately depending on the machine.
- Mac/CPU and Ubuntu/GPU environments are expected to differ in **hardware-specific dependencies**.
- After pulling code changes that affect package structure, rerun:

```bash
pip install -e .
```

- If you move to a different GPU machine later, you may need a different PyTorch/CUDA combination, but the rest of the setup should remain the same.
