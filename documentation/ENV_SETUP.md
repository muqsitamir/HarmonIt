# Environment Setup (HarmonIt)

Goal: keep `requirements.txt` **torch-free** and install PyTorch **per-machine** (Mac vs GPU lab/cluster) to avoid binary / CUDA mismatches.

---

## Rule of thumb

- `requirements.txt` = Python packages **except** `torch` / `torchvision`
- PyTorch is installed **separately** using conda (preferred), matched to the machine.

---

## Mac (Apple Silicon, CPU)

```bash
conda create -n harmonit311 python=3.11 -y
conda activate harmonit311

# PyTorch + torchvision (CPU build)
conda install -c pytorch -c conda-forge pytorch torchvision -y

# Project deps (no torch here)
pip install -r requirements.txt
pip install -e .

# Sanity check
python scripts/check_env.py
```

## NVIDIA GPU

```bash
conda create -n harmonit_gpu python=3.11 -y
conda activate harmonit_gpu

# Example for CUDA 12.1 runtime (change if needed)
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1 -y

pip install -r requirements.txt
pip install -e .

python scripts/check_env.py
```