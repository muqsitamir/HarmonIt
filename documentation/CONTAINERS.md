# HarmonIt container workflow

## Files
- `Dockerfile.cpu`: CPU image for Mac and Windows laptops, and any CPU-only debugging.
- `Dockerfile.gpu`: GPU image for Ubuntu GPU machines.
- `docker-compose.yml`: convenience entry points.
- `Makefile`: one-line build/run commands.
- `pyproject.toml`: package metadata so the repo can be installed with `pip install -e .`.
- `scripts/run_cpu.sh` and `scripts/run_gpu.sh`: lightweight wrappers.

## Recommended usage by machine
- **Mac laptop**: use `Dockerfile.cpu`
- **Windows laptop**: use `Dockerfile.cpu` via Docker Desktop + WSL2
- **Lab PC (Ubuntu + GPU)**: use `Dockerfile.gpu`

## First-time host setup
### CPU hosts
Install Docker only.

### GPU Ubuntu hosts
Install:
1. NVIDIA driver
2. Docker Engine
3. NVIDIA Container Toolkit

Then verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

## Build
```bash
make build-cpu
make build-gpu
```

Rebuild the image whenever one of these changes:
- `requirements.txt`
- `pyproject.toml`
- `Dockerfile.cpu`
- `Dockerfile.gpu`
- OS-level packages or image build steps

If only Python source files change, a rebuild is usually not needed because the repo is mounted into the container.

## Sanity checks
```bash
make check-env-cpu
make check-env-gpu
```

## QC and training
```bash
make qc-cpu
make train-probe-gpu
```

## Notes
- Raw data is **not** copied into the image. Keep it on disk and mount it.
- `requirements.txt` should stay **torch-free**. The CPU image installs CPU PyTorch; the GPU image installs its GPU stack separately.
- Keep MLflow outputs outside the image so runs persist.
- On Windows, keep the repo inside the WSL filesystem for better performance when possible.
