SHELL := /bin/bash
PROJECT := HarmonIt

build-cpu:
	docker build -f Dockerfile.cpu -t harmonit-cpu:latest .

build-gpu:
	docker build -f Dockerfile.gpu -t harmonit-gpu:latest .

shell-cpu:
	docker run --rm -it \
		-v $(PWD):/workspace/$(PROJECT) \
		-w /workspace/$(PROJECT) \
		-e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
		harmonit-cpu:latest bash

shell-gpu:
	docker run --rm -it --gpus all \
		-v $(PWD):/workspace/$(PROJECT) \
		-w /workspace/$(PROJECT) \
		-e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
		harmonit-gpu:latest bash

check-env-cpu:
	docker run --rm -it \
		-v $(PWD):/workspace/$(PROJECT) \
		-w /workspace/$(PROJECT) \
		harmonit-cpu:latest python scripts/check_env.py

check-env-gpu:
	docker run --rm -it --gpus all \
		-v $(PWD):/workspace/$(PROJECT) \
		-w /workspace/$(PROJECT) \
		-e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
		harmonit-gpu:latest python scripts/check_env.py

qc-cpu:
	docker run --rm -it \
		-v $(PWD):/workspace/$(PROJECT) \
		-w /workspace/$(PROJECT) \
		-e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
		harmonit-cpu:latest python scripts/qc_dataloader.py

train-probe-gpu:
	docker run --rm -it --gpus all \
		-v $(PWD):/workspace/$(PROJECT) \
		-w /workspace/$(PROJECT) \
		-e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
		harmonit-gpu:latest python scripts/train_site_probe.py
