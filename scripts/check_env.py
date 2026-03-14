import os
import platform

def main():
    print("=== HarmonIt Environment Check ===")
    print("Platform:", platform.platform())
    print("Python:", platform.python_version())
    print("Conda env:", os.environ.get("CONDA_DEFAULT_ENV", "(none)"))

    # Torch / torchvision
    try:
        import torch
        print("torch:", torch.__version__)
        print("torch CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA version (torch):", torch.version.cuda)
            print("GPU count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"GPU[{i}]:", torch.cuda.get_device_name(i))
    except Exception as e:
        print("torch: NOT OK ->", repr(e))

    try:
        import torchvision
        print("torchvision:", torchvision.__version__)
        # quick import test that fails when torchvision/torch mismatch
        from torchvision.models import resnet18
        _ = resnet18(weights=None)
        print("torchvision.models.resnet18: OK")
    except Exception as e:
        print("torchvision: NOT OK ->", repr(e))

    # MLflow
    try:
        import mlflow
        print("mlflow:", mlflow.__version__)
        print("MLFLOW_TRACKING_URI:", os.environ.get("MLFLOW_TRACKING_URI", "(default)"))
    except Exception as e:
        print("mlflow: NOT OK ->", repr(e))

    print("=== Done ===")

if __name__ == "__main__":
    main()