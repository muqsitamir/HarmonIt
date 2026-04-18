import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_confusion_matrix_png(cm: np.ndarray, out_path: Path, class_names: list[str]):
    n = len(class_names)
    fig = plt.figure(figsize=(max(8, 0.55 * n), max(7, 0.50 * n)))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Site Probe Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_positions = np.arange(n)
    plt.xticks(tick_positions, class_names, rotation=90)
    plt.yticks(tick_positions, class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)