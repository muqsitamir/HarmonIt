from pathlib import Path
import hashlib
import subprocess

# Helper functions for fingerprinting and reproducibility
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_info() -> dict:
    """Best-effort git fingerprinting. Returns commit SHA and dirty flag."""
    info = {"git_commit": "unknown", "git_dirty": "unknown"}
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["git_commit"] = sha
        code = subprocess.call(["git", "diff", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        info["git_dirty"] = (code != 0)
    except Exception:
        pass
    return info
