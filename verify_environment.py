"""Verify the local computer-vision environment and CUDA access."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache"
MPL_CACHE = CACHE_ROOT / "matplotlib"
YOLO_CACHE = CACHE_ROOT / "Ultralytics"

MPL_CACHE.mkdir(parents=True, exist_ok=True)
YOLO_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT))

import cv2  # noqa: E402
import mediapipe  # noqa: E402
import numpy  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
import ultralytics  # noqa: E402


def main() -> None:
    print(f"Python environment: {Path(os.sys.executable)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"MediaPipe: {mediapipe.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"NumPy: {numpy.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime: {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    device = torch.cuda.get_device_name(0)
    checksum = torch.ones((1024, 1024), device="cuda").sum().item()
    print(f"GPU: {device}")
    print(f"CUDA tensor checksum: {checksum}")


if __name__ == "__main__":
    main()
