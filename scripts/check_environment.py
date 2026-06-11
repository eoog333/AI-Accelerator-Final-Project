#!/usr/bin/env python3
"""Print local runtime and project asset availability."""

from __future__ import annotations

import importlib.util
import importlib
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    "requirements.txt",
    "src/mnist/train_mnist_13568.py",
    "src/mnist/eval_pgm.py",
    "src/yolo/video_to_pgm.py",
    "data/custom_digits",
    "data/custom_digits_pgm",
    "data/pgm",
    "data/videos/test.mp4",
    "models/mnist_13568_lab.pt",
    "models/mnist_13568_local.pt",
    "models/mnist_13568_colab.pt",
    "models/yolo_digits.pt",
]


def package_status(name: str) -> str:
    if not importlib.util.find_spec(name):
        return "missing"
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"broken ({type(exc).__name__}: {exc})"
    if name == "torch" and not hasattr(module, "__version__"):
        return "broken (missing __version__)"
    return "ok"


def path_status(path: str) -> str:
    target = ROOT / path
    if not target.exists():
        return "missing"
    if target.is_file():
        return f"ok file {target.stat().st_size} bytes"
    count = sum(1 for _ in target.rglob("*") if _.is_file())
    return f"ok dir {count} files"


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"project_root={ROOT}")
    print()

    print("packages:")
    for name in ["numpy", "cv2", "PIL", "torch", "ultralytics"]:
        print(f"  {name}: {package_status(name)}")
    print()

    if package_status("torch") == "ok":
        import torch
        print("torch:")
        print(f"  version: {getattr(torch, '__version__', 'unknown')}")
        cuda = getattr(torch, "cuda", None)
        cuda_available = bool(cuda and cuda.is_available())
        print(f"  cuda_available: {cuda_available}")
        if cuda_available:
            print(f"  cuda_device_count: {torch.cuda.device_count()}")
            print(f"  cuda_device: {torch.cuda.get_device_name(0)}")
            print(f"  cudnn_available: {torch.backends.cudnn.is_available()}")
            print(f"  cudnn_benchmark: {torch.backends.cudnn.benchmark}")
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends else None
        print(f"  mps_available: {bool(mps and mps.is_available())}")
        print()

    print("paths:")
    for path in PATHS:
        print(f"  {path}: {path_status(path)}")


if __name__ == "__main__":
    main()
