#!/usr/bin/env python3
"""Train a YOLO digit detector for handwriting videos."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/yolo/dataset.yaml", help="YOLO dataset yaml path")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda:0, mps, cpu, or leave empty")
    parser.add_argument("--project", default="runs/yolo_digits")
    parser.add_argument("--name", default="train")
    parser.add_argument("--out", default="models/yolo_digits.pt", help="Where to copy best.pt")
    return parser.parse_args()


def ultralytics_device(device: str | None) -> str | int | None:
    if not device:
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    device = ultralytics_device(args.device)
    if device is not None:
        train_kwargs["device"] = device

    result = model.train(**train_kwargs)
    elapsed = time.perf_counter() - start

    save_dir = Path(getattr(result, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, out)
        print(f"best_weights={out}")
    else:
        print(f"warning=best.pt_not_found expected={best}")
    print(f"total_time_sec={elapsed:.6f}")


if __name__ == "__main__":
    main()
