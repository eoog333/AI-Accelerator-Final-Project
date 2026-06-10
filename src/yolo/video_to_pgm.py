#!/usr/bin/env python3
"""Detect digits in a video with YOLO and save 28x28 PGM crops."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionRecord:
    frame_idx: int
    timestamp_ms: float
    det_idx: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    pgm_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Trained YOLO weights")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out-dir", default="data/pgm", help="PGM output directory")
    parser.add_argument("--csv-out", default="results/video_detections.csv")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--sample-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--pad", type=float, default=0.15, help="BBox padding ratio")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def padded_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int, pad: float) -> tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad
    py = bh * pad
    return (
        clamp(int(round(x1 - px)), 0, width - 1),
        clamp(int(round(y1 - py)), 0, height - 1),
        clamp(int(round(x2 + px)), 1, width),
        clamp(int(round(y2 + py)), 1, height),
    )


def mnist_style_crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # MNIST convention is bright digit on dark background.
    if float(gray.mean()) > 127.0:
        gray = 255 - gray

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        binary = binary[y : y + h, x : x + w]

    h, w = binary.shape[:2]
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    xoff = (28 - new_w) // 2
    yoff = (28 - new_h) // 2
    canvas[yoff : yoff + new_h, xoff : xoff + new_w] = resized
    return canvas


def write_pgm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.shape != (28, 28):
        raise ValueError(f"PGM image must be 28x28, got {image.shape}")
    with path.open("wb") as f:
        f.write(b"P5\n28 28\n255\n")
        f.write(image.astype(np.uint8).tobytes())


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    out_dir = Path(args.out_dir)
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    records: list[DetectionRecord] = []
    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.sample_stride != 0:
            frame_idx += 1
            continue

        height, width = frame.shape[:2]
        predict_kwargs = {"conf": args.conf, "imgsz": args.imgsz, "verbose": False}
        if args.device:
            predict_kwargs["device"] = args.device
        result = model.predict(frame, **predict_kwargs)[0]

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for det_idx, (xyxy, conf) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = padded_box(*xyxy, width=width, height=height, pad=args.pad)
                image = mnist_style_crop(frame, (x1, y1, x2, y2))
                pgm_path = out_dir / f"frame_{frame_idx:06d}_det_{det_idx:02d}.pgm"
                write_pgm(pgm_path, image)
                saved += 1
                timestamp_ms = (frame_idx / fps * 1000.0) if fps > 0 else 0.0
                records.append(
                    DetectionRecord(frame_idx, timestamp_ms, det_idx, float(conf), x1, y1, x2, y2, str(pgm_path))
                )

        frame_idx += 1

    cap.release()

    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DetectionRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)

    elapsed = time.perf_counter() - start
    print(f"frames_read={frame_idx}")
    print(f"pgm_saved={saved}")
    print(f"detection_csv={csv_out}")
    print(f"total_time_sec={elapsed:.6f}")
    if saved:
        print(f"avg_time_per_saved_pgm_ms={elapsed * 1000.0 / saved:.6f}")


if __name__ == "__main__":
    main()

