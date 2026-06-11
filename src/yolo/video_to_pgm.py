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
    event_idx: int
    frame_idx: int
    timestamp_ms: float
    first_frame_idx: int
    last_frame_idx: int
    frames_seen: int
    class_id: int
    class_name: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    pgm_path: str


@dataclass
class ActiveEvent:
    event_idx: int
    class_id: int
    class_name: str
    first_frame_idx: int
    last_frame_idx: int
    frames_seen: int
    best_frame_idx: int
    best_timestamp_ms: float
    best_conf: float
    best_box: tuple[int, int, int, int]
    best_image: np.ndarray


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
    parser.add_argument("--event-iou", type=float, default=0.25, help="IoU threshold for continuing the same event")
    parser.add_argument(
        "--event-gap-frames",
        type=int,
        default=8,
        help="Close an event after this many processed frames without a matching detection",
    )
    parser.add_argument("--min-event-frames", type=int, default=1, help="Ignore events seen for fewer frames")
    parser.add_argument("--max-events", type=int, default=0, help="Save only the first N events; 0 means no limit")
    parser.add_argument(
        "--max-detections-per-frame",
        type=int,
        default=1,
        help="Use only the top N detections per frame; 0 means all detections",
    )
    parser.add_argument("--clean-out-dir", action="store_true", help="Remove existing .pgm files in out-dir first")
    return parser.parse_args()


def ultralytics_device(device: str | None) -> str | int | None:
    if not device:
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


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


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def class_name_from_result(result: object, class_id: int) -> str:
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def safe_label(label: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in label)
    return clean.strip("_") or "unknown"


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


def save_event(event: ActiveEvent, out_dir: Path, records: list[DetectionRecord], saved: int) -> int:
    label = safe_label(event.class_name)
    x1, y1, x2, y2 = event.best_box
    pgm_path = out_dir / (
        f"frame_{event.best_frame_idx:06d}_digit_{label}_conf_{event.best_conf:.2f}_{saved + 1:04d}.pgm"
    )
    write_pgm(pgm_path, event.best_image)
    records.append(
        DetectionRecord(
            event_idx=saved + 1,
            frame_idx=event.best_frame_idx,
            timestamp_ms=event.best_timestamp_ms,
            first_frame_idx=event.first_frame_idx,
            last_frame_idx=event.last_frame_idx,
            frames_seen=event.frames_seen,
            class_id=event.class_id,
            class_name=event.class_name,
            conf=event.best_conf,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            pgm_path=str(pgm_path),
        )
    )
    return saved + 1


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1:
        raise SystemExit("--sample-stride must be >= 1")
    if args.event_gap_frames < 0:
        raise SystemExit("--event-gap-frames must be >= 0")
    if args.min_event_frames < 1:
        raise SystemExit("--min-event-frames must be >= 1")
    if args.max_events < 0:
        raise SystemExit("--max-events must be >= 0")
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
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_out_dir:
        for path in out_dir.glob("*.pgm"):
            path.unlink()

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    records: list[DetectionRecord] = []
    active_events: list[ActiveEvent] = []
    frame_idx = 0
    next_event_idx = 1
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.sample_stride != 0:
            frame_idx += 1
            continue

        height, width = frame.shape[:2]
        timestamp_ms = (frame_idx / fps * 1000.0) if fps > 0 else 0.0
        still_active: list[ActiveEvent] = []
        for event in active_events:
            if frame_idx - event.last_frame_idx <= args.event_gap_frames:
                still_active.append(event)
            elif event.frames_seen >= args.min_event_frames and (args.max_events == 0 or saved < args.max_events):
                saved = save_event(event, out_dir, records, saved)
        active_events = still_active

        predict_kwargs = {"conf": args.conf, "imgsz": args.imgsz, "verbose": False}
        device = ultralytics_device(args.device)
        if device is not None:
            predict_kwargs["device"] = device
        result = model.predict(frame, **predict_kwargs)[0]

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)
            order = np.argsort(-confs)
            if args.max_detections_per_frame > 0:
                order = order[: args.max_detections_per_frame]

            matched_event_indices: set[int] = set()
            for det_order_idx in order:
                xyxy = boxes[det_order_idx]
                conf = float(confs[det_order_idx])
                class_id = int(class_ids[det_order_idx])
                class_name = class_name_from_result(result, class_id)
                x1, y1, x2, y2 = padded_box(*xyxy, width=width, height=height, pad=args.pad)
                box = (x1, y1, x2, y2)
                image = mnist_style_crop(frame, (x1, y1, x2, y2))

                best_match_idx = -1
                best_iou = 0.0
                for event_idx, event in enumerate(active_events):
                    if event_idx in matched_event_indices or event.class_id != class_id:
                        continue
                    iou = box_iou(event.best_box, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = event_idx

                if best_match_idx >= 0 and best_iou >= args.event_iou:
                    event = active_events[best_match_idx]
                    event.last_frame_idx = frame_idx
                    event.frames_seen += 1
                    if conf > event.best_conf:
                        event.best_frame_idx = frame_idx
                        event.best_timestamp_ms = timestamp_ms
                        event.best_conf = conf
                        event.best_box = box
                        event.best_image = image
                    matched_event_indices.add(best_match_idx)
                elif args.max_events == 0 or saved + len(active_events) < args.max_events:
                    active_events.append(
                        ActiveEvent(
                            event_idx=next_event_idx,
                            class_id=class_id,
                            class_name=class_name,
                            first_frame_idx=frame_idx,
                            last_frame_idx=frame_idx,
                            frames_seen=1,
                            best_frame_idx=frame_idx,
                            best_timestamp_ms=timestamp_ms,
                            best_conf=conf,
                            best_box=box,
                            best_image=image,
                        )
                    )
                    next_event_idx += 1

        frame_idx += 1

    cap.release()

    for event in active_events:
        if event.frames_seen >= args.min_event_frames and (args.max_events == 0 or saved < args.max_events):
            saved = save_event(event, out_dir, records, saved)

    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DetectionRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)

    elapsed = time.perf_counter() - start
    print(f"frames_read={frame_idx}")
    print(f"events_saved={saved}")
    print(f"detection_csv={csv_out}")
    print(f"total_time_sec={elapsed:.6f}")
    if saved:
        print(f"avg_time_per_saved_pgm_ms={elapsed * 1000.0 / saved:.6f}")


if __name__ == "__main__":
    main()
