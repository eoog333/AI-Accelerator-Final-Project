#!/usr/bin/env python3
"""Evaluate trained MNIST classifier on PGM files produced from video."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import DigitCNN


def to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("L").resize((28, 28))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.1307) / 0.3081
    return torch.from_numpy(array).unsqueeze(0)


class PGMEvalDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.samples: list[tuple[Path, int]] = []

        labeled = False
        for digit_dir in sorted(self.root.iterdir()) if self.root.exists() else []:
            if digit_dir.is_dir() and digit_dir.name.isdigit():
                labeled = True
                label = int(digit_dir.name)
                self.samples.extend((path, label) for path in sorted(digit_dir.glob("*.pgm")))

        if not labeled and self.root.exists():
            self.samples.extend((path, -1) for path in sorted(self.root.glob("*.pgm")))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        image = Image.open(path).convert("L")
        return to_tensor(image), label, str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pgm-root", default="data/pgm")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metrics-out", default="results/pgm_eval.json")
    parser.add_argument("--max-images", type=int, default=0, help="Evaluate only the first N PGM files; 0 means all")
    return parser.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.max_images < 0:
        raise SystemExit("--max-images must be >= 0")
    device = select_device(args.device)
    total_start = time.perf_counter()

    dataset = PGMEvalDataset(args.pgm_root)
    if args.max_images > 0:
        dataset.samples = dataset.samples[: args.max_images]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.weights, map_location=device)
    model = DigitCNN(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records = []
    total = 0
    correct = 0
    infer_time = 0.0
    pred_counts = {digit: 0 for digit in range(10)}
    per_digit: dict[str, dict[str, int | float]] = {}

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            infer_time += time.perf_counter() - start

            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            for path, label, prediction, confidence in zip(
                paths, labels.cpu().tolist(), pred.cpu().tolist(), probs.max(dim=1).values.cpu().tolist()
            ):
                pred_counts[int(prediction)] += 1
                is_labeled = int(label) >= 0
                ok = bool(is_labeled and int(label) == int(prediction))
                if is_labeled:
                    total += 1
                    correct += int(ok)
                    key = str(int(label))
                    per_digit.setdefault(key, {"correct": 0, "total": 0, "accuracy": 0.0})
                    per_digit[key]["total"] = int(per_digit[key]["total"]) + 1
                    per_digit[key]["correct"] = int(per_digit[key]["correct"]) + int(ok)
                records.append(
                    {
                        "path": path,
                        "label": int(label),
                        "prediction": int(prediction),
                        "confidence": float(confidence),
                        "correct": ok if is_labeled else None,
                    }
                )

    for stats in per_digit.values():
        total_digit = int(stats["total"])
        stats["accuracy"] = int(stats["correct"]) / total_digit if total_digit else 0.0

    elapsed = time.perf_counter() - total_start
    metrics = {
        "device": str(device),
        "pgm_root": args.pgm_root,
        "samples": len(dataset),
        "max_images": args.max_images,
        "labeled_samples": total,
        "accuracy": correct / total if total else None,
        "correct": correct,
        "total": total,
        "prediction_counts": pred_counts,
        "per_digit": per_digit,
        "inference_time_sec": infer_time,
        "avg_latency_ms": infer_time * 1000.0 / len(dataset) if len(dataset) else 0.0,
        "total_time_sec": elapsed,
        "records": records,
    }

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for record in records:
        print(f"INPUT: {Path(record['path']).name}")
        print(f"Result of classification: {record['prediction']}")
        print()

    print(f"Total Images : {len(dataset)}")
    print(f"Total Time   : {infer_time:.3f} sec")
    print()
    for digit in range(10):
        print(f"Digit {digit} : {pred_counts[digit]}")


if __name__ == "__main__":
    main()
