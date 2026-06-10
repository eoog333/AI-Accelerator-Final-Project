#!/usr/bin/env python3
"""Train MNIST classifier while measuring 1,3,5,6,8 accuracy and latency."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageOps, ImageStat
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from model import DigitCNN


DIGITS = (1, 3, 5, 6, 8)
MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def mnist_style_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L")
    mean = ImageStat.Stat(image).mean[0]
    if mean > 127.0:
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    image = image.point(lambda p: 255 if p > 48 else 0)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    width, height = image.size
    scale = 20.0 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    canvas.paste(image, offset)
    return canvas


def augment_image(image: Image.Image) -> Image.Image:
    angle = random.uniform(-8.0, 8.0)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    return ImageChops.offset(image, dx, dy)


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.1307) / 0.3081
    return torch.from_numpy(array).unsqueeze(0)


class MNISTIdxDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: str | Path, train: bool, download: bool, augment: bool) -> None:
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.train = train
        self.augment = augment
        if download:
            self.download()

        prefix = "train" if train else "t10k"
        images_path = self.raw / f"{prefix}-images-idx3-ubyte.gz"
        labels_path = self.raw / f"{prefix}-labels-idx1-ubyte.gz"
        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"MNIST files not found in {self.raw}. Run with --download once or copy MNIST IDX gzip files there."
            )
        images, labels = self.read_idx(images_path, labels_path)
        keep = np.isin(labels, np.array(DIGITS, dtype=np.uint8))
        self.images = images[keep]
        self.labels = labels[keep].astype(np.int64)

    def download(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        for filename, url in MNIST_URLS.items():
            target = self.raw / filename
            if target.exists():
                continue
            print(f"downloading={url}")
            urllib.request.urlretrieve(url, target)

    @staticmethod
    def read_idx(images_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
        with gzip.open(images_path, "rb") as f:
            magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid MNIST image file: {images_path}")
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
        with gzip.open(labels_path, "rb") as f:
            magic, count = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid MNIST label file: {labels_path}")
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return images, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(self.images[index], mode="L")
        if self.augment:
            image = augment_image(image)
        return to_tensor(image), int(self.labels[index])


class CustomDigitDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: list[tuple[Path, int]], augment: bool) -> None:
        self.samples = samples
        self.augment = augment

    @classmethod
    def from_root(cls, root: str | Path, augment: bool) -> "CustomDigitDataset":
        self.root = Path(root)
        samples: list[tuple[Path, int]] = []
        for digit in DIGITS:
            digit_dir = self.root / str(digit)
            if not digit_dir.exists():
                continue
            for ext in ("*.pgm", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
                samples.extend((path, digit) for path in sorted(digit_dir.glob(ext)))
        return cls(samples, augment=augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = mnist_style_image(Image.open(path))
        if self.augment:
            image = augment_image(image)
        return to_tensor(image), label


def split_custom_dataset(root: str | Path, val_ratio: float, seed: int) -> tuple[CustomDigitDataset, CustomDigitDataset]:
    full = CustomDigitDataset.from_root(root, augment=False)
    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    for digit in DIGITS:
        digit_samples = [sample for sample in full.samples if sample[1] == digit]
        rng.shuffle(digit_samples)
        val_count = max(1, round(len(digit_samples) * val_ratio)) if digit_samples else 0
        val_samples.extend(digit_samples[:val_count])
        train_samples.extend(digit_samples[val_count:])
    return CustomDigitDataset(train_samples, augment=True), CustomDigitDataset(val_samples, augment=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--custom-root", default="data/custom_digits")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--download", action="store_true", help="Download MNIST if missing")
    parser.add_argument("--custom-val-ratio", type=float, default=0.2)
    parser.add_argument("--model-out", default="models/mnist_13568.pt")
    parser.add_argument("--metrics-out", default="results/mnist_metrics.json")
    parser.add_argument("--limit-train", type=int, default=0, help="Optional quick-debug train limit")
    return parser.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def filter_digits(dataset: Dataset, digits: tuple[int, ...]) -> Subset:
    indices = [i for i, (_, label) in enumerate(dataset) if int(label) in digits]
    return Subset(dataset, indices)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    per_digit = {str(d): {"correct": 0, "total": 0, "accuracy": 0.0} for d in DIGITS}
    infer_time = 0.0

    with torch.no_grad():
        for images, labels in loader:
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

            pred = logits.argmax(dim=1)
            matches = pred.eq(labels)
            total += labels.numel()
            correct += int(matches.sum().item())
            for label, ok in zip(labels.cpu().tolist(), matches.cpu().tolist()):
                key = str(int(label))
                if key in per_digit:
                    per_digit[key]["total"] += 1
                    per_digit[key]["correct"] += int(ok)

    for stats in per_digit.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "per_digit": per_digit,
        "inference_time_sec": infer_time,
        "avg_latency_ms": infer_time * 1000.0 / total if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    total_start = time.perf_counter()

    train_set = MNISTIdxDataset(args.data_dir, train=True, download=args.download, augment=True)
    test_set = MNISTIdxDataset(args.data_dir, train=False, download=args.download, augment=False)
    custom_train_set, custom_eval_set = split_custom_dataset(args.custom_root, args.custom_val_ratio, args.seed)

    if args.limit_train > 0:
        train_set = Subset(train_set, list(range(min(args.limit_train, len(train_set)))))

    combined_train = ConcatDataset([train_set, custom_train_set]) if len(custom_train_set) else train_set
    train_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    custom_eval_loader = DataLoader(custom_eval_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DigitCNN(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_start = time.perf_counter()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)
        eval_stats = evaluate(model, test_loader, device)
        epoch_stats = {
            "epoch": epoch,
            "loss": running_loss / seen if seen else 0.0,
            "epoch_time_sec": time.perf_counter() - epoch_start,
            "test_accuracy": eval_stats["accuracy"],
            "per_digit": eval_stats["per_digit"],
        }
        history.append(epoch_stats)
        print(
            f"epoch={epoch} loss={epoch_stats['loss']:.6f} "
            f"test_accuracy={epoch_stats['test_accuracy']:.6f} "
            f"epoch_time_sec={epoch_stats['epoch_time_sec']:.6f}"
        )

    final_stats = evaluate(model, test_loader, device)
    custom_stats = evaluate(model, custom_eval_loader, device) if len(custom_eval_set) else None
    elapsed = time.perf_counter() - total_start

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "digits": DIGITS, "normalize": [(0.1307,), (0.3081,)]}, model_out)

    metrics = {
        "device": str(device),
        "digits": DIGITS,
        "mnist_train_samples": len(train_set),
        "custom_train_samples": len(custom_train_set),
        "custom_eval_samples": len(custom_eval_set),
        "total_train_samples": len(combined_train),
        "epochs": args.epochs,
        "total_time_sec": elapsed,
        "mnist_test": final_stats,
        "custom_eval": custom_stats,
        "history": history,
        "model": str(model_out),
    }
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"model={model_out}")
    print(f"metrics={metrics_out}")
    print(f"total_time_sec={elapsed:.6f}")
    print(f"mnist_avg_latency_ms={final_stats['avg_latency_ms']:.6f}")
    for digit, stats in final_stats["per_digit"].items():
        print(f"mnist_digit={digit} accuracy={stats['accuracy']:.6f} correct={stats['correct']} total={stats['total']}")
    if custom_stats:
        print(f"custom_accuracy={custom_stats['accuracy']:.6f}")
        print(f"custom_avg_latency_ms={custom_stats['avg_latency_ms']:.6f}")
        for digit, stats in custom_stats["per_digit"].items():
            if stats["total"]:
                print(f"custom_digit={digit} accuracy={stats['accuracy']:.6f} correct={stats['correct']} total={stats['total']}")


if __name__ == "__main__":
    main()
