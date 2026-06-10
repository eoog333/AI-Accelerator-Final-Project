#!/usr/bin/env python3
"""Convert collected 6/8 handwriting photos to MNIST-style 28x28 PGM files."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps, ImageStat


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".pgm")


def mnist_style_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L")

    # MNIST uses a bright digit on a dark background.
    if ImageStat.Stat(image).mean[0] > 127.0:
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


def iter_images(root: Path, digit: str) -> list[Path]:
    digit_dir = root / digit
    return sorted(path for path in digit_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default="data/custom_digits")
    parser.add_argument("--out-root", default="data/custom_digits_pgm")
    parser.add_argument("--digits", nargs="+", default=["6", "8"])
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    total = 0

    for digit in args.digits:
        paths = iter_images(src_root, digit)
        out_dir = out_root / digit
        out_dir.mkdir(parents=True, exist_ok=True)

        for index, path in enumerate(paths):
            image = mnist_style_image(Image.open(path))
            out_path = out_dir / f"{digit}_{index:04d}.pgm"
            image.save(out_path)

        total += len(paths)
        print(f"digit={digit} input={len(paths)} output_dir={out_dir}")

    print(f"total_preprocessed={total}")
    print(f"out_root={out_root}")


if __name__ == "__main__":
    main()

