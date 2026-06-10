#!/usr/bin/env bash
set -euo pipefail

YOLO_WEIGHTS="${YOLO_WEIGHTS:-models/yolo_digits.pt}"
VIDEO="${VIDEO:-data/videos/test.mp4}"
PGM_DIR="${PGM_DIR:-data/pgm}"
MNIST_WEIGHTS="${MNIST_WEIGHTS:-models/mnist_13568.pt}"

python src/yolo/video_to_pgm.py \
  --weights "$YOLO_WEIGHTS" \
  --video "$VIDEO" \
  --out-dir "$PGM_DIR" \
  --csv-out results/video_detections.csv

python src/mnist/train_mnist_13568.py \
  --custom-root data/custom_digits \
  --epochs "${EPOCHS:-8}" \
  --model-out "$MNIST_WEIGHTS" \
  --metrics-out results/mnist_metrics.json

python src/mnist/eval_pgm.py \
  --weights "$MNIST_WEIGHTS" \
  --pgm-root "$PGM_DIR" \
  --metrics-out results/pgm_eval.json

