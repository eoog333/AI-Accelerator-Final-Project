#!/usr/bin/env bash
set -euo pipefail

YOLO_WEIGHTS="${YOLO_WEIGHTS:-models/yolo_digits.pt}"
VIDEO="${VIDEO:-data/videos/test.mp4}"
PGM_DIR="${PGM_DIR:-data/pgm}"
MNIST_WEIGHTS="${MNIST_WEIGHTS:-models/mnist_13568.pt}"
MAX_EVENTS="${MAX_EVENTS:-0}"
MAX_IMAGES="${MAX_IMAGES:-0}"
EVENT_GAP_FRAMES="${EVENT_GAP_FRAMES:-8}"
MIN_EVENT_FRAMES="${MIN_EVENT_FRAMES:-1}"
CUSTOM_REPEAT="${CUSTOM_REPEAT:-20}"

yolo_args=(
  --weights "$YOLO_WEIGHTS"
  --video "$VIDEO"
  --out-dir "$PGM_DIR"
  --csv-out results/video_detections.csv
  --event-gap-frames "$EVENT_GAP_FRAMES"
  --min-event-frames "$MIN_EVENT_FRAMES"
  --clean-out-dir
)

if [[ "$MAX_EVENTS" -gt 0 ]]; then
  yolo_args+=(--max-events "$MAX_EVENTS")
fi

python src/yolo/video_to_pgm.py "${yolo_args[@]}"

python src/mnist/train_mnist_13568.py \
  --custom-root data/custom_digits \
  --custom-repeat "$CUSTOM_REPEAT" \
  --epochs "${EPOCHS:-8}" \
  --model-out "$MNIST_WEIGHTS" \
  --metrics-out results/mnist_metrics.json

eval_args=(
  --weights "$MNIST_WEIGHTS"
  --pgm-root "$PGM_DIR"
  --metrics-out results/pgm_eval.json
)

if [[ "$MAX_IMAGES" -gt 0 ]]; then
  eval_args+=(--max-images "$MAX_IMAGES")
fi

python src/mnist/eval_pgm.py "${eval_args[@]}"
