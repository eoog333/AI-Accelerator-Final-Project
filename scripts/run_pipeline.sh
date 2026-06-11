#!/usr/bin/env bash
set -euo pipefail

YOLO_WEIGHTS="${YOLO_WEIGHTS:-models/yolo_digits.pt}"
VIDEO="${VIDEO:-data/videos/test.mp4}"
PGM_DIR="${PGM_DIR:-data/pgm}"
MNIST_WEIGHTS="${MNIST_WEIGHTS:-models/mnist_13568_lab.pt}"
MAX_EVENTS="${MAX_EVENTS:-0}"
MAX_IMAGES="${MAX_IMAGES:-0}"
EVENT_GAP_FRAMES="${EVENT_GAP_FRAMES:-8}"
MIN_EVENT_FRAMES="${MIN_EVENT_FRAMES:-1}"
CUSTOM_REPEAT="${CUSTOM_REPEAT:-20}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-data/mnist}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-1}"

device_args=()
if [[ -n "$DEVICE" ]]; then
  device_args=(--device "$DEVICE")
fi

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

python src/yolo/video_to_pgm.py "${yolo_args[@]}" "${device_args[@]}"

train_args=(
  --download
  --data-dir "$DATA_DIR"
  --custom-root data/custom_digits
  --custom-repeat "$CUSTOM_REPEAT"
  --epochs "${EPOCHS:-10}"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --model-out "$MNIST_WEIGHTS"
  --metrics-out results/mnist_metrics_lab.json
)

if [[ "$AMP" == "1" ]]; then
  train_args+=(--amp)
fi

python src/mnist/train_mnist_13568.py "${train_args[@]}" "${device_args[@]}"

eval_args=(
  --weights "$MNIST_WEIGHTS"
  --pgm-root "$PGM_DIR"
  --metrics-out results/pgm_eval_lab.json
)

if [[ "$MAX_IMAGES" -gt 0 ]]; then
  eval_args+=(--max-images "$MAX_IMAGES")
fi

python src/mnist/eval_pgm.py "${eval_args[@]}" "${device_args[@]}"
