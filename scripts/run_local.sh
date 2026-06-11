#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_local.sh check
  scripts/run_local.sh train-mnist
  scripts/run_local.sh eval-pgm
  scripts/run_local.sh video-to-pgm
  scripts/run_local.sh video-pipeline

Common environment variables:
  PYTHON=python3
  DEVICE=cpu|cuda|mps
  CUSTOM_ROOT=data/custom_digits
  PGM_ROOT=data/pgm
  MAX_IMAGES=15
  MNIST_WEIGHTS=models/mnist_13568_local.pt
  YOLO_WEIGHTS=models/yolo_digits.pt
  VIDEO=data/videos/test.mp4

Training variables:
  EPOCHS=10
  CUSTOM_REPEAT=20
  BATCH_SIZE=128
  NUM_WORKERS=2
  LR=0.001

YOLO event variables:
  MAX_EVENTS=15
  EVENT_GAP_FRAMES=8
  CONF=0.35
  IMGSZ=640
EOF
}

device_args=()
if [[ -n "${DEVICE:-}" ]]; then
  device_args=(--device "$DEVICE")
fi

cmd="${1:-}"
case "$cmd" in
  check)
    "$PYTHON" scripts/check_environment.py
    ;;

  train-mnist)
    "$PYTHON" src/mnist/train_mnist_13568.py \
      --download \
      --data-dir "${DATA_DIR:-data/mnist}" \
      --custom-root "${CUSTOM_ROOT:-data/custom_digits}" \
      --custom-repeat "${CUSTOM_REPEAT:-20}" \
      --epochs "${EPOCHS:-10}" \
      --batch-size "${BATCH_SIZE:-128}" \
      --num-workers "${NUM_WORKERS:-2}" \
      --lr "${LR:-0.001}" \
      --model-out "${MNIST_WEIGHTS:-models/mnist_13568_local.pt}" \
      --metrics-out "${MNIST_METRICS:-results/mnist_metrics_local.json}" \
      "${device_args[@]}"
    ;;

  eval-pgm)
    eval_args=(
      --weights "${MNIST_WEIGHTS:-models/mnist_13568_local.pt}"
      --pgm-root "${PGM_ROOT:-data/pgm}"
      --metrics-out "${PGM_METRICS:-results/pgm_eval_local.json}"
    )
    if [[ "${MAX_IMAGES:-0}" -gt 0 ]]; then
      eval_args+=(--max-images "${MAX_IMAGES}")
    fi
    "$PYTHON" src/mnist/eval_pgm.py "${eval_args[@]}" "${device_args[@]}"
    ;;

  video-to-pgm)
    yolo_args=(
      --weights "${YOLO_WEIGHTS:-models/yolo_digits.pt}"
      --video "${VIDEO:-data/videos/test.mp4}"
      --out-dir "${PGM_ROOT:-data/pgm}"
      --csv-out "${YOLO_CSV:-results/video_detections_local.csv}"
      --clean-out-dir
      --event-gap-frames "${EVENT_GAP_FRAMES:-8}"
      --conf "${CONF:-0.35}"
      --imgsz "${IMGSZ:-640}"
    )
    if [[ "${MAX_EVENTS:-0}" -gt 0 ]]; then
      yolo_args+=(--max-events "${MAX_EVENTS}")
    fi
    "$PYTHON" src/yolo/video_to_pgm.py "${yolo_args[@]}" "${device_args[@]}"
    ;;

  video-pipeline)
    "$0" video-to-pgm
    "$0" eval-pgm
    ;;

  -h|--help|help|"")
    usage
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
